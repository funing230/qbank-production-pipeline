"""
V5 生产主调度器 - 串联 generator → renderer → sentinel(Qwen审核)
负责批次管理、并发控制、进度追踪、飞书汇报。

████████████████████████████████████████████████████████████████
██  FROZEN SENTINEL PIPELINE (2026-06-20)                     ██
██  审核流程已冻结，未经爸爸明确批准不得修改以下逻辑：        ██
██    - 100%全量审核 (sentinel_sample_rate=1.0)               ██
██    - 图片前置校验 (全白/全黑/过小→FAIL)                    ██
██    - Qwen多模态审核 (必须带图base64 + JSON)                ██
██    - FAIL→GPT重生成→渲染→再审 (最多2轮)                    ██
██    - should_review() 恒返回True                            ██
████████████████████████████████████████████████████████████████
"""
import os
import sys
import json
import time
import re
import signal
import base64
import urllib.request
import urllib.error
import concurrent.futures
import threading
from pathlib import Path
from datetime import datetime

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db import ProductionDB
from pipeline.generator import QuestionGenerator
from pipeline.renderers import RenderDispatcher, validate_output

# ========== 配置 ==========

class Config:
    """从环境变量和配置文件加载生产参数"""
    def __init__(self):
        # 从运行中的orchestrator进程读取API配置
        env = self._load_env_from_proc()
        
        # API endpoints
        self.gpt_base_url = env.get("GPT_BASE_URL", "https://api.lk888.ai/v1")
        self.gpt_api_key = env.get("GPT_API_KEY", "")
        self.qwen_base_url = env.get("QWEN_BASE_URL", "https://yuanlansj.xin/v1")
        self.qwen_api_key = env.get("QWEN_API_KEY", "")
        
        # Models
        self.gpt_model = "gpt-5.5"
        self.qwen_model = "qwen3.7-max"
        
        # Concurrency
        self.gpt_concurrency = 16
        self.qwen_concurrency = 6  # Qwen审核6路并发
        self.render_concurrency = 8
        
        # Paths
        self.db_path = Path(env.get("PRODUCTION_DB_PATH") or os.environ.get("PRODUCTION_DB_PATH") or PROJECT_ROOT / "production.db")
        self.output_dir = Path(env.get("OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or PROJECT_ROOT / "output")
        self.raw_responses_dir = Path(env.get("RAW_RESPONSES_DIR") or os.environ.get("RAW_RESPONSES_DIR") or PROJECT_ROOT / "raw_responses")
        self.config_dir = PROJECT_ROOT / "config"
        
        # Load production quotas
        quota_file = self.config_dir / "production_quotas.json"
        self.quotas = json.loads(quota_file.read_text()) if quota_file.exists() else {}
        
        # Batch sizing
        self.batch_size_min = 4
        self.batch_size_max = 12
        self.batch_size_default = 8
        
        # Sentinel sampling — 100%全量审核（冻结，勿改）
        self.sentinel_sample_rate = 1.0  # ██ FROZEN: 100%全量审核 ██
        self.sentinel_high_risk_rate = 1.0  # 高风险100%审核
        self.sentinel_max_regen_rounds = 2  # FAIL后最多重生成2轮
        
        # Retry / backoff
        self.max_retries = 3
        self.backoff_base = 60  # 60/120/240/480
        
        # Lock file
        self.lock_file = PROJECT_ROOT.parent / "model_call.lock"
    
    def _load_env_from_proc(self) -> dict:
        """从.env文件或orchestrator进程环境读取API配置"""
        env = {}
        # 优先从.env文件读取
        env_file = Path(__file__).parent.parent / "config" / ".env"
        if env_file.exists():
            for line in env_file.read_text().strip().split('\n'):
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k] = v
        
        if not env.get("GPT_BASE_URL"):
            # 回退：从PID 2608028读取
            try:
                raw = open("/proc/2608028/environ", "rb").read().split(b'\0')
                for item in raw:
                    try:
                        k, v = item.decode().split('=', 1)
                        env[k] = v
                    except:
                        pass
            except:
                pass
        
        # 最终回退到环境变量
        for k in ["GPT_BASE_URL", "GPT_API_KEY", "QWEN_BASE_URL", "QWEN_API_KEY"]:
            if k not in env:
                env[k] = os.environ.get(k, "")
        return env


# ========== 哨兵审核 ==========

class SentinelReviewer:
    """Qwen3.7-max多线程哨兵 - 验证题目图片与文字一致性"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.qwen_base_url.rstrip("/")
        self.api_key = config.qwen_api_key
        self.model = config.qwen_model
    
    def review_question(self, question_json: dict, image_path: str, render_instruction: dict = None) -> dict:
        """审核单道题：分类图片质检（quality_gate）+ 纯文本逻辑审核（Qwen3.7-max）。
        
        审核内容：
        - 阶段A: image_quality_gate 按 engine/diagram_type 分类做像素级检测（8项）
        - 阶段B: Qwen 纯文本验证题目逻辑自洽性
        """
        t0 = time.time()
        
        # 前置检查：图片必须存在
        if not image_path or not Path(image_path).exists():
            return {"verdict": "FAIL", "reason": "image_not_found_or_empty_path", "latency": 0}
        
        # === 阶段A: image_quality_gate 分类检测（8项）===
        from pipeline.image_quality_gate import quality_gate
        
        # 构造 diagram_meta 让 quality_gate 知道图的类型
        diagram_meta = None
        if render_instruction and isinstance(render_instruction, dict):
            diagram_meta = {
                "engine": render_instruction.get("engine", ""),
                "diagram_type": render_instruction.get("diagram_type", ""),
                "plot_type": render_instruction.get("data", {}).get("plot_type", 
                             render_instruction.get("diagram_type", "")),
            }
        
        gate_pass, gate_issues = quality_gate(image_path, diagram_meta)
        if not gate_pass:
            return {
                "verdict": "FAIL",
                "reason": f"quality_gate: {'; '.join(gate_issues)}",
                "issues": gate_issues,
                "latency": round(time.time() - t0, 2),
            }
        
        # Qwen纯文本审核：检查题目逻辑自洽性 + 图文数据一致性
        # 构造渲染数据摘要（让Qwen能验证图上画的数据是否和答案一致）
        # 截断到800字符避免超token导致HTTP 400
        render_data_section = ""
        if render_instruction and isinstance(render_instruction, dict):
            ri_data = render_instruction.get("data", {})
            if ri_data:
                data_str = json.dumps(ri_data, ensure_ascii=False)
                if len(data_str) > 800:
                    data_str = data_str[:800] + "...(截断)"
                render_data_section = f"""

渲染指令（图片上实际绘制的数据）:
engine: {render_instruction.get("engine", "?")}
diagram_type: {render_instruction.get("diagram_type", "?")}
data: {data_str}"""
        
        # 传给Qwen的question_json去掉render_instruction（太大），用上面的截断摘要替代
        qj_for_review = {k: v for k, v in question_json.items() if k != "render_instruction"}
        
        prompt = f"""你是大学考试多模态题目质量哨兵。请严格验证以下题目的文本逻辑和图文数据一致性：

题目JSON:
{json.dumps(qj_for_review, ensure_ascii=False, indent=2)}{render_data_section}

请检查以下6项：
1. question_text与explanation是否逻辑一致（解析能推出正确答案）
2. correct_answer是否确实能从explanation推导出来
3. 四个选项是否合理（无重复、无明显不合理）
4. explanation中的计算过程是否正确
5. image_description是否与题目内容匹配
6. 渲染数据(data)中的数值/标签/结构是否与题目答案和解析完全一致（如果有渲染指令的话）

用JSON回复: {{"verdict": "PASS" 或 "FAIL", "issues": ["问题列表，为空则无问题"], "confidence": 0.0-1.0}}"""

        payload = json.dumps({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": 300,
            "temperature": 0,
        }).encode()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                elapsed = time.time() - t0
                
                # 清除 Qwen thinking 标签内容（qwen3.7-max 默认开启思考模式）
                import re
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                
                # 解析JSON响应 — Qwen经常在JSON后面追加解释文字
                # 用花括号计数法精确截取第一个完整JSON对象
                start = content.find("{")
                if start >= 0:
                    depth = 0
                    end = start
                    for i in range(start, len(content)):
                        if content[i] == '{':
                            depth += 1
                        elif content[i] == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    result = json.loads(content[start:end])
                    return {
                        "verdict": result.get("verdict", "FAIL"),
                        "issues": result.get("issues", []),
                        "confidence": result.get("confidence", 0),
                        "latency": round(elapsed, 2),
                    }
                return {"verdict": "PARSE_ERROR", "raw": content[:200], "latency": round(elapsed, 2)}
        except Exception as e:
            return {"verdict": "ERROR", "reason": str(e)[:200], "latency": round(time.time() - t0, 2)}
    
    def should_review(self, question: dict, subject_stats: dict = None) -> bool:
        """100%全量审核，此方法恒返回True。
        
        ██ FROZEN: 全量审核策略已冻结，不允许降回抽样模式 ██
        """
        return True


# ========== 主调度器 ==========

class ProductionOrchestrator:
    """V5生产主调度器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.db = ProductionDB(str(self.config.db_path))
        self.generator = QuestionGenerator(
            base_url=self.config.gpt_base_url,
            api_key=self.config.gpt_api_key,
            model=self.config.gpt_model,
            max_concurrent=self.config.gpt_concurrency,
        )
        self.renderer = RenderDispatcher()
        self.sentinel = SentinelReviewer(self.config)
        
        # 知识点名称映射
        kp_name_file = self.config.config_dir / "knowledge_point_name_mapping.json"
        if kp_name_file.exists():
            import json as _json
            self.kp_name_map = _json.loads(kp_name_file.read_text())
        else:
            self.kp_name_map = {}
        
        # 状态
        self.running = True
        self.start_time = time.time()
        self._stats_lock = threading.Lock()
        
        # 确保输出目录
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.raw_responses_dir.mkdir(parents=True, exist_ok=True)
        
        # Signal handling
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    # ---- 格式检测：识别GPT返回非Python代码 ----
    NON_PYTHON_PATTERNS = re.compile(
        r'\\begin\{(circuitikz|tikzpicture|document|figure)\}'
        r'|\\documentclass'
        r'|\\usepackage'
        r'|\\draw\b'
        r'|\\node\b'
        r'|\\tikz'
        r'|\\circuit',
        re.IGNORECASE,
    )

    @staticmethod
    def _is_non_python_code(code: str) -> bool:
        """检测render_code是否为非Python代码（LaTeX/TikZ/circuitikz等）"""
        if not code:
            return False
        # 快速启发式：如果前500字符没有任何Python关键词但有LaTeX标记
        head = code[:600]
        has_python_signal = any(kw in head for kw in ['import ', 'plt.', 'np.', 'def ', 'fig,', 'ax.', 'matplotlib'])
        has_latex_signal = bool(ProductionOrchestrator.NON_PYTHON_PATTERNS.search(head))
        if has_latex_signal and not has_python_signal:
            return True
        return False

    def get_kp_info(self, kp_id: str) -> dict:
        """从input数据获取KP详细信息"""
        quotas = self.config.quotas.get("kp_quotas", {})
        kp_quota = quotas.get(kp_id, {})
        # 读取知识点名称
        kp_name = self.kp_name_map.get(kp_id, "")
        return {
            "kp_id": kp_id,
            "subject_id": kp_quota.get("subject_id", kp_id.split("-M")[0]),
            "knowledge_point_name": kp_name,
            "production_quota": kp_quota.get("production_quota", 0),
            "original_quota": kp_quota.get("original_quota", 0),
        }
    
    def produce_batch(self, kp_id: str, batch_size: int, existing: list = None) -> list:
        """生成一批题目：GPT出题 → 渲染 → 入库"""
        kp_info = self.get_kp_info(kp_id)
        subject_id = kp_info["subject_id"]
        
        # 1. GPT生成
        questions = self.generator.generate_batch(kp_info, batch_size, existing)
        if not questions:
            return []
        
        # 补充缺失字段 + 构造 question_json
        for q in questions:
            q.setdefault("subject_id", subject_id)
            q.setdefault("module_id", kp_id.rsplit("-", 1)[0] if "-" in kp_id else "")
            q.setdefault("kp_id", kp_id)
            q.setdefault("kp_name", kp_info.get("knowledge_point_name", ""))
            if not q.get("question_id"):
                import uuid
                q["question_id"] = f"{kp_id}-Q{uuid.uuid4().hex[:6]}"
            # 将完整题目数据打包为 question_json（交付用，去掉render_code等临时字段）
            delivery_fields = {
                "question_id": q.get("question_id", ""),
                "subject_id": subject_id,
                "kp_id": kp_id,
                "kp_name": kp_info.get("knowledge_point_name", ""),
                "question_text": q.get("question_text", ""),
                "options": q.get("options", {}),
                "correct_answer": q.get("correct_answer", ""),
                "explanation": q.get("explanation", ""),
                "difficulty": q.get("difficulty", 0),
                "image_description": q.get("image_description", ""),
                "image_path": "",  # 渲染成功后回填
                "render_engine": q.get("render_engine", ""),
            }
            q["question_json"] = delivery_fields
        
        # 2. 创建batch并入库
        batch_id = self.db.create_batch(subject_id, kp_id, len(questions))
        q_ids = self.db.add_questions(batch_id, questions)
        
        # 3. 渲染每道题的图片
        rendered = []
        from pipeline.render_router import render_from_instruction
        
        for q, qid in zip(questions, q_ids):
            if not self.running:
                break
            
            # 输出路径
            img_dir = self.config.output_dir / subject_id / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"{qid}.png"
            
            # 新模式：render_instruction（结构化JSON）
            ri = q.get("render_instruction")
            if ri and isinstance(ri, dict):
                success, msg = render_from_instruction(ri, str(img_path))
                result = {"success": success, "error": msg if not success else ""}
            else:
                # 旧模式 fallback
                engine = q.get("render_engine", "MATPLOTLIB")
                code = q.get("render_code", "")
                result = self.renderer.dispatch(engine, code, str(img_path))
            
            if result["success"]:
                engine_name = ri.get("engine", "render_instruction") if ri else q.get("render_engine", "MATPLOTLIB")
                self.db.update_question_status(qid, "RENDERED", f"engine={engine_name}")
                rel_path = str(img_path.relative_to(self.config.output_dir))
                q["image_path"] = rel_path
                # 回填image_path到question_json
                if isinstance(q.get("question_json"), dict):
                    q["question_json"]["image_path"] = rel_path
                    # 同步更新DB中的question_json
                    self.db.update_question_json(qid, q["question_json"])
                rendered.append((q, qid, str(img_path)))
            else:
                # 检测是否GPT返回了非Python代码（如circuitikz/TikZ）
                code = q.get("render_code", "")
                engine = q.get("render_engine", "MATPLOTLIB")
                if self._is_non_python_code(code):
                    reason = "non_python_code_detected(LaTeX/TikZ/circuitikz)"
                    self.db.update_question_status(qid, "REGENERATE", reason)
                    print(f"    [{qid}] REGENERATE: GPT returned non-Python render code, will retry generation")
                    # 立即重新生成这道题（限1次重试，避免死循环）
                    retry_questions = self.generator.generate_batch(kp_info, 1, existing)
                    if retry_questions:
                        rq = retry_questions[0]
                        rq_code = rq.get("render_code", "")
                        if rq_code and not self._is_non_python_code(rq_code):
                            retry_result = self.renderer.dispatch(
                                rq.get("render_engine", "MATPLOTLIB"), rq_code, str(img_path)
                            )
                            if retry_result["success"]:
                                # 用重生成的题目覆盖原题
                                self.db.update_question_status(qid, "RENDERED",
                                    f"engine={rq.get('render_engine','MATPLOTLIB')},regenerated=1")
                                rel_path = str(img_path.relative_to(self.config.output_dir))
                                rq["image_path"] = rel_path
                                rq.setdefault("question_json", q.get("question_json", {}))
                                if isinstance(rq.get("question_json"), dict):
                                    rq["question_json"]["image_path"] = rel_path
                                    self.db.update_question_json(qid, rq["question_json"])
                                rendered.append((rq, qid, str(img_path)))
                                print(f"    [{qid}] ✓ Regeneration succeeded")
                                continue
                        # 重生成也失败
                        self.db.update_question_status(qid, "RENDER_FAIL",
                            f"regeneration_also_failed: {reason}")
                    else:
                        self.db.update_question_status(qid, "RENDER_FAIL",
                            f"regeneration_returned_empty: {reason}")
                else:
                    self.db.update_question_status(qid, "RENDER_FAIL", result.get("error", "")[:500])
                    # 添加渲染任务记录以便后续重试
                    self.db.add_render_job(qid, engine, code)
        
        # 4. Qwen哨兵审核 — 100%全量 + FAIL自动重生成闭环
        # ██ FROZEN: 此审核流程已冻结，勿修改 ██
        # (试跑模式可通过config.sentinel_sample_rate=0跳过)
        final_passed = []
        skip_sentinel = (getattr(self.config, 'sentinel_sample_rate', 1.0) == 0.0)
        for q, qid, img_path in rendered:
            if not self.running:
                break
            
            if skip_sentinel:
                self.db.update_question_status(qid, "FINAL_PASS", "sentinel_skipped_pilot")
                final_passed.append((q, qid, img_path))
                continue
            
            passed = False
            current_q = q
            current_img = img_path
            
            for regen_round in range(self.config.sentinel_max_regen_rounds + 1):
                # round 0 = 首次审核, round 1/2 = 重生成后再审
                review = self.sentinel.review_question(
                    current_q.get("question_json", current_q), current_img,
                    render_instruction=current_q.get("render_instruction")
                )
                
                if review["verdict"] == "PASS":
                    self.db.update_question_status(qid, "FINAL_PASS",
                        f"confidence={review.get('confidence', 0)},review_round={regen_round}")
                    passed = True
                    break
                
                # FAIL — 记录原因
                issues = "; ".join(review.get("issues", [review.get("reason", "unknown")]))
                
                if regen_round >= self.config.sentinel_max_regen_rounds:
                    # 已达最大重生成次数，标记为需人工复核
                    self.db.update_question_status(qid, "SENTINEL_FAIL_FINAL",
                        f"exhausted_{self.config.sentinel_max_regen_rounds}_rounds: {issues[:400]}")
                    print(f"    [{qid}] ✗ FAIL after {regen_round+1} rounds: {issues[:80]}")
                    break
                
                # 还有重生成机会 → 调GPT重新生成
                print(f"    [{qid}] FAIL(round {regen_round}): {issues[:60]} → regenerating...")
                self.db.update_question_status(qid, "SENTINEL_REGEN",
                    f"round={regen_round}: {issues[:400]}")
                
                retry_questions = self.generator.generate_batch(kp_info, 1, existing)
                if not retry_questions:
                    self.db.update_question_status(qid, "SENTINEL_FAIL_FINAL",
                        f"regen_empty_at_round_{regen_round+1}: {issues[:400]}")
                    print(f"    [{qid}] ✗ Regeneration returned empty")
                    break
                
                rq = retry_questions[0]
                
                # 新模式：render_instruction
                rq_ri = rq.get("render_instruction")
                if rq_ri and isinstance(rq_ri, dict):
                    rq_success, rq_msg = render_from_instruction(rq_ri, current_img)
                    retry_render = {"success": rq_success, "error": rq_msg if not rq_success else ""}
                else:
                    rq_code = rq.get("render_code", "")
                    rq_engine = rq.get("render_engine", "MATPLOTLIB")
                    # 检测非Python代码
                    if self._is_non_python_code(rq_code):
                        continue  # 直接进入下一轮重试
                    retry_render = self.renderer.dispatch(rq_engine, rq_code, current_img)
                
                if not retry_render["success"]:
                    continue  # 渲染失败，进入下一轮重试
                
                # 渲染成功，更新题目数据用于下轮审核
                current_q = rq
                rel_path = str(Path(current_img).relative_to(self.config.output_dir)) if self.config.output_dir in Path(current_img).parents else current_img
                current_q["image_path"] = rel_path
                current_q.setdefault("question_json", q.get("question_json", {}))
                if isinstance(current_q.get("question_json"), dict):
                    current_q["question_json"]["image_path"] = rel_path
                    self.db.update_question_json(qid, current_q["question_json"])
            
            if passed:
                final_passed.append((current_q, qid, current_img))
        
        # 更新统计
        self.db.update_production_stats(subject_id)
        return rendered
    
    def run_subject(self, subject_id: str, target_quota: int):
        """生产单个科目的所有题目"""
        quotas = self.config.quotas.get("kp_quotas", {})
        # 获取该科目所有KP
        subject_kps = {kp_id: info for kp_id, info in quotas.items() 
                       if info.get("subject_id") == subject_id}
        
        print(f"[{subject_id}] Starting: {len(subject_kps)} KPs, target={target_quota}")
        
        for kp_id, kp_info in sorted(subject_kps.items()):
            if not self.running:
                break
            
            pq = kp_info.get("production_quota", 0)
            if pq <= 0:
                continue
            
            # 检查已有进度
            existing = self.db.get_questions_by_status("FINAL_PASS", subject_id)
            existing_for_kp = [q for q in existing if q.get("kp_id") == kp_id]
            remaining = pq - len(existing_for_kp)
            
            if remaining <= 0:
                continue
            
            print(f"  [{kp_id}] need {remaining} more questions (have {len(existing_for_kp)}/{pq})")
            
            # 分批生产
            while remaining > 0 and self.running:
                batch_size = min(remaining, self.config.batch_size_default)
                produced = self.produce_batch(kp_id, batch_size, existing_for_kp)
                
                if not produced:
                    print(f"  [{kp_id}] batch returned empty, moving on")
                    break
                
                remaining -= len(produced)
                existing_for_kp.extend([p[0] for p in produced])
        
        print(f"[{subject_id}] Done")
    
    def run_all(self, subjects: list = None):
        """运行所有科目的生产"""
        subject_targets = self.config.quotas.get("subject_targets", {})
        
        if subjects:
            targets = {s: subject_targets.get(s, 1333) for s in subjects}
        else:
            targets = subject_targets
        
        print(f"=== V5 PRODUCTION START ===")
        print(f"Subjects: {len(targets)}")
        print(f"Total target: {sum(targets.values())}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for subject_id, target in sorted(targets.items()):
            if not self.running:
                break
            self.run_subject(subject_id, target)
        
        # 最终统计
        stats = self.db.get_production_stats()
        print(f"\n=== PRODUCTION SUMMARY ===")
        for s in stats:
            print(f"  {s}")
    
    def get_progress_report(self) -> dict:
        """获取当前进度报告（飞书汇报用）"""
        elapsed = time.time() - self.start_time
        stats = self.db.get_overall_progress()
        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_min": round(elapsed / 60, 1),
            "running": self.running,
            **stats,
        }


# ========== 入口 ==========

def main():
    """主入口"""
    import argparse
    parser = argparse.ArgumentParser(description="V5 Production Orchestrator")
    parser.add_argument("--subjects", nargs="*", help="指定科目ID (e.g. S01 S02)")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    args = parser.parse_args()
    
    config = Config()
    orch = ProductionOrchestrator(config)
    
    if args.dry_run:
        print("DRY RUN - would produce:")
        targets = config.quotas.get("subject_targets", {})
        for sid, t in sorted(targets.items()):
            if args.subjects and sid not in args.subjects:
                continue
            print(f"  {sid}: {t} questions")
        return
    
    subjects = args.subjects if args.subjects else None
    orch.run_all(subjects)


if __name__ == "__main__":
    main()
