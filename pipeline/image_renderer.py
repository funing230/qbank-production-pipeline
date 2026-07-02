"""
gpt-image-2 renderer for image_prompt pipeline.

Takes GPT-generated image_prompt, appends the frozen style suffix, calls either
an OpenAI-compatible image generation endpoint or the lk888 async media API,
and writes the image to disk.
"""
import base64
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Tuple

from pipeline.generator import QuestionGenerator


class GPTImageRenderer:
    RETRY_STATUS_CODES = {429, 502, 503, 504}
    LK888_PENDING_STATUSES = {
        "pending", "queued", "running", "processing", "generating", "created", "submitted",
        "处理中", "进行中", "排队中", "已提交", "生成中",
    }
    LK888_FAILED_STATUSES = {"failed", "fail", "error", "cancelled", "canceled", "timeout", "失败", "错误", "已取消"}

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        size: str = None,
        provider: str = None,
        name: str = "primary",
    ):
        self.name = name
        self.base_url = self._normalize_base_url(base_url or os.environ.get("GPT_IMAGE_BASE_URL") or os.environ.get("GPT5_BASE_URL") or "https://api.lk888.ai/v1")
        self.api_key = api_key or os.environ.get("GPT_IMAGE_API_KEY") or os.environ.get("OPENAI_IMAGE_API_KEY") or os.environ.get("GPT5_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("GPT_IMAGE_MODEL", "gpt-image-2")
        self.size = size or os.environ.get("GPT_IMAGE_SIZE", "1024x1024")
        self.provider = (provider or os.environ.get("GPT_IMAGE_PROVIDER") or self._infer_provider(self.base_url)).lower()
        self.lk888_poll_interval = float(os.environ.get("GPT_IMAGE_LK888_POLL_INTERVAL", "5"))
        self.lk888_timeout = float(os.environ.get("GPT_IMAGE_LK888_TIMEOUT", "240"))
        self.lk888_result_path = os.environ.get("GPT_IMAGE_LK888_RESULT_PATH", "").strip()
        self.fallback_base_url = (os.environ.get("GPT_IMAGE_FALLBACK_BASE_URL") or "https://tu.go2api.cc/v1").rstrip("/")
        self.fallback_api_key = os.environ.get("GO2API_IMAGE_API_KEY") or os.environ.get("GPT_IMAGE_FALLBACK_API_KEY", "")
        self.fallback_model = os.environ.get("GPT_IMAGE_FALLBACK_MODEL", "gpt-image-2-medium")
        self.fallback_provider = (os.environ.get("GPT_IMAGE_FALLBACK_PROVIDER") or self._infer_provider(self.fallback_base_url)).lower()
        self.enable_fallback = os.environ.get("GPT_IMAGE_ENABLE_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
        self.max_retries = int(os.environ.get("GPT_IMAGE_RETRY_MAX", "3"))
        self.fallback_concurrency = int(os.environ.get("GPT_IMAGE_FALLBACK_CONCURRENCY", "4"))
        self._fallback_semaphore = threading.BoundedSemaphore(self.fallback_concurrency)
        self._fallback_active = threading.Event()
        self._fallback_reason = ""
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        url = (base_url or "").rstrip("/")
        if url.endswith("/media/generate"):
            return url[: -len("/media/generate")]
        return url

    @staticmethod
    def _infer_provider(base_url: str) -> str:
        return "lk888" if "lk888.ai" in (base_url or "") else "openai"

    @property
    def fallback_active(self) -> bool:
        return self._fallback_active.is_set()

    @property
    def fallback_reason(self) -> str:
        with self._lock:
            return self._fallback_reason

    def render(self, image_prompt: str, output_path: str) -> Tuple[bool, str]:
        final_prompt = QuestionGenerator.assemble_final_image_prompt(image_prompt)
        if not image_prompt or not image_prompt.strip():
            return False, "empty image_prompt"

        providers = []
        if not self.fallback_active:
            providers.append(("primary", self.base_url, self.api_key, self.model, self.provider))
        if self.enable_fallback:
            providers.append(("fallback", self.fallback_base_url, self.fallback_api_key, self.fallback_model, self.fallback_provider))

        errors = []
        for provider_name, base_url, api_key, model, provider_type in providers:
            if not api_key:
                errors.append(f"{provider_name}: missing api key")
                continue
            if provider_name == "fallback":
                with self._fallback_semaphore:
                    ok, msg = self._render_with_retries(provider_name, base_url, api_key, model, final_prompt, output_path, provider_type)
            else:
                ok, msg = self._render_with_retries(provider_name, base_url, api_key, model, final_prompt, output_path, provider_type)
            if ok:
                return True, msg
            errors.append(msg)
            if provider_name == "primary" and self.enable_fallback:
                self._activate_fallback(msg)
        return False, " | ".join(errors)[-500:]

    def _activate_fallback(self, reason: str):
        with self._lock:
            if not self._fallback_active.is_set():
                self._fallback_reason = reason[:240]
                self._fallback_active.set()

    def _render_with_retries(self, provider_name: str, base_url: str, api_key: str, model: str,
                             prompt: str, output_path: str, provider_type: str) -> Tuple[bool, str]:
        last_msg = ""
        for attempt in range(self.max_retries + 1):
            if provider_type == "lk888":
                ok, msg, retryable = self._render_lk888_once(provider_name, base_url, api_key, model, prompt, output_path)
            else:
                ok, msg, retryable = self._render_openai_once(provider_name, base_url, api_key, model, prompt, output_path)
            if ok:
                suffix = " fallback_active" if provider_name == "fallback" else ""
                return True, f"{provider_name} image generated{suffix}: {msg}"
            last_msg = msg
            if not retryable or attempt >= self.max_retries:
                break
            time.sleep(min(60, 5 * (2 ** attempt)))
        return False, f"{provider_name} image API error: {last_msg[:300]}"

    def _render_openai_once(self, provider_name: str, base_url: str, api_key: str, model: str,
                            prompt: str, output_path: str) -> Tuple[bool, str, bool]:
        payload = {"model": model, "prompt": prompt, "size": self.size, "n": 1}
        try:
            body = self._json_request(f"{base_url}/images/generations", api_key, payload, timeout=120)
            ok, msg = self._write_image_from_response(body, Path(output_path))
            if ok:
                return True, msg, False
            return False, f"{provider_name} returned no image: {msg}; body={str(body)[:240]}", False
        except urllib.error.HTTPError as exc:
            return self._http_error_result(exc)
        except Exception as exc:
            return False, str(exc)[:300], False

    def _render_lk888_once(self, provider_name: str, base_url: str, api_key: str, model: str,
                           prompt: str, output_path: str) -> Tuple[bool, str, bool]:
        payload = {"model": model, "prompt": prompt, "size": self.size, "n": 1}
        try:
            body = self._json_request(f"{base_url}/media/generate", api_key, payload, timeout=120)
            task_id = self._extract_task_id(body)
            if not task_id:
                ok, msg = self._write_image_from_response(body, Path(output_path))
                if ok:
                    return True, msg, False
                return False, f"{provider_name} returned no task_id/image: {str(body)[:240]}", False
            return self._poll_lk888_result(base_url, api_key, task_id, Path(output_path))
        except urllib.error.HTTPError as exc:
            return self._http_error_result(exc)
        except Exception as exc:
            return False, str(exc)[:300], False

    def _poll_lk888_result(self, base_url: str, api_key: str, task_id: str, output: Path) -> Tuple[bool, str, bool]:
        deadline = time.time() + self.lk888_timeout
        last_body: Any = None
        while time.time() < deadline:
            for url in self._lk888_result_urls(base_url, task_id):
                try:
                    body = self._json_get(url, api_key, timeout=60)
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        continue
                    return self._http_error_result(exc)
                last_body = body
                ok, msg = self._write_image_from_response(body, output)
                if ok:
                    return True, f"task_id={task_id}; {msg}", False
                # Authoritative terminal signal: trust is_final (per lk888 docs),
                # not a status-text whitelist. lk888 may change the Chinese status
                # wording (e.g. 进行中 -> 运行中) without notice; relying on the
                # whitelist caused premature abandonment. is_final=true means the
                # task ended (success or failure); only then do we give up.
                is_final = self._first_value(body, ("is_final",))
                status = str(self._first_value(body, ("status", "state", "task_status", "job_status")) or "").lower()
                if is_final is True or (isinstance(is_final, str) and is_final.lower() in {"true", "1", "yes"}):
                    return False, f"task_id={task_id} final without image (status={status}): {str(body)[:240]}", False
                if is_final is None and status in self.LK888_FAILED_STATUSES:
                    # Fallback only when is_final is absent from the response.
                    return False, f"task_id={task_id} failed: {str(body)[:240]}", False
                # Not final yet -> keep polling until deadline.
            time.sleep(self.lk888_poll_interval)
        return False, f"task_id={task_id} timed out waiting for image; last={str(last_body)[:240]}", True

    def _lk888_result_urls(self, base_url: str, task_id: str):
        quoted = urllib.parse.quote(str(task_id), safe="")
        if self.lk888_result_path:
            path = self.lk888_result_path.format(task_id=quoted)
            yield f"{base_url}/{path.lstrip('/')}"
            return
        candidates = [
            f"skills/task-status?task_id={quoted}",
            f"media/status?task_id={quoted}",
            f"media/generate/{quoted}",
            f"media/generations/{quoted}",
            f"media/task/{quoted}",
            f"media/tasks/{quoted}",
            f"media/result/{quoted}",
            f"media/results/{quoted}",
            f"media/status/{quoted}",
            f"media/status?task_id={quoted}",
            f"media/result?task_id={quoted}",
            f"media/results?task_id={quoted}",
            f"media/task?task_id={quoted}",
            f"media/tasks?task_id={quoted}",
        ]
        for path in candidates:
            yield f"{base_url}/{path}"

    def _json_request(self, url: str, api_key: str, payload: dict, timeout: int) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _json_get(self, url: str, api_key: str, timeout: int) -> Any:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _write_image_from_response(self, body: Any, output: Path) -> Tuple[bool, str]:
        b64 = self._first_value(body, ("b64_json", "base64", "image_base64"))
        if b64:
            if isinstance(b64, str) and b64.startswith("data:image"):
                b64 = b64.split(",", 1)[1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(base64.b64decode(b64))
            return self._verify_output(output)

        url = self._first_value(body, ("url", "image_url", "output_url", "file_url", "result_url"))
        if url:
            output.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(str(url), timeout=120) as img_resp:
                output.write_bytes(img_resp.read())
            return self._verify_output(output)
        return False, "no b64_json/url found"

    def _extract_task_id(self, body: Any) -> str:
        value = self._first_value(body, ("task_id", "taskId", "id", "job_id", "jobId"))
        return str(value) if value else ""

    def _first_value(self, obj: Any, keys: tuple) -> Any:
        if isinstance(obj, dict):
            for key in keys:
                value = obj.get(key)
                if value:
                    return value
            for value in obj.values():
                found = self._first_value(value, keys)
                if found:
                    return found
        elif isinstance(obj, list):
            for value in obj:
                found = self._first_value(value, keys)
                if found:
                    return found
        return None

    def _http_error_result(self, exc: urllib.error.HTTPError) -> Tuple[bool, str, bool]:
        detail = exc.read().decode("utf-8", errors="replace")[:240]
        return False, f"HTTP {exc.code}: {detail}", exc.code in self.RETRY_STATUS_CODES

    @staticmethod
    def _verify_output(path: Path) -> Tuple[bool, str]:
        if not path.exists():
            return False, "image file not created"
        size = path.stat().st_size
        if size < 1024:
            return False, f"image file too small: {size} bytes"
        return True, f"{size} bytes"
