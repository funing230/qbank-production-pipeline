# renderers package
# 兼容桩：orchestrator.py 仍然导入 RenderDispatcher 和 validate_output
# 实际渲染已走 render_router.py，这里只是避免 ImportError


class RenderDispatcher:
    """兼容旧代码的渲染分发器（实际渲染走 render_router）"""

    def dispatch(self, engine: str, code: str, output_path: str):
        """
        旧接口：接收 engine + code，调用子进程渲染。
        现在重定向到 render_executor（保底路径）。
        """
        from pipeline.render_executor import render_question_image
        success, error_msg = render_question_image(
            render_code=code,
            output_path=output_path,
            render_engine=engine,
            timeout=60,
        )
        return {"success": success, "error": error_msg, "path": output_path if success else None}


def validate_output(output_path: str) -> tuple[bool, str]:
    """兼容旧代码的输出验证（实际走 quality_gate）"""
    import os
    if not os.path.exists(output_path):
        return False, "文件不存在"
    if os.path.getsize(output_path) < 1000:
        return False, "文件过小"
    return True, ""
