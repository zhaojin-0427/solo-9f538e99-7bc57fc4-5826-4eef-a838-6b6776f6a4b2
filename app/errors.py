"""跨模块共享的整批失败异常。"""

from __future__ import annotations


class PreparationError(Exception):
    """整批制备/诊断失败；status_code 决定 HTTP 响应码。"""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
