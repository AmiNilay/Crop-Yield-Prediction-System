import sys
import traceback
from typing import Optional


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Create a detailed error message with file name and line number."""
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error message: [{str(error)}]"

    file_name: str = exc_tb.tb_frame.f_code.co_filename
    line_number: int = exc_tb.tb_lineno
    return (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )


class CustomException(Exception):
    """Formatted exception that captures script file, line number, and message."""

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message: str = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        return self.error_message