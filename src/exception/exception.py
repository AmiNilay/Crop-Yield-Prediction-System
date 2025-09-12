import sys
from typing import Any  # Import 'Any' from the typing module

def error_message_detail(error: Exception, error_detail: Any) -> str:
    """
    Creates a detailed error message with file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f"Error message: [{str(error)}]"
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error message [{str(error)}]"
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class that formats the error message.
    """
    def __init__(self, error_message: Exception, error_detail: Any):
        # We pass the original error message to the parent Exception class
        super().__init__(error_message)
        # We create our custom, more detailed error message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message