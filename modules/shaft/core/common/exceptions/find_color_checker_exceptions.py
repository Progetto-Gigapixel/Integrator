class FindColorCheckerException(Exception):
    def __init__(self, message):
        super().__init__(message)


class AutoFindColorCheckerException(FindColorCheckerException):
    def __init__(self, message):
        super().__init__(message)
