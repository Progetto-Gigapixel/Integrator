class LensfunException(Exception):
    def __init__(self, message):
        self.message = message


class LensfunMissingDataException(LensfunException):
    def __init__(self, message):
        super().__init__(message)
