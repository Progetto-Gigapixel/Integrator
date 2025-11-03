class StartupCheckerException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class LensfunMissingData(StartupCheckerException):
    def __init__(self, message):
        super().__init__(message)


class LensfunMissingCamera(LensfunMissingData):
    def __init__(self, message):
        super().__init__(message)


class LensfunMissingLens(LensfunMissingData):
    def __init__(self, message):
        super().__init__(message)
