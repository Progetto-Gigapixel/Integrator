class VignettingCorrectionException(Exception):
    def __init__(self, message):
        self.message = message


class VignettingCorrectionMissingDataException(VignettingCorrectionException):
    def __init__(self, message):
        super().__init__(message)
