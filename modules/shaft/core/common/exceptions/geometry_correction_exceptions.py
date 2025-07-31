class GeometricCorrectionException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GeometricCorrectionMissingDataException(GeometricCorrectionException):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class GeometricCorrectionManualException(GeometricCorrectionException):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
