class EquipmentException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class EquipmentTransferException(EquipmentException):
    def __init__(self, message):
        super().__init__(message)


class EquipmentFileNotRawException(EquipmentException):
    def __init__(self, message):
        super().__init__(message)


class EquipmentFileNotFoundException(EquipmentException):
    def __init__(self, message):
        super().__init__(message)
