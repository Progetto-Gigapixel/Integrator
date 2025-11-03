class DecodeRawException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class DecodeRawIOException(DecodeRawException):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class DecodeRawIpnutException(DecodeRawException):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
