class RawTherapeeException(Exception):
    def __init__(self, message, origin_class=None):
        if origin_class:
            message = f"{origin_class}: {message}"
        super().__init__(message)
        self.message = message

class RawTherapeeProcessException(Exception):
    def __init__(self, message, origin_class=None, return_code=None):
        if origin_class:
            message = f"{origin_class}: {message} /n Return code: {return_code}"
        super().__init__(message)
        self.message = message
