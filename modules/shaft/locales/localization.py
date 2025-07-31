import gettext


# Singleton class for language configuration
class LanguageConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LanguageConfig, cls).__new__(cls)
            # Initialization of the translator variable
            cls._instance._ = lambda x: x  # Default is English (identity function)
        return cls._instance  # Return the Singleton instance

    def set_language(self, lang="en"):
        try:
            # If the language is English, use the identity function
            if lang == "en":
                self._ = lambda x: x
            # Otherwise, load the corresponding translation file
            else:
                lan = gettext.translation(
                    "cocoa", localedir="locales", languages=[lang]
                )
                lan.install()
                self._ = lan.gettext
        except Exception as e:
            self._ = lambda x: x  # Revert to default if error occurs

    def __call__(self, text):
        return self._(text)  # Return the translated text


_ = LanguageConfig()  # Singleton for language configuration

# Testing
# _.set_language("it")
