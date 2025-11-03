from core.common.observable.observable import Observable


class AppStore(Observable):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppStore, cls).__new__(cls)
            cls._instance.state = {}
        return cls._instance

    def set(self, key, value):
        self.update(value)
        self.state[key] = value

    def get(self, key):
        return self.state.get(key)


appStore = AppStore()
