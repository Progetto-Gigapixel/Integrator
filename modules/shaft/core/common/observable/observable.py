class Observable:
    def __init__(self):
        """Initializes the observable object."""
        self._observers = []
        self._state = {}

    def subscribe(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def unsubscribe(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    def notify_observers(self):
        for observer in self._observers:
            observer.update(self._state)

    def get_state(self):
        return self._state

    def update(self, updates):
        self._state = updates
        self.notify_observers()


class Observer:
    def update(self, state):
        pass
