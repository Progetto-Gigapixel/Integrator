
import copy

from components.StitchPoint import Point


class StateManager:
    def __init__(self, initial_state: list[list[Point]] = [[], []]):
        """
        Manages state changes and provides undo/redo functionality.
        """
        if len(initial_state) != 2:
            raise ValueError("Initial state must be a matrix with exactly 2 rows")
        self.current_state=[[],[]]
        self.current_state = copy.deepcopy(initial_state)

        # History stack for undo
        self.history = []

        # Redo stack (optional)
        self.redo_stack = []
    def update_state(self, value):
        """
        Update the state and save the current state to history.
        """
        if self.current_state != [[], []]:
            self._save_state()
        self.current_state = copy.deepcopy(value)

    def _save_state(self):
        """
        Save the current state to the history stack.
        """
        self.history.append(copy.deepcopy(self.current_state))
        self.redo_stack.clear()  # Clear redo stack when a new action is performed
        if len(self.history) >= 10:
            self.history.pop(0)

    def undo(self):
        """
        Revert the state to the previous state.
        """
        if self.history:
            # Save the current state to the redo stack
            self.redo_stack.append(copy.deepcopy(self.current_state))
            # Restore the previous state print(f"history {self.history}")
            self.current_state = self.history.pop()
        else:
            self.current_state = [[],[]]

    def redo(self):
        """
        Redo the last undone state change.
        """
        if self.redo_stack:
            # Save the current state to the history stack
            self.history.append(copy.deepcopy(self.current_state))
            # Restore the state from the redo stack
            self.current_state = self.redo_stack.pop()
        else:
            print("Nothing to redo.")

    def __str__(self):
        return str(self.current_state)

    def get_current_state(self) -> list[list[Point]]:
        return self.current_state

    def clear_history(self):
        """
        Clear the undo and redo history stacks.
        """
        self.history.clear()
        self.redo_stack.clear()
        self.current_state = [[], []]    