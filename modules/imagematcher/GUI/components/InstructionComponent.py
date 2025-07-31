import tkinter as tk
import json

class InstructionComponent:

    def __init__(self, master):
        self.master = master
        self.label = tk.Label(master, text="Left Click: Add a point on the canvas. Right Click: Remove a point.")
        self.label.pack(side="left", padx=10, pady=0)

        self.button = tk.Button(master, text="More Instructions", command=self.show_instructions)
        self.button.pack(side="left", padx=10, pady=0)

    def show_instructions(self):
        instruction_window = tk.Toplevel(self.master)
        instruction_window.title("Instructions")
        instruction_label = tk.Label(instruction_window, text="Manual Stitching Application Guide", font=("TkDefaultFont", 12, "bold"))
        instruction_label.pack(side="top", padx=10, pady=10)
        menu_frame = tk.Frame(instruction_window, width=200, bg="lightgray")
        menu_frame.pack(side="left", fill="y")
        content_frame = tk.Frame(instruction_window, width=400, height=300)
        content_frame.pack_propagate(False)
        content_frame.pack(side="right", fill="both", expand=True)
        content_text = tk.Text(content_frame, wrap="word")
        content_text.pack(fill="both", expand=True, padx=10, pady=10)
        content_text.tag_configure("bold", font=("TkDefaultFont",10, "bold"))
        content_text.config(state="disabled")
        actions = self.load_actions_from_file()
        for action, guide in actions.items():
            action_button = tk.Button(menu_frame, text=action, command=lambda a=action: self.display_guide(content_text, actions[a]))
            action_button.pack(fill="x")

    def display_guide(self, content_text, guide):
        content_text.config(state="normal")
        content_text.delete("1.0", tk.END)
        parts = guide.split("**")
        for i, part in enumerate(parts):
            if i % 2 == 0:
                content_text.insert(tk.END, part)
            else:
                content_text.insert(tk.END, part, "bold")
        content_text.config(state="disabled")
    def load_actions_from_file(self):
        try:
            with open("actions.json", "r") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "Example Action": "This is an example guide. Load the JSON file containing the actions in the application directory."
            }
