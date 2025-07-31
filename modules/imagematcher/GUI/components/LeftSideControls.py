import tkinter as tk

from components.StitchPoint import Point

class LeftSideControls:
    def __init__(self, parent, top1):
        self.control_buttons_frame = tk.Frame(parent)
        self.control_buttons_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_visible = True
        # self.parent = parent
        self.top1 = top1

        # Default to None, will be set via setters
        self.load_keypoints_func = None
        self.calculate_homography_func = None
        self.save_homography_func = None
        self.history_undo_func = None
        self.history_redo_func = None

        self.toggle_button = tk.Button(self.top1, text="Hide Controls", command=lambda: self.toggle_controls("RightSideControls"))
        self.toggle_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.add_point_list()

    # --- Setters for Button Functions ---
    def set_load_keypoints_func(self, func):
        self.load_keypoints_func = func

    def set_calculate_homography_func(self, func):
        self.calculate_homography_func = func

    def set_save_homography_func(self, func):
        self.save_homography_func = func

    def set_history_undo_func(self, func):
        self.history_undo_func = func

    def set_history_redo_func(self, func):
        self.history_redo_func = func

    def set_stitch_preview_func(self, func):
        # Default Stiching options
        self.st_opt = {
            "save" : False,
            "name" : None,
            "ba" : True,
            "tl" : False
        }
        self.stitch_preview_func = func
        #### Aggiungi label per Cambiare il nome

    # def set_stich_option_func(self, func):
    #     self.stich_option_func = func

    # def get_control_buttons_frame(self):
    #     return self.control_buttons_frame

    # --- Method to Initialize Buttons with Correct Commands ---
    def initialize_buttons(self):
        self.homo_frame = tk.Frame(self.control_buttons_frame)
        self.homo_frame.pack(side=tk.TOP, fill=tk.X)
        self.homo_label = tk.Label(self.homo_frame, text="Homography:")
        self.homo_label.pack(side=tk.TOP)
        self.homography_text = tk.Text(self.homo_frame, height=5, width=20)
        self.homography_text.pack(side=tk.TOP, padx=5, pady=5)

        self.buttons_frame = tk.Frame(self.homo_frame)
        self.buttons_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_keypts_button = tk.Button(
            self.buttons_frame, text="Load KPTs",
            command=self.load_keypoints_func if self.load_keypoints_func else self.default_action
        )
        self.load_keypts_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.homo_button = tk.Button(
            self.buttons_frame, text="Calculate H",
            command=self.calculate_homography_func if self.calculate_homography_func else self.default_action
        )
        self.homo_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.save_homo_button = tk.Button(
            self.control_buttons_frame, text="Save H",
            command=self.save_homography_func if self.save_homography_func else self.default_action
        )
        self.save_homo_button.pack(side=tk.TOP, padx=5, pady=5)

        self.history_frame = tk.Frame(self.control_buttons_frame)
        self.history_frame.pack(side=tk.TOP, fill=tk.X)

        self.undo = tk.Button(
            self.history_frame, text="Undo",
            command=self.history_undo_func if self.history_undo_func else self.default_action
        )
        self.undo.pack(side=tk.LEFT, padx=5, pady=5)

        self.redo = tk.Button(
            self.history_frame, text="Redo",
            command=self.history_redo_func if self.history_redo_func else self.default_action
        )
        self.redo.pack(side=tk.RIGHT, padx=5, pady=5)

        self.stitch_frame = tk.Frame(self.control_buttons_frame)
        self.stitch_button = tk.Button(self.control_buttons_frame, text="Stitch", command=lambda: self.stitch_preview_func(self.st_opt) if hasattr(self,"stitch_preview_func") else self.default_action())
        self.stitch_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.option_button = tk.Button(self.control_buttons_frame, text="Stiching Options", command= self.open_stitch_options )
        self.option_button.pack(side=tk.RIGHT, padx=5, pady=5)
        # Bind undo and redo to Ctrl+Z and Ctrl+Y
        self.control_buttons_frame.bind_all("<Control-z>", lambda event: self.history_undo_func() if self.history_undo_func else self.default_action())
        self.control_buttons_frame.bind_all("<Control-y>", lambda event: self.history_redo_func() if self.history_redo_func else self.default_action())

    def show_components(self):
        """Call this after setting functions to properly show buttons."""
        self.initialize_buttons()

    # Default function if no function is set
    def default_action(self):
        print("No function assigned!")

    def toggle_controls(self, flag):

        print(f"Toggling controls: {flag}")
        if self.controls_visible:
            self.control_buttons_frame.pack_forget()  # Hide the frame
        else:
            self.control_buttons_frame.pack(fill=tk.X)  # Show the frame
        self.controls_visible = not self.controls_visible

    def add_point_list(self):
        self.point_list_frame = tk.Frame(self.control_buttons_frame)
        self.point_list_frame.pack(side=tk.TOP, pady=5)
        # Add a label for the point list
        self.point_list_label = tk.Label(self.point_list_frame, text="Clicked Points:")
        self.point_list_label.pack(side=tk.TOP)
        self.point_list_text = tk.Text(self.point_list_frame,  width=20, height=10)
        self.point_list_text.pack(side=tk.BOTTOM, padx=5, pady=5)

    def set_homography_text(self, text):
        self.homography_text.delete('1.0', tk.END)
        self.homography_text.insert('1.0', text)

    def set_pointlist_text(self, point_list: list[Point]):
        """Update the point list text widget."""
        self.point_list_text.delete('1.0', tk.END)
        for i, point in enumerate(point_list):
            self.point_list_text.insert(tk.END, f"P{i}: ({point.x:.1f}, {point.y:.1f})\n")

    def open_stitch_options(self):
        options_window = tk.Toplevel(self.control_buttons_frame)
        options_window.title("Stitching Options")
        options_window.resizable(False, False)
        # options_window.geometry(f"{self.control_buttons_frame.winfo_width()}x{options_window.winfo_height()}")

        def save_options():
            self.st_opt["save"] = save_var.get()
            self.st_opt["name"] = name_entry.get() + ".tiff"
            self.st_opt["tl"] = tl_var.get()
            # self.st_opt["normals"] = normals_var.get()
            # self.st_opt["reflections"] = reflections_var.get()
            # self.st_opt["mesh"] = mesh_var.get()
            options_window.destroy()

        save_var = tk.BooleanVar(value=self.st_opt["save"])
        tl_var = tk.BooleanVar(value=self.st_opt["tl"])
        # normals_var = tk.BooleanVar(value=self.st_opt.get("normals", False))
        # reflections_var = tk.BooleanVar(value=self.st_opt.get("reflections", False))
        # mesh_var = tk.BooleanVar(value=self.st_opt.get("mesh", False))

        tk.Label(options_window, text="Save:").pack(anchor=tk.W)
        tk.Checkbutton(options_window, variable=save_var).pack(anchor=tk.W)

        tk.Label(options_window, text="Name: (.tiff will be added automatically)").pack(anchor=tk.W)
        name_entry = tk.Entry(options_window)
        name_entry.insert(0, self.st_opt["name"] if self.st_opt["name"] else "")
        name_entry.pack(anchor=tk.W)

        # tk.Label(options_window, text="Timelapse:").pack(anchor=tk.W)
        # tk.Checkbutton(options_window, variable=tl_var).pack(anchor=tk.W)

        # tk.Label(options_window, text="Stitching Options:").pack(anchor=tk.W)
        # tk.Checkbutton(options_window, text="Normals", variable=normals_var).pack(anchor=tk.W)
        # tk.Checkbutton(options_window, text="Reflections", variable=reflections_var).pack(anchor=tk.W)
        # tk.Checkbutton(options_window, text="Mesh", variable=mesh_var).pack(anchor=tk.W)

        tk.Button(options_window, text="Save", command=save_options).pack(anchor=tk.W)
