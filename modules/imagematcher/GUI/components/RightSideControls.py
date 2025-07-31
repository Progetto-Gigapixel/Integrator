import tkinter as tk
import numpy as np
from cv2.detail import normalizeUsingWeightMap

from components.StitchPoint import Point
from tokenize import String

class RightSideControls:
    def __init__(self, parent, top):
        self.control_buttons_frame = tk.Frame(parent)
        self.control_buttons_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.controls_visible = True
        # self.parent = parent
        self.top = top
        self.update_reprojection_errors = None
        self.clean_reprojection_error = None
        self.filter_error_n = None
        self.filter_error= None
        self.toggle_button = tk.Button(self.top, text="Hide Controls", command=lambda: self.toggle_controls("LeftSideControls"))
        self.toggle_button.pack(side=tk.RIGHT, padx=5, pady=5)


    def show_components(self):
        """Call this after setting functions to properly show buttons."""
        self.initialize_buttons()

    def add_point_list(self):
        self.point_list_frame = tk.Frame(self.control_buttons_frame)
        self.point_list_frame.pack(side=tk.TOP, pady=5)
        # Add a label for the point list
        self.point_list_label = tk.Label(self.point_list_frame, text="Clicked Points:")
        self.point_list_label.pack(side=tk.TOP)
        self.point_list_text = tk.Text(self.point_list_frame,  width=20, height=10)
        self.point_list_text.pack(side=tk.BOTTOM, padx=5, pady=5)

    def set_point_list_text(self, text):
        self.point_list_text.delete('1.0', tk.END)
        self.point_list_text.insert('1.0', text)

    def set_pointlist_text(self, point_list: list[Point]):
        """Update the point list text widget."""
        self.point_list_text.delete('1.0', tk.END)
        for i, point in enumerate(point_list):
            self.point_list_text.insert(tk.END, f"P{i}: ({point.x:.1f}, {point.y:.1f})\n")

    def add_reprojection_error(self, parent):
        reprojection_frame = tk.Frame(parent)
        reprojection_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        # Add a label for the reprojection errors
        self.error_mean_label=tk.StringVar(value="Mean: 0.0, min : 0.0, max: 0.0")
        reprojection_label_mean = tk.Label(reprojection_frame, textvariable=self.error_mean_label)
        reprojection_label_mean.pack(side=tk.TOP, anchor=tk.W)
        # Add a Text widget to display the reprojection errors
        self.reprojection_text = tk.Text(reprojection_frame, height=5, width=20)
        self.reprojection_text.pack(side=tk.TOP)

    def set_clean_reprojection_error_func(self, func):
        self.clean_reprojection_error = func

    def set_reprojection_text(self, reprojection_errors: list|str):
        if isinstance(reprojection_errors, str):
            self.reprojection_text.delete('1.0', tk.END)
            self.reprojection_text.insert('1.0', reprojection_errors)
            self.set_error_values(0,0,0)
            self.error_mean_label.set("Mean : " +str(self.error_mean)+", min : " +str(self.error_min)+", max: " +str(self.error_max))
            self.filter_threshold_scale.config(from_=self.error_min, to=self.error_max)
            return
        self.reprojection_text.delete('1.0', tk.END)
        for i, error in enumerate(reprojection_errors):
            self.reprojection_text.insert(tk.END, f"Point {i}: {error:.2f}\n")
        # calculate the mean of reprojection errors
        self.error_mean=round(np.mean(reprojection_errors),2)
        # self.set_error_value(self.error_mean, self.min_error, self.max_error)
        self.error_max=round(np.max(reprojection_errors),2)
        self.error_min=round(np.min(reprojection_errors),2)
        self.error_mean_label.set("Mean : " +str(self.error_mean)+", min : " +str(self.error_min)+", max: " +str(self.error_max))
        self.filter_threshold_scale.config(from_=self.error_min, to=self.error_max)

#
    def add_filter_error_threshold(self, parent):
        filter_threshold_frame = tk.Frame(parent)
        filter_threshold_frame.pack(side=tk.TOP, fill=tk.Y)

        label = tk.Label(filter_threshold_frame, text="Choose threshold")
        label.pack(side=tk.TOP, padx=5)
        self.filter_threshold_scale = tk.Scale(filter_threshold_frame, from_=self.error_min, to=self.error_max, orient=tk.HORIZONTAL, length=200, command=self._update_entry_from_scale)
        self.filter_threshold_scale.pack()
        filter_value_frame = tk.Frame(parent)
        filter_value_frame.pack(side=tk.TOP, fill=tk.Y)
        self.error_threshold_entry = tk.Entry(filter_value_frame,width=10)
        self.error_threshold_entry.pack(side=tk.LEFT,pady=5)

        filter_button = tk.Button(filter_value_frame, text="F", command=lambda: self.filter_error(self.filter_threshold_scale.get()) if self.filter_error is not None else self.default_action() )
        filter_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self._update_entry_from_scale(0.0)
        self.error_threshold_entry.bind("<Return>", self._update_scale_from_entry)

    def set_update_reprojection_errors_func(self, func):
        self.update_reprojection_errors = func

    def set_filter_error_n_func(self, func):
        self.filter_error_n = func

    def set_filter_error_func(self, func):
            self.filter_error= func

    def set_error_values(self, mean, min, max):
        self.error_mean = mean
        self.error_min = min
        self.error_max = max
    def _update_entry_from_scale(self, value):
        self.error_threshold_entry.delete(0, tk.END)
        self.error_threshold_entry.insert(0, value)
    def _update_scale_from_entry(self, event):
        value = self.error_threshold_entry.get()
        self.filter_threshold_scale.set(value)

    def add_show_reprojection_errors(self, parent):
        error_frame = tk.Frame(parent)
        error_frame.pack(side=tk.TOP, fill=tk.X)
        label = tk.Label(error_frame, text="show reprojection errors:")
        label.pack(side=tk.TOP)
        button_frame = tk.Frame(error_frame)
        button_frame.pack(side=tk.TOP)
        error_button = tk.Button(button_frame, text="R", command= lambda: self.update_reprojection_errors() if self.update_reprojection_errors else self.default_action())
        error_button.pack(side=tk.LEFT, padx=5, pady=5)
        error_button.bind("<Return>", lambda e: self.update_reprojection_errors() if self.update_reprojection_errors else self.default_action())
        clean_button = tk.Button(button_frame, text="Clean", command=lambda: [ self.set_reprojection_text(""), self.clean_reprojection_error() if self.clean_reprojection_error else self.default_action()])
        clean_button.pack(side=tk.LEFT, padx=5, pady=5)

    def add_show_best_errors(self, parent):
        error_frame = tk.Frame(parent)
        error_frame.pack(side=tk.TOP, fill=tk.X)
        label = tk.Label(error_frame, text="show best points:")
        label.pack(side=tk.TOP)

        entry_button_frame = tk.Frame(error_frame)
        entry_button_frame.pack(side=tk.TOP)

        self.error_n_entry = tk.Entry(entry_button_frame, width=10, validate='key', validatecommand=(error_frame.register(lambda P: P.isdigit() or P == ""), '%P'))
        self.error_n_entry.pack(side=tk.LEFT, pady=5)
        self.error_n_entry.bind("<Return>", lambda e: self.filter_error_n(best_points=self.error_n_entry.get()) if self.filter_error_n else self.default_action())
        error_button = tk.Button(entry_button_frame, text="E", command= lambda: self.filter_error_n(best_points=self.error_n_entry.get()) if self.filter_error_n else self.default_action())
        error_button.pack(side=tk.LEFT, padx=5, pady=5)

    def default_action(self):
        print("Unbound button: add function!")

    def initialize_buttons(self):
        self.add_point_list()
        self.add_show_reprojection_errors(self.control_buttons_frame)
        self.add_show_best_errors(self.control_buttons_frame)
        self.add_reprojection_error(self.control_buttons_frame)
        self.add_filter_error_threshold(self.control_buttons_frame)

    def toggle_controls(self, flag):
        print(f"Toggling controls: {flag}")
        if self.controls_visible:
            self.control_buttons_frame.pack_forget()  # Hide the frame
        else:
            self.control_buttons_frame.pack(fill=tk.X)  # Show the frame
        self.controls_visible = not self.controls_visible
