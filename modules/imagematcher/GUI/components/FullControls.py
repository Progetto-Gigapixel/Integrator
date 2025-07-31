import tkinter as tk
import numpy as np
import re
from tkinter import ttk
from tkinter.constants import X
from components.StitchPoint import Point

class FullControls:
    def __init__(self, root):
        self.root = root
        # self.root.title("Homography Tool")

        # Callback placeholders (inizialmente no-op)
        self._on_find_keypoints = lambda: None
        self._on_recompute_homography = lambda: None
        self._on_clean_reprojection = lambda: None
        self._on_threshold_changed = lambda val: None
        self._on_recompute_error = lambda: None
        self._on_preview_and_save = lambda: None
        self._on_undo = lambda: None
        self._on_redo = lambda: None
        self.build_ui()

    def build_ui(self):
        # Undo and Redo
        self.undo_redo_frame = ttk.Frame(self.root)
        self.undo_redo_frame.pack(pady=5, fill='x')

        self.undo_redo_label = ttk.Label(self.undo_redo_frame, text="Action History:")
        self.undo_redo_label.pack(side=tk.LEFT, padx=5)
        self.undo_button = ttk.Button(self.undo_redo_frame, text="Undo", command=self._undo)
        self.undo_button.pack(side=tk.LEFT, padx=5, expand=True, fill='x')

        self.redo_button = ttk.Button(self.undo_redo_frame, text="Redo", command=self._redo)
        self.redo_button.pack(side=tk.LEFT, padx=5, expand=True, fill='x')

        # 1. Find keypoints
        self.find_keypoints_button = ttk.Button(self.root, text="1. Find keypoints", command=self._find_keypoints)
        self.find_keypoints_button.pack(pady=5, fill='x')

        # self.recompute_homography_button = ttk.Button(self.root, text="Recompute homography", command=self._recompute_homography, state=tk.DISABLED)
        # self.recompute_homography_button.pack(pady=5, fill='x')
        self.left_text_label = ttk.Label(self.root, text="Left Image Keypoints:")
        self.left_text_label.pack(pady=5, fill='x')
        self.left_keypoints_frame = ttk.Frame(self.root)
        self.left_keypoints_frame.pack(pady=5, fill='x')
        self.left_keypoints_text_area = tk.Text(self.left_keypoints_frame, width=40,height=5)
        self.left_keypoints_text_area.pack(side=tk.LEFT,fill=X, expand=True)
        self.left_keypoints_scrollbar = ttk.Scrollbar(self.left_keypoints_frame, orient=tk.VERTICAL, command=self.left_keypoints_text_area.yview)
        self.left_keypoints_scrollbar.pack(side=tk.RIGHT, fill='y')
        self.left_keypoints_text_area.config(yscrollcommand=self.left_keypoints_scrollbar.set)

        self.right_text_label = ttk.Label(self.root, text="Right Image Keypoints:")
        self.right_text_label.pack(pady=5, fill='x')
        self.right_keypoints_frame = ttk.Frame(self.root)
        self.right_keypoints_frame.pack(pady=5, fill='x')
        self.right_keypoints_text_area = tk.Text(self.right_keypoints_frame,width=40 ,height=5)
        self.right_keypoints_text_area.pack(side=tk.LEFT,fill=X, expand=True)
        self.right_keypoints_scrollbar = ttk.Scrollbar(self.right_keypoints_frame, orient=tk.VERTICAL, command=self.right_keypoints_text_area.yview)
        self.right_keypoints_scrollbar.pack(side=tk.RIGHT, fill='y')
        self.right_keypoints_text_area.config(yscrollcommand=self.right_keypoints_scrollbar.set)

        self.search_frame = ttk.Frame(self.root)
        self.search_label = tk.Label(self.search_frame, text="Search Tiepoints:")
  
        self.search_entry = tk.Entry(self.search_frame)
        self.search_button = tk.Button(self.search_frame, text="Search", command=self._search_point)
        self.search_entry.bind('<Return>', lambda e: self._search_point())
        self.search_frame.pack(pady=5, fill='x')
        self.search_label.pack(side=tk.LEFT, pady=5, fill='x')
        self.search_entry.pack(side=tk.LEFT, pady=5, fill='x')
        self.search_button.pack(side=tk.LEFT, pady=5, fill='x')

        self.search_result = tk.Label(self.root, text="")
        self.search_result.pack(pady=5, fill='x')

        self.reprojection_button = ttk.Button(self.root, text="2. Reprojection error", command=self._recompute_error)
        self.reprojection_frame = ttk.Frame(self.root)
        self.reprojection_text = tk.Text(self.reprojection_frame, width=40,height=5)
        self.reprojection_scrollbar = ttk.Scrollbar(self.reprojection_frame, orient=tk.VERTICAL, command=self.reprojection_text.yview)
        self.reprojection_text.config(yscrollcommand=self.reprojection_scrollbar.set)

        self.reprojection_button.pack(pady=5, fill='x')
        self.reprojection_frame.pack(pady=5, fill='x')
        self.reprojection_text.pack(side=tk.LEFT,fill=X, expand=True)
        self.reprojection_scrollbar.pack(side=tk.RIGHT, fill='y')

        # 3. Threshold filter
        self.threshold_title = ttk.Label(self.root, text="3. Reprojection error Filter", relief="raised")
        self.threshold_label = ttk.Label(self.root, text="(Min, Max, Mean)")
        self.threshold_slider = ttk.Scale(self.root, from_=0, to=1, orient=tk.HORIZONTAL, command=self._threshold_changed)
        self.threshold_title.pack(pady=5, fill='x')
        self.threshold_label.pack(pady=5)
        self.threshold_slider.pack(fill='x', padx=10)

        self.recompute_error_frame = ttk.Frame(self.root)
        self.threshold_value_label = ttk.Label(self.recompute_error_frame, text="Current threshold: 0")
        self.recompute_error_button = ttk.Button(self.recompute_error_frame, text="Filter errors", command=self._filter_error)
        self.recompute_error_frame.pack(pady=5, fill='x')
        self.threshold_value_label.pack(side=tk.LEFT, padx=5)
        self.recompute_error_button.pack(side=tk.RIGHT, padx=5)
        self.recompute_error_frame.pack(anchor='center')
        self.filter_frame = ttk.Frame(self.root)
        self.filter_frame.pack(pady=5,fill='x')  # Reduced padding to decrease space
        self.label_frame= ttk.Frame(self.filter_frame)
        self.label_frame.pack(side=tk.TOP, padx=5)
        self.filter_label = ttk.Label(self.label_frame, text="Select N best points:")
        self.filter_label.pack(side=tk.LEFT, padx=5)
        self.filter_entry = ttk.Entry(self.label_frame, validate="key", validatecommand=(self.root.register(self._validate_integer), '%P'))
        self.filter_entry.pack(side=tk.LEFT, padx=5)
        self.undo_frame = ttk.Frame(self.filter_frame)
        self.undo_frame.pack(side=tk.BOTTOM, pady=2, fill=tk.X, expand=True)
        self.filter_button = ttk.Button(self.undo_frame, text="Select", command=self._filter_n_best)
        self.undo_button = ttk.Button(self.undo_frame, text="Undo", command=lambda: {self._undo(), self._recompute_error()})
        self.filter_button.pack(side=tk.LEFT, fill=X, padx=5, expand=True)
        self.undo_button.pack(side=tk.LEFT, fill=X, padx=5, expand=True)
        # Save homography and tiepoints
        self.save_button = ttk.Button(self.root, text="4. Save changes", command=self._save_homography)
        self.save_button.pack( pady=10, fill='x',expand=True)
        # 4. Preview and save
        self.preview_button = ttk.Button(self.root, text="5. Preview and save", command=self._preview_and_save)
        self.preview_button.pack(pady=10, fill='x',expand=True)

    # --- Set text methods ---
    def set_search_result(self, text):
        self.search_result.config(text=text)

    def set_left_keypoints_text(self, point_list: list[Point]):
        self.left_keypoints_text_area.delete(1.0, tk.END)
        for i, point in enumerate(point_list):
            self.left_keypoints_text_area.insert(tk.END, f"{point.text}: ({point.x:.1f}, {point.y:.1f})\n")

    def set_right_keypoints_text(self, point_list: list[Point]):
        self.right_keypoints_text_area.delete(1.0, tk.END)
        for i, point in enumerate(point_list):
            self.right_keypoints_text_area.insert(tk.END, f"{point.text}: ({point.x:.1f}, {point.y:.1f})\n")

    def set_reprojection_text(self, point_list: list[Point]|str):
        if isinstance(point_list, str):
            self.reprojection_text.delete('1.0', tk.END)
            self.reprojection_text.insert('1.0', point_list)
            self.set_error_values(0,0,0)
            return
        self.reprojection_text.delete('1.0', tk.END)
        for i, p in enumerate(point_list):
            self.reprojection_text.insert(tk.END, f"{p.text}: {p.reprojection_error :.6f}\n")
        # calculate the mean of reprojection errors
        reprojection_errors = [p.reprojection_error for p in point_list]
        error_mean=round(np.mean(reprojection_errors),2)
        error_max=round(np.max(reprojection_errors),2)
        error_min=round(np.min(reprojection_errors),2)
        self.set_error_values(error_min,error_max,error_mean)

    # --- Set threshold slider values ---
    def set_error_values(self, min_val, max_val, mean_val):
        self.threshold_slider.config(from_=min_val, to=max_val)
        self.threshold_slider.set(mean_val)
        self.threshold_label.config(text=f"(Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f})")
    # --- Callback methods ---
    def _search_point(self):
        self._on_search_point(self.search_entry.get().strip())
        self.search_entry.delete(0, tk.END)

    def _filter_n_best(self):
        entry_value = self.filter_entry.get().strip()
        if not entry_value:
            return
        self._on_filter_n_best(entry_value)

    def _save_homography(self):
        self._on_save_homography()
    
    def _find_keypoints(self):
        self._on_find_keypoints()

    def _recompute_homography(self):
        self._on_recompute_homography()

    def _clean_reprojection(self):
        self._on_clean_reprojection()

    def _threshold_changed(self, val):
        self.threshold_value_label.config(text=f"Current threshold: {float(val):.2f}")
        self._on_threshold_changed(val)

    def _recompute_error(self):
        self._on_recompute_error()

    def _preview_and_save(self):
        self._on_preview_and_save()

    def _undo(self):
        self._on_undo()

    def _redo(self):
        self._on_redo()

    def _filter_error(self):
        current_threshold = self.threshold_slider.get()
        self._on_filter_error(current_threshold)

    # --- Callback setters ---
    def set_on_save_homography(self, callback):
        self._on_save_homography = callback

    def set_on_search_point(self, callback):
        self._on_search_point = callback

    def set_on_filter_n_best(self, callback):
        self._on_filter_n_best = callback

    def set_on_filter_error(self, callback):
        self._on_filter_error = callback

    def set_on_find_keypoints(self, callback):
        self._on_find_keypoints = callback

    def set_on_recompute_homography(self, callback):
        self._on_recompute_homography = callback

    def set_on_clean_reprojection(self, callback):
        self._on_clean_reprojection = callback

    def set_on_threshold_changed(self, callback):
        self._on_threshold_changed = callback

    def set_on_recompute_error(self, callback):
        self._on_recompute_error = callback

    def set_on_preview_and_save(self, callback):
        self._on_preview_and_save = callback

    def set_on_undo(self, callback):
        self._on_undo = callback

    def set_on_redo(self, callback):
        self._on_redo = callback

    def _validate_integer(self, value_if_allowed):
        if value_if_allowed == "":
            return True
        try:
            int(value_if_allowed)
            return True
        except ValueError:
            return False

# # Esempio d'uso
# if __name__ == "__main__":
#     root = tk.Tk()
#     ui = HomographyToolUI(root)

#     # Collegare callback
#     ui.set_on_find_keypoints(lambda: print("Callback: Find keypoints"))
#     ui.set_on_recompute_homography(lambda: print("Callback: Recompute homography"))
#     ui.set_on_clean_reprojection(lambda: print("Callback: Clean reprojection"))
#     ui.set_on_threshold_changed(lambda val: print(f"Callback: Threshold changed to {val}"))
#     ui.set_on_recompute_error(lambda: print("Callback: Recompute error"))
#     ui.set_on_preview_and_save(lambda: print("Callback: Preview and save"))
#     ui.set_on_undo(lambda: print("Callback: Undo"))
#     ui.set_on_redo(lambda: print("Callback: Redo"))
#     ui.set_threshold_values(0.1, 10, 199)
#     root.mainloop()
