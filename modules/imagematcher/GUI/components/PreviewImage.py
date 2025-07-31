import tkinter as tk
from tkinter import Canvas, Scrollbar, Button
from PIL import Image, ImageTk
from typing import Optional

from cv2.typing import Prim
from components.ScrollableCanvas import ScrollableCanvas

class PreviewImage:
    def __init__(self, image : Image.Image, title: str):
        self.root = tk.Tk()
        self.root.title(title)
        self.title = title
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        self.image = image
        self.img_width, self.img_height = self.image.size

        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas_frame = tk.Frame(self.frame)
        self.canvas_frame.pack(side=tk.TOP, expand=tk.YES,fill=tk.BOTH)
        self.canvas = Canvas(self.canvas_frame, width=800, height=600, scrollregion=(0, 0, self.img_width, self.img_height))

        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.v_scroll = Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll = Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.pack(side=tk.TOP, expand=tk.YES,fill=tk.BOTH)

        self.canvas.config(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)

        self.scale_factor = 1.0
        self.zoom_frame = tk.Frame(self.frame)
        self.zoom_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.zoom_in_button = Button(self.zoom_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.RIGHT)

        self.zoom_out_button = Button(self.zoom_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.RIGHT)

        self.save_label = tk.Label(self.zoom_frame, text="Save as:")
        self.save_label.pack(side=tk.LEFT)

        self.save_entry = tk.Entry(self.zoom_frame)
        self.save_entry.pack(side=tk.LEFT)

        self.save_button = Button(self.zoom_frame, text="Save", command=self.save_image)
        self.save_button.pack(side=tk.LEFT)
        self._enable_mouse_scroll()

    def save_image(self):
        save_name = self.save_entry.get()
        if save_name and save_name.endswith(".tiff"):
            output_name = save_name
        else:
            output_name = self.title

        self.image.save(output_name)
        self.root.title("Saved!")
        self.save_button.pack(side=tk.RIGHT)

    def zoom(self, event):
        if event.delta > 0:
            self.scale_factor *= 1.3
        else:
            self.scale_factor *= 0.7
        self.apply_zoom()

    def zoom_in(self):
        print("Zoom image zoom")
        self.scale_factor *= 1.3
        self.apply_zoom()

    def zoom_out(self):
        self.scale_factor *= 0.7
        self.apply_zoom()

    def apply_zoom(self):
        new_width = int(self.img_width * self.scale_factor)
        new_height = int(self.img_height * self.scale_factor)
        resized_image = self.image.resize((new_width, new_height), Image.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(resized_image)
        self.canvas.itemconfig(self.image_id, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan(self, event):
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.canvas.scan_dragto(dx, dy, gain=1)

    def _enable_mouse_scroll(self):
        # self.canvas.bind("<ButtonPress-1>", self.start_pan)
        # self.canvas.bind("<B1-Motion>", self.pan)
        """Enable scrolling with the mouse wheel and trackpad gestures."""
        self.canvas.bind("<MouseWheel>", self._on_mouse_scroll)  # Windows/macOS vertical
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mouse_scroll)  # Windows/macOS horizontal
        self.canvas.bind("<Button-4>", self._on_linux_scroll_up)  # Linux vertical up
        self.canvas.bind("<Button-5>", self._on_linux_scroll_down)  # Linux vertical down
        self.canvas.bind("<Shift-Button-4>", self._on_linux_scroll_left)  # Linux horizontal left
        self.canvas.bind("<Shift-Button-5>", self._on_linux_scroll_right)  # Linux horizontal right

    def _on_mouse_scroll(self, event):
        """Scroll vertically using the mouse wheel (Windows/macOS)."""
        self.canvas.yview_scroll(-1 * (event.delta // 120), "units")

    def _on_shift_mouse_scroll(self, event):
        """Scroll horizontally when Shift + Mouse Wheel is used."""
        self.canvas.xview_scroll(-1 * (event.delta // 120), "units")

    def _on_linux_scroll_up(self, event):
        """Scroll up on Linux."""
        self.canvas.yview_scroll(-1, "units")

    def _on_linux_scroll_down(self, event):
        """Scroll down on Linux."""
        self.canvas.yview_scroll(1, "units")

    def _on_linux_scroll_left(self, event):
        """Scroll left on Linux."""
        self.canvas.xview_scroll(-1, "units")

    def _on_linux_scroll_right(self, event):
        """Scroll right on Linux."""
        self.canvas.xview_scroll(1, "units")
    def close_window(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# if __name__ == "__main__":
#     # Example of usage
#     image_path = "path_to_your_image.jpg"
#     image = Image.open(image_path)
#     preview = PreviewImage(image, "Image Preview")
#     preview._enable_mouse_scroll()
#     preview.canvas.bind("<ButtonPress-1>", preview.start_pan)
#     preview.canvas.bind("<B1-Motion>", preview.pan)
#     preview.root.mainloop()
