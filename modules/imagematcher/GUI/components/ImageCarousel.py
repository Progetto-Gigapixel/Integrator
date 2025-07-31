import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from core.IO import loadImagesFromFolder


class ImageCarousel:
    def __init__(self, root, image_folder, on_select, rows, cols,mask, thumbnail_size=(100, 100), display_size=(400, 400),index=1):
        """
        A custom widget for displaying a carousel of image thumbnails in a grid layout and a larger view of the selected image.

        Args:
            root: The parent Tkinter widget.
            image_folder: The folder path containing the images.
            on_select: Callback function triggered when an image is selected.
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.
            thumbnail_size: The size of the thumbnails (width, height).
            display_size: The size of the larger image display (width, height).
        """
        self.root = root
        self.image_folder = image_folder
        self.thumbnail_size = thumbnail_size
        self.display_size = display_size
        self.on_select = on_select
        self.rows = rows
        self.cols = cols
        self.index= index
        self.mask= mask

        # Validate mask dimensions
        if len(mask) != rows or any(len(row) != cols for row in mask):
            raise ValueError("Mask dimensions do not match the grid dimensions (rows x cols).")
        # Load images from the folder in a grid structure
        self.image_grid = self.load_images_from_folder(image_folder, rows, cols)
        if not self.image_grid:
            raise ValueError("No images found in the specified folder or incorrect grid dimensions.")

        # Create a frame for the carousel
        self.carousel_frame = ttk.Frame(root)
        self.carousel_frame.pack(side=tk.TOP, fill=tk.X, pady=10)

        # Create a canvas for the thumbnails
        self.canvas = tk.Canvas(self.carousel_frame, height=thumbnail_size[1])
        # Add scrollbars for the carousel
        self.scrollbar = ttk.Scrollbar(self.carousel_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.scrollbar = ttk.Scrollbar(self.carousel_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Create a frame inside the canvas to hold the thumbnails
        self.thumbnail_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.thumbnail_frame, anchor=tk.NW)

        # Load and display thumbnails in a grid layout
        print("create thumbnail")
        self.thumbnails = [[None for _ in range(cols)] for _ in range(rows)]
        for i in range(rows-1,-1,-1):
            for j in range(cols):
                image_path = self.image_grid[i][j]['path']
                if image_path:
                    thumbnail = self._create_thumbnail(image_path, i, j,self.mask[i][j])
                    self.thumbnails[i][j] = thumbnail


        # Bind the canvas to update scroll region
        self.thumbnail_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._bind_mouse_scroll(self.canvas)

    def _bind_mouse_scroll(self, widget):
        widget.bind("<Enter>", lambda e: self._enable_scroll(widget))
        widget.bind("<Leave>", lambda e: self._disable_scroll())

    def _enable_scroll(self, widget):
        widget.bind_all("<MouseWheel>", self._on_mousewheel)
        widget.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)

    def _disable_scroll(self):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Shift-MouseWheel>")
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")

    def _create_thumbnail(self, image_path, row, col, clickable=True):
        style = ttk.Style()
        style.theme_use("clam")
        image = Image.open(image_path)
        style.configure("Custom.TButton",
                bordercolor="gray",  # Border color
                borderwidth=1)   # Font
        style.map("Custom.TButton",
          background=[("active", "yellow"),   # Background color when clicked
                      ("pressed", "green")],  # Background color when pressed
          foreground=[("active", "red"),      # Text color when clicked
                      ("pressed", "white")])  # Text color when pressed
        image.thumbnail(self.thumbnail_size)
        photo = ImageTk.PhotoImage(image)
        # Create a button for the thumbnail
        if clickable:
            state=tk.NORMAL
        else:
            state= tk.DISABLED
        button = ttk.Button(self.thumbnail_frame, image=photo, command=lambda p=image_path: self.on_select(p, row, col, self.index), state=state,style="Custom.TButton")
        button.image = photo  # Keep a reference to avoid garbage collection
        # from image path get directory name
        directory=os.path.dirname(image_path)
        label_text = f"({row},{col})[{os.path.basename(directory)}]"
        label = ttk.Label(button, text=label_text, background="white", foreground="black", font=("Arial", 8))
        label.place(relx=1.0, rely=0.0, anchor=tk.NE)  # Position in the top-right corner
        button.grid(row=self.rows-1-row, column=col, padx=5, pady=5)
        return button

    def load_images_from_folder(self,base_folder,rows, cols):
        image_grid = loadImagesFromFolder(base_folder=base_folder, loadSIFT=False)
        return image_grid

    def update_mask(self, new_mask,image_path, row, col):
        # Validate new mask dimensions
        if len(new_mask) != self.rows or any(len(row) != self.cols for row in new_mask):
            raise ValueError("New mask dimensions do not match the grid dimensions (rows x cols).")

        self.mask = new_mask

        # Update button states based on the new mask
        for i in range(self.rows):
            for j in range(self.cols):
                if self.thumbnails[i][j]:
                    if self.mask[i][j]:
                        self.thumbnails[i][j].config(state=tk.NORMAL)
                    else:
                        self.thumbnails[i][j].config(state=tk.DISABLED)
