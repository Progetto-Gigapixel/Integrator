from os import X_OK
import tkinter as tk
import numpy as np
from typing import Optional, Callable
from PIL import Image, ImageTk
from components.StitchPoint import Point
from tkinter import Frame, Button, Event, Canvas
import platform

class ScrollableCanvas:
    _next_id = 1  # Class variable to track next available ID

    def __init__(self, parent: tk.Widget, tools: tk.Frame , image_path: str, width: Optional[int] = None, height: Optional[int] = None):
        """
        Create a canvas with vertical and horizontal scrollbars.

        Args:
            parent: The parent widget to contain the canvas
            image_path: Path to the image to be displayed
            width: Optional width of the canvas
            height: Optional height of the canvas
        """
        # Assign unique sequential ID
        self.image_id = ScrollableCanvas._next_id
        ScrollableCanvas._next_id += 1
        print(f"Creating ScrollableCanvas with ID: {self.image_id}")

        self.detached_window = None
        self.parent = parent
        self.canvas_frame = tk.Frame(self.parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.tools = tools # Frame for zoom buttons
        self.image_path = image_path  # Store the image path
        self.zoom_level = 1.0  # Track zoom level

        self.bind_func_canvas = {} # store function to call for each kind of click
        self.image_width = width
        self.image_height = height
        self.canvas = self._create_canvas(width, height)
        self.vsb = self._create_vertical_scrollbar()
        self.hsb = self._create_horizontal_scrollbar()
        self._configure_scrollbars()
        self._create_controls(self.tools)
        self._pack_components()

        # Load and display image

        self.original_image = Image.open(self.image_path)
        self.tk_image = ImageTk.PhotoImage(self.original_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image,tags="image")
        self.canvas.config(scrollregion=self.canvas.bbox("all")) # Set scroll region to match the image size
        # Color Index for points
        self.colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple", "pink", "brown"]
        self.colorIndex = 0
        self.manualIndex = 0 # Index for manual points

        self.automatic_points = [] # Store the points drawn by the automatic algorithm
        self.drawn_points = [] # Store the drawn points by the user
        self.error_circles = [] # Store the circles
        # Enable Mouse & Trackpad Scrolling
        self._enable_mouse_scroll()
        self.parent.bind("<Control-plus>", lambda e: self.zoom_image(1.1))
        self.parent.bind("<Control-minus>", lambda e: self.zoom_image(0.9))

        # self._enable_zoom()

    def _create_controls(self, frame: tk.Frame):
        """Create zoom in and zoom out buttons."""
        self.control_frame = tk.Frame(frame)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        self.zoom_out_button = tk.Button(self.control_frame, text="Zoom Out", command=lambda: self.zoom_image(0.5))
        self.zoom_in_button = tk.Button(self.control_frame, text="Zoom In", command=lambda: self.zoom_image(2))
        self.detach_button = tk.Button(self.control_frame, text="Detach", command=self.detach_canvas)
        self.reattach_button = tk.Button(self.control_frame, text="Reattach", command=self.reattach_canvas)
        self.zoom_level = 1
        # self.search_label = tk.Label(self.control_frame, text="Cerca Tiepoint")
        # self.search_entry = tk.Entry(self.control_frame, validate='key', validatecommand=(self.control_frame.register(lambda P: P == "" or P.isdigit()), '%P'))
        # self.search_button = tk.Button(self.control_frame, text="Cerca", command=lambda: self.search_point(self.search_entry.get().strip()))
        # self.search_entry.bind('<Return>', lambda e: self.search_point(self.search_entry.get().strip()))
        self.pathlabel = tk.Label(self.control_frame, textvariable= tk.StringVar(value=self.image_path),wraplength=256,relief="raised")

        # Bind ctrl+ and ctrl- for zoom
        # self.parent.bind("<Control-plus>", lambda e: self.zoom_image(1.1))
        # self.parent.bind("<Control-minus>", lambda e: self.zoom_image(0.9))

    def _create_canvas(self, width: Optional[int], height: Optional[int]) -> tk.Canvas:
        """Create and configure the canvas."""

        canvas = tk.Canvas(self.canvas_frame, relief=tk.SUNKEN, highlightthickness=0)
        if width:
            canvas.config(width=width)
        if height:
            canvas.config(height=height)
        return canvas

    def _create_vertical_scrollbar(self) -> tk.Scrollbar:
        """Create vertical scrollbar."""
        return tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)

    def _create_horizontal_scrollbar(self) -> tk.Scrollbar:
        """Create horizontal scrollbar."""
        return tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)

    def _configure_scrollbars(self):
        """Configure the scrollbars and canvas scroll commands."""
        self.vsb.config(command=self.canvas.yview)
        self.hsb.config(command=self.canvas.xview)
        self.canvas.config(yscrollcommand=self.vsb.set, xscrollcommand=self.hsb.set)
        self.canvas.config(scrollregion=self.canvas.bbox("all")) # Set scroll region to match the image size

    def _pack_components(self):
        """Pack all components in the correct order."""
        self.pathlabel.pack(side=tk.RIGHT, padx=5, pady=5)
        self.vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5, pady=5)
        if self.detached_window is None:
            self.detach_button.pack(side=tk.LEFT, padx=5, pady=5)
        else:
            self.reattach_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5, pady=5)
        # self.search_label.pack(side=tk.LEFT, padx=5, pady=5)
        # self.search_entry.pack(side=tk.LEFT, padx=5, pady=5)
        # self.search_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)


    def _enable_mouse_scroll(self):
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

    def get_canvas(self) -> tk.Canvas:
        """Return the canvas object."""
        return self.canvas

    def destroy(self) -> None:
        """Destroy the canvas and scrollbars."""
        self.canvas.destroy()
        self.vsb.destroy()
        self.hsb.destroy()
        self.zoom_out_button.destroy()
        self.zoom_in_button.destroy()
        self.detach_button.destroy()

    def detach_canvas(self):
        if self.detached_window is None or not tk.Toplevel.winfo_exists(self.detached_window):
            self.detached_window = tk.Toplevel(self.parent)
            self.detached_window.title(f"Detached Canvas {self.image_id}")
            self.detached_window.protocol("WM_DELETE_WINDOW", self.reattach_canvas)
            self.control_frame.pack_forget()
            self.canvas_frame.pack_forget()
            self.canvas.pack_forget()
            self.canvas_frame = Frame(self.detached_window)
            self.canvas_frame.pack(fill=tk.BOTH, expand=True)
            self.canvas = self._create_canvas(self.image_width, self.image_height)

            self.original_image = Image.open(self.image_path)
            self.tk_image = ImageTk.PhotoImage(self.original_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image,tags="image")
            self.zoom_level = 1.0

            self.vsb = self._create_vertical_scrollbar()
            self.hsb = self._create_horizontal_scrollbar()
            self._configure_scrollbars()
            new_tool_frame = Frame(self.detached_window)
            new_tool_frame.pack(side=tk.BOTTOM, fill=tk.X)
            self._create_controls(new_tool_frame)
            self._pack_components()
            self.update_point_list()
            self._enable_mouse_scroll()
            # self.canvas_frame.bind("<Control-plus>", lambda e: self.zoom_image(1.1))
            # self.canvas_frame.bind("<Control-minus>", lambda e: self.zoom_image(0.9))


            for button, func in self.bind_func_canvas.items():
                self.canvas.bind(button, lambda event, f=func: f(event, self.image_id))

    def reattach_canvas(self):
        if self.detached_window:
            self.detached_window.destroy()
            self.detached_window = None

        self.canvas_frame.pack_forget()
        self.canvas.pack_forget()

        self.canvas_frame = Frame(self.parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        # self.canvas = Canvas(self.canvas_frame, bg="white")
        # self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = self._create_canvas(self.image_width, self.image_height)
        self.vsb = self._create_vertical_scrollbar()
        self.hsb = self._create_horizontal_scrollbar()
        self.original_image = Image.open(self.image_path)
        self.tk_image = ImageTk.PhotoImage(self.original_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image,tags="image")
        self.zoom_level = 1.0
        self._configure_scrollbars()
        self._create_controls(self.tools)
        self._pack_components()
        self._enable_mouse_scroll()
        self.update_point_list()
        # self.parent.bind("<Control-plus>", lambda e: self.zoom_image(1.1))
        # self.parent.bind("<Control-minus>", lambda e: self.zoom_image(0.9))


    def zoom_image(self, scale_factor: float, absolute: bool = False) -> None:
        # Get canvas dimensions
        self.zoom_level = self.zoom_level * scale_factor
        self.canvas.scale("zoomable", 0, 0, scale_factor, scale_factor)
        self.fontSize = int(10 * scale_factor)
        for child_widget in self.canvas.find_withtag("text"):
            self.canvas.itemconfigure(child_widget, font=("Helvetica", self.fontSize))
        w0, h0 = self.original_image.size
        new_size = (int(w0 * self.zoom_level), int(h0 * self.zoom_level))
        resized_image = self.original_image.resize(new_size, Image.Resampling.LANCZOS)
        self.image1_tk = ImageTk.PhotoImage(resized_image)
        self.canvas.delete("image")
        # Create image at lowest layer
        image_id = self.canvas.create_image(0, 0, image=self.image1_tk, anchor=tk.NW, tags="image")
        self.canvas.tag_lower(image_id)

        # Move elements with the "highlight" tag accordingly with the zoom
        for hgl_item in self.canvas.find_withtag("highlight"):
            coords = self.canvas.coords(hgl_item)
            new_coords = [coord * scale_factor for coord in coords]
            self.canvas.coords(hgl_item, *new_coords)
        # Move elements with the "highlight" tag accordingly with the zoom
        for dot_item in self.canvas.find_withtag("dot"):
            coords = self.canvas.coords(dot_item)
            new_coords = [coord * scale_factor for coord in coords]
            self.canvas.coords(dot_item, *new_coords)
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def bind_canvas_click_events(self, button_to_manage: str, manage_click: Callable[[Event, int], None]):
        """Bind mouse click events to the canvases."""
        print(f"Bind click events for image {self.image_id}")

        self.bind_func_canvas[button_to_manage] = manage_click

        self.canvas.bind(button_to_manage, lambda event: manage_click(event, self.image_id))  # Left-click

    def change_picture(self, image_path: str, width: Optional[int] = None, height: Optional[int] = None) -> None:
        """Destroy the canvas and create a new one"""
        print(f"Changing picture: {image_path}")
        self.vsb.destroy()
        self.hsb.destroy()
        self.canvas.delete("image")
        self.canvas.destroy()
        self.canvas = self._create_canvas(width, height)
        self.vsb = self._create_vertical_scrollbar()
        self.hsb = self._create_horizontal_scrollbar()
        self._configure_scrollbars()
        self._pack_components()
        self._enable_mouse_scroll()

        # Update image path and load new image
        self.image_path = image_path  # Update the image path
        self.pathlabel.config(textvariable=tk.StringVar(value=self.image_path), relief="raised")
        self.original_image = Image.open(image_path)  # Use new image path
        self.tk_image = ImageTk.PhotoImage(self.original_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image, tags="image")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.colorIndex = 0
        self.manualIndex = 0
        self.drawn_points = []
        self.zoom_level = 1.0


    def draw_a_relative_point(self, x, y):
        """Draw a point on the canvas, taking into account the current zoom level and the scrolling.
        Use this if the point is in the original image coordinates. (Human drawn points)"""
        # Get the absolute canvas coordinates accounting for scroll position
        x = self.canvas.canvasx(x)  # Get absolute x coordinate
        y = self.canvas.canvasy(y)  # Get absolute y coordinate

        # Convert to normalized coordinates relative to original image
        x = x / self.zoom_level
        y = y / self.zoom_level

        color = self.colors[self.manualIndex % len(self.colors)]

        dot = self.canvas.create_oval(
            x * self.zoom_level -1 , y * self.zoom_level - 1,
            x * self.zoom_level + 1, y * self.zoom_level + 1,
            fill=color, outline=color, tags="dot"
        )
        text = "M" + str(self.manualIndex)
        text_id = self.canvas.create_text(
            x * self.zoom_level + 5, y * self.zoom_level + 5,
            text=text, fill=color, anchor=tk.NW,
            font=('Arial', int(10),'bold'),
            activefill=color, tags="zoomable"
        )
        self.drawn_points.append(Point(x, y, dot, text, text_id))  # Store relative coordinates
        self.manualIndex += 1
        return dot, text

    def draw_an_absolute_point(self,index, p : Point):
        """Draw a point on the canvas, Using the absolute point not taking care of the zoom.
        this function doesn't add the point at the list of drawn points
        Use this fucntion to update the canvas. (PC drawn points)"""
        # Get the absolute canvas coordinates accounting for scroll position
        """Draw a point on the canvas, using the absolute position as reference.
        Use this if the point is already in absolute coordinates. (PC drawn points)"""
        if p.text.startswith("M"):
            color = self.colors[self.manualIndex % len(self.colors)]
            p.text = "M" + str(self.manualIndex)
            self.manualIndex += 1
        else:
            color = self.colors[self.colorIndex % len(self.colors)]
            p.text = "A" + str(self.colorIndex)
            self.colorIndex += 1
        # Adjust coordinates based on the current zoom level
        x = p.x * self.zoom_level
        y = p.y * self.zoom_level

        # Draw the point exactly at the adjusted coordinates (x, y)
        dot = self.canvas.create_oval(
            x - 1, y - 1,
            x + 1, y + 1,
            fill=color, outline=color, tags="dot"
        )

        # Draw the text next to the point
        text_id = self.canvas.create_text(
            x + 5, y + 5,
            text=p.text, fill=color, anchor=tk.NW,
            font=('Arial', 10, 'bold'),
            activefill=color, tags="zoomable"
        )
        if p.reprojection_error > 0:
            self.draw_reprojection_error(p)
        p.text_id = text_id
        p.dot = dot
        return dot, text_id

    def highlight_point(self, p: Point):
        self.canvas.delete("highlight")

        x0 = (p.x - (40 - 4 * self.zoom_level)) * self.zoom_level
        y0 = (p.y - (40 - 4 * self.zoom_level)) * self.zoom_level
        x1 = (p.x + (40 - 4 * self.zoom_level)) * self.zoom_level
        y1 = (p.y + (40 - 4 * self.zoom_level)) * self.zoom_level

        system_os = platform.system().lower()

        if "linux" in system_os:
            # Su Linux (X11), stipple è supportato
            self.canvas.create_oval(
                x0, y0, x1, y1,
                fill="blue",
                outline="blue",
                stipple="gray25",
                width=0,
                tags="highlight"
            )
        else:
            # Su Windows/macOS: fallback
            self.canvas.create_oval(
                x0, y0, x1, y1,
                fill="",             # Nessun riempimento
                outline="blue",
                width=2,
                tags="highlight"
            )

    def draw_reprojection_error(self, p: Point):
        """x y absolute coordinates """
        x = p.x * self.zoom_level
        y = p.y * self.zoom_level
        r = int(p.reprojection_error) / 5 * 2 * self.zoom_level
        
        if r > 50:
            r = 50
            
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        system_os = platform.system().lower()
        
        if "linux" in system_os:
            # Su Linux (X11), stipple è supportato
            dot = self.canvas.create_oval(
                (x0), (y0),
                (x1), (y1),
                fill="red",
                outline="red",
                stipple="gray50",
                width=0,
                tags=["error", "zoomable"]
            )
        else:
            # Su Windows/macOS: fallback
            dot = self.canvas.create_oval(
                (x0), (y0),
                (x1), (y1),
                fill="",             # Nessun riempimento
                outline="red",
                width=2,
                tags=["error", "zoomable"]
            )

        p.set_error_circle(dot)

    def load_an_absolute_point(self, x, y):
        """Draw a point on the canvas, using the absolute position as reference.
        Use this if the point is already in absolute coordinates. (PC drawn points)"""
        color = self.colors[self.colorIndex % len(self.colors)]
        # Adjust coordinates based on the current zoom level
        x_zoom = x * self.zoom_level
        y_zoom = y * self.zoom_level

        # Draw the point exactly at the adjusted coordinates (x, y)
        dot = self.canvas.create_oval(
            x_zoom - 1, y_zoom - 1,
            x_zoom + 1, y_zoom + 1,
            fill=color, outline=color, tags="dot"
        )

        # Draw the text next to the point
        text = "A"+str(self.colorIndex)
        text_id = self.canvas.create_text(
            x_zoom + 5, y_zoom + 5,
            text=text, fill=color, anchor=tk.NW,
            font=('Arial', 10, 'bold'),
            activefill=color, tags="zoomable"
        )
        # Store the drawn point
        p = Point(x, y, dot, text, text_id)
        self.drawn_points.append(p)
        self.colorIndex += 1
        return dot, text

    def remove_closest_point(self, x: float, y: float, other_canvas: "ScrollableCanvas") :
        """Remove the closest point within a radius of 5 pixels. Takes as input x, y from the click event."""
        x = self.canvas.canvasx(x) / self.zoom_level
        y = self.canvas.canvasy(y) / self.zoom_level
        zoom_level = self.zoom_level
        min_distance = float('inf')
        closest_index = -1
        for i, p in enumerate(self.drawn_points):
            distance = np.sqrt((p.x - x) ** 2 + (p.y - y) ** 2)
            if distance < min_distance and distance <= 5 / zoom_level:  # Check if within 5 pixels (scaled)
                min_distance = distance
                closest_index = i
        print(f"Remove point at index {closest_index} - this Canvas {self.image_id}")
        if closest_index != -1:
            # Remove the closest point
            p : Point = self.drawn_points.pop(closest_index)
            self.remove_point(p)
            self.update_point_list()

            #remove the point with the same index in the other canva
            other_drawn_points = other_canvas.get_points()
            if len(other_drawn_points) > closest_index: # Check if the other canvas has the same index
                print(f"Remove point at index {closest_index} - Other Canvas {other_canvas.image_id}")
                p : Point = other_drawn_points.pop(closest_index)
                other_canvas.remove_point(p)
                # Update the indices of Controllerthe remaining points
                other_canvas.update_point_list()
            # print(f"Removed point at: ({p.x}, {p.y})")

    def remove_point(self,p : Point):
        """Remove the closest point within a radius of 5 pixels."""
        text_value = self.canvas.itemcget(p.text_id, "text")  # Get value of text
        self.canvas.delete(p.dot)     # Delete the dot
        self.canvas.delete(p.text_id)    # Delete the index label
        if p.error_circle:
            self.canvas.delete(p.error_circle)
        id = p.text
        if p.text.startswith("A"):
            self.colorIndex -= 1        
        # Move back the color index, it is also used as text
        # Update the indices of the remaining points
        for i, p in enumerate(self.drawn_points):
            self.canvas.itemconfig(p.text, text=str(i))
        print(f"Removed point: {text_value}")

    def search_point(self, point_number: str) -> Point | None:
        if not (point_number.startswith("M") or point_number.startswith("A")) or not point_number[1:].isdigit():
            print("Invalid point number format")
            found = False  # Inizializziamo `found` fuori dal ciclo
            return None
        # self.search_entry.update_idletasks()  # Assicura che il valore sia aggiornato
        # if not point_number:
        #     return None
        for point in self.drawn_points:
            if point.text == str(point_number):
                self.highlight_point(point)
                found = True
                return point
        if not found:
            print("Point not found")
            return None

    def _get_drawings(self):
        """Retrieve and print all drawings on the canvas."""
        all_items = self.canvas.find_all()  # Get all item IDs
        for item_id in all_items:
            item_type = self.canvas.type(item_id)  # Get the type of the item
            if item_type == 'rectangle':
                coords = self.canvas.coords(item_id)  # Get coordinates
                fill_color = self.canvas.itemcget(item_id, 'fill')  # Get fill color
                outline_color = self.canvas.itemcget(item_id, 'outline')  # Get outline color
                print(f'Rectangle: {coords}, Fill Color: {fill_color}, Outline Color: {outline_color}')
            elif item_type == 'oval':
                coords = self.canvas.coords(item_id)
                fill_color = self.canvas.itemcget(item_id, 'fill')
                outline_color = self.canvas.itemcget(item_id, 'outline')
                print(f'Oval: {coords}, Fill Color: {fill_color}, Outline Color: {outline_color}')
            elif item_type == 'text':
                text = self.canvas.itemcget(item_id, 'text')  # Get text
                coords = self.canvas.coords(item_id)
                print(f'Text: "{text}" at {coords}')

    def get_points(self) -> list[Point]:
        """Retrieve and print all points on the canvas."""
        return self.drawn_points

    def set_points(self, points: list[Point]) -> None:
        """Set the points on the canvas."""
        self.drawn_points = points
        self.update_point_list()

    def update_point_list(self):
        """Update the point list display for the specified image."""
        #print(f"update point list image {self.image_id}")
        clicked_points = self.drawn_points
        # I punti nella text area dovranno essere aggiornati dal Controller (SIFTMatcherApp)
        self._clear_canvas()
        # Draw all points back on the canvas with the updated indices
        for index, p in enumerate(clicked_points):
            self.draw_an_absolute_point(index, p)

    def _clear_canvas(self):
        """Clear all drawings on the canvas. But not the list of drawn points"""
        self.canvas.delete("error")
        self.canvas.delete("highlight")
        self.canvas.delete("zoomable")
        self.canvas.delete("dot")
        self.colorIndex = 0
        self.manualIndex = 0

    def clean_canvas(self):
        """Clean all drawings on the canvas and the list of drawn points"""
        self._clear_canvas()
        self.drawn_points = []
