import tkinter as tk
# from tkinter import tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
# from typing import Optional
from components.ImageCarousel import ImageCarousel
from components.StateManager import StateManager
from core.ImageGridStitcher import ImageGridStitcher
from components.ZoomableImage import ZoomableImage
from components.ScrollableCanvas import ScrollableCanvas
from components.StitchPoint import Point
from components.FullControls import FullControls
from components.InstructionComponent import InstructionComponent
from core.GridMatcher import find_center
from core.IO import save_global_homography_to_h5, save_global_homography_to_txt, save_images_grid_to_file, save_matches_to_h5
class SIFTMatcherApp:
    def __init__(self, match_list: list[list[object]]=[] , images_grid=[],image_dir="",ransac_threshold=1.0,bundle_adjustment=True, siftoptions : dict = {}):
        # Initialize the Tkinter application
        self.root = tk.Tk()
        self.root.title("SIFT Matches with Scrollable Tkinter")
        self.ransacThreshold = ransac_threshold
        self.ba = bundle_adjustment

        self.sift = cv2.SIFT_create(**siftoptions)
        #given a list of images
        ### TO componize this part
        self.image_dir=image_dir
        self.images_grid = images_grid
        self.nrows= len(self.images_grid)
        self.ncols= len(self.images_grid[0])
        self.center_row_index, self.center_col_index = find_center(self.nrows, self.ncols)
        self.match_list= match_list
        if (  self.center_row_index== 1):
            self.match_direction='left'
        else:
            self.match_direction='right'
         # can be left of right or top or bottom
        #given a list of matchpoints for each images
        self.H=None
        self.imgcoord1= [ self.center_row_index, self.center_col_index]
        self.imgcoord2= [self.center_row_index,self.center_col_index+1]
        self.set_image( self.imgcoord1[0], self.imgcoord1[1],1)
        self.set_image( self.imgcoord2[0], self.imgcoord2[1],2)

        if self.image1 is None or self.image2 is None:
            raise ValueError("Error: Could not load images.")
        self.img_panel = None

        self.merged_image=None

        # Lists to store clicked points and their canvas objects
        self.reprojection_errors=[]
        self.error_mean=0.0
        self.error_max=0.0

        # self.error_circles=[]
        # Initialize GUI components
        self.setup_gui()

        self.history = StateManager([self.canvas1.get_points(), self.canvas2.get_points()])

        # Print all the initial points of each canvas
        print("Initial points for canvas1:")
        for point in self.canvas1.get_points():
            print(f"Point: ({point.x}, {point.y})")

        print("Initial points for canvas2:")
        for point in self.canvas2.get_points():
            print(f"Point: ({point.x}, {point.y})")

        self.update_homography_matrix()

    def set_image(self, i,j, canvas_index):
        if canvas_index==1:
            self.imgcoord1= [i,j]
            self.image1 = Image.open(self.images_grid[i][j]['path'])
            self.image1_gs = cv2.imread(self.images_grid[i][j]['path'], cv2.IMREAD_GRAYSCALE)
            self.image1_pil = self.image1
            self.image1_og= ImageTk.PhotoImage(self.image1_pil)
            self.image1path = self.images_grid[i][j]['path']
            self.keypoints1 = self.images_grid[i][j]['keypoints']
            self.descriptors1 = self.images_grid[i][j]['descriptors']
            self.imgcoord1[0]=i
            self.imgcoord1[1]=j
        else:
            self.image2 = Image.open(self.images_grid[i][j]['path'])
            self.image2_pil = self.image2
            self.image2_gs = cv2.imread(self.images_grid[i][j]['path'], cv2.IMREAD_GRAYSCALE)
            self.image2_og= ImageTk.PhotoImage(self.image2_pil)
            self.image2path = self.images_grid[i][j]['path']
            # self.tiepts2 = []
            #self.keypoints2 = self.match_list[i][j]['homo_info']['dst_pts']
            self.keypoints2 =  self.images_grid[i][j]['keypoints']
            self.descriptors2 = self.images_grid[i][j]['descriptors']
            print(f"{self.imgcoord1} ---> {self.match_direction}")
            self.H = self.match_list[self.imgcoord1[0]][self.imgcoord1[1]][self.match_direction]['homography']
        return

    def on_select_image(self,image_path, row, col, index):
        """Select the displayable images"""

        print(f'image {index} row {row} col {col} selected {image_path}')

        self.selected_image=image_path
        if index==1:
            self.keypoints1 =  self.images_grid[row][col]['keypoints']
            self.descriptors1 = self.images_grid[row][col]['descriptors']
        else:
            self.keypoints2 =  self.images_grid[row][col]['keypoints']
            self.descriptors2 = self.images_grid[row][col]['descriptors']

        self.updateCanvas(self.selected_image,index)
        self.canvas1.clean_canvas()
        self.canvas2.clean_canvas()
        self.reprojection_errors=[]
        self.error_mean=0.0
        self.error_max=0.0
        if (index)==1:
            mask_2 = np.full((self.nrows, self.ncols), False)
            for r in range(max(0, int(row) - 1), min(self.nrows, int(row) + 2)):
                for c in range(max(0, int(col) - 1), min(self.ncols, int(col) + 2)):
                    if r != int(row) or c != int(col):
                        mask_2[r][c] = True
            self.imgcoord1= [row,col]
            self.images_carousel2.update_mask(mask_2,image_path, row, col)
            self.fullControls.set_left_keypoints_text(self.canvas1.get_points())
            self.fullControls.set_right_keypoints_text([])
            self.fullControls.set_reprojection_text("")
        else:
            self.imagecoord2 = [row,col]
            self.match_direction = determine_match_direction(self.imgcoord1, self.imagecoord2)
        self.history.clear_history()

    def updateCanvas(self, path, index):
        if index== 1:
            self.image1 = Image.open(path)
            self.image1_pil = self.image1
            self.image1path = path
            self.image1_og= ImageTk.PhotoImage(self.image1_pil)

            self.canvas1.change_picture(path)
            self.canvas1.bind_canvas_click_events("<Button-1>", self.on_image_click)  # Left-click
            self.canvas1.bind_canvas_click_events("<Button-3>", self.on_image_click)  # Right-click
        else:
            self.image2 = Image.open(path)
            self.image2_pil = self.image2
            self.image2path= path
            self.image2_og= ImageTk.PhotoImage(self.image2_pil)
            self.canvas2.change_picture(path)
            self.canvas2.bind_canvas_click_events("<Button-1>",  self.on_image_click)  # Left-click
            self.canvas2.bind_canvas_click_events("<Button-3>",  self.on_image_click)  # Right-click

    def setup_gui(self):
        """Initialize the GUI components."""
        self.container_frame = tk.Frame(self.root)
        self.container_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.frame1 = tk.Frame(self.container_frame)
        self.frame1.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.frame2 = tk.Frame(self.container_frame)
        self.frame2.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.control_frame = tk.Frame(self.container_frame)
        self.control_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)  # fill in altezza verticale
        self.top1 = tk.Frame(self.frame1)
        self.top2 = tk.Frame(self.frame2)

        # Mask the carousel
        mask_ini_1=  np.full((self.nrows, self.ncols), True)
        mask_ini_2 = np.full((self.nrows, self.ncols), True)

        self.images_carousel1 = ImageCarousel(self.top1, self.image_dir, on_select=self.on_select_image,rows=self.nrows,cols=self.ncols,mask= mask_ini_1,index=1)
        self.images_carousel2 = ImageCarousel(self.top2, self.image_dir, on_select=self.on_select_image, rows=self.nrows,cols=self.ncols,mask= mask_ini_2,index=2)
        self.top1.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=False)
        self.top2.pack(side=tk.TOP, padx=10, pady=10, fill=tk.BOTH, expand=False)
        self.tool1 = tk.Frame(self.frame1)
        self.tool2 = tk.Frame(self.frame2)

        self.controls_visible1 = True
        self.controls_visible2 = True

        self.tool1.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y, expand=False)
        self.tool2.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y, expand=False)

        self.canvas1 = ScrollableCanvas(self.frame1, self.top1, self.image1path)
        self.canvas2 = ScrollableCanvas(self.frame2, self.top2, self.image2path)
        self.canvas1.bind_canvas_click_events("<Button-1>", self.on_image_click);
        self.canvas1.bind_canvas_click_events("<Button-3>", self.on_image_click);
        self.canvas2.bind_canvas_click_events("<Button-1>", self.on_image_click);
        self.canvas2.bind_canvas_click_events("<Button-3>", self.on_image_click);
        self.fullControls = FullControls(self.control_frame)
        self.fullControls.set_on_find_keypoints(self.load_keypoints)
        self.fullControls.set_on_undo(self.history_undo)
        self.fullControls.set_on_redo(self.history_redo)
        self.fullControls.set_on_recompute_error(self.update_reprojection_errors)
        self.fullControls.set_on_filter_error(self.filter_error)
        self.fullControls.set_on_filter_n_best(self.select_best_points)
        self.fullControls.set_on_save_homography(self.save_homography_matrix)
        self.fullControls.set_on_search_point(self.search_point)
        self.fullControls.set_on_preview_and_save(self.stitch_preview)

        # self.fullControls.set_on_recompute_homography(lambda: (
        #     self.update_homography_matrix(),
        #     self.save_homography_matrix())
        # )
        w1,h1=self.image1_pil.size
        w2,h2 = self.image2_pil.size
        self.canvas1.update_point_list()
        self.canvas2.update_point_list()
        self.instruction = InstructionComponent(self.root)

    def toggle_controls(self,index):
        """Toggle the visibility of the control frame."""
        print(f'toggle_controls {index}')
        if index==1:
            if self.controls_visible1:
                self.tool1.pack_forget()  # Hide the frame
            else:
                self.tool1.pack(fill=tk.X)  # Show the frame
            self.controls_visible1 = not self.controls_visible1
        else:
            if self.controls_visible2:
                self.tool2.pack_forget()  # Hide the frame
            else:
                self.tool2.pack(fill=tk.X)  # Show the frame
            self.controls_visible2 = not self.controls_visible2
        return

    def load_keypoints(self):
        """Load SIFT keypoints into the clicked_points lists."""
        i = self.imgcoord1[0]
        j = self.imgcoord1[1]
        self.inliers_mask = self.match_list[i][j][self.match_direction]['inliers_mask']
        good_keypoints1=self.match_list[i][j][self.match_direction]['src_pts']
        good_keypoints2=self.match_list[i][j][self.match_direction]['dst_pts']
        match =self.match_list[int(self.imgcoord1[0])][int(self.imgcoord1[1])]
        self.H =match[self.match_direction]['homography']
        self.inliers_mask=match[self.match_direction]['inliers_mask']
        self.canvas1.clean_canvas()
        for (x, y) in good_keypoints1:
            dot,text=self.canvas1.load_an_absolute_point(x, y)
        # self.canvas1.update_point_list()
        self.fullControls.set_right_keypoints_text(self.canvas1.get_points())

        self.canvas2.clean_canvas()
        for (x, y) in good_keypoints2:
            dot,text=self.canvas2.load_an_absolute_point(x, y)
        # self.canvas2.update_point_list()
        self.fullControls.set_left_keypoints_text(self.canvas1.get_points())
        self.fullControls.set_right_keypoints_text(self.canvas2.get_points())
        # Calcola e mostra i reprojection errors
        # self.update_reprojection_errors()
        self.history.update_state([self.canvas1.get_points(),self.canvas2.get_points()])
        # print(f"Lenght point_list1 {len(self.canvas1.drawn_points)} point_list2 {len(self.canvas2.drawn_points)}")
        #
    def history_undo(self):
        self.history.undo()
        if self.history.current_state:
            # print(f"history_undo2: {len(self.history.current_state[0])}")
            self.canvas1.set_points(self.history.current_state[0])
            self.canvas2.set_points(self.history.current_state[1])
            self.canvas1.update_point_list()
            self.canvas2.update_point_list()
            self.update_homography_matrix()
            # self.save_homography_matrix()
            self.fullControls.set_reprojection_text("")
            self.fullControls.set_left_keypoints_text(self.canvas1.get_points())
            self.fullControls.set_right_keypoints_text(self.canvas2.get_points())
            self.fullControls.set_reprojection_text("Don't forget to calc again the errors")

    def history_redo(self):
        self.history.redo()
        if self.history.current_state:
            self.canvas1.set_points(self.history.get_current_state()[0])
            self.canvas2.set_points(self.history.get_current_state()[1])
            self.fullControls.set_left_keypoints_text(self.canvas1.get_points())
            self.fullControls.set_right_keypoints_text(self.canvas2.get_points())

    def stitch_preview(self):
        """ Stiching option:
            save: True or False
            name: Name of the file, can be None if save is False
            ba: bundle adjustment, True or False
            tl: Time Lapse, True or False
            normals: stitch normals, True or False
            reflections: stitch reflections, True or False
            mesh: stitch mesh, True or False """
        stitcher = ImageGridStitcher(self.images_grid, self.match_list)

        def get_unique_output_name(base_name, suffix, extension):
            i = 1
            output_name = f"{base_name}_{suffix}_{i}.{extension}"
            while os.path.exists(output_name):
                i += 1
                output_name = f"{base_name}_{suffix}_{i}.{extension}"
            return output_name

        # base_name = os.path.basename(self.image_dir)

        output_name = get_unique_output_name(os.path.join(self.image_dir, os.path.basename(self.image_dir)), "stiched", "tiff")
        merged_image = stitcher.get_stitched_image_from_center(
            save=True,
            image_name=output_name,
            rows=self.nrows,
            cols=self.ncols,
            ba=self.ba,
            GUI=True
        )
        gh = stitcher.getGlobalHomographies()
        pano_y, pano_x= stitcher.getPanoramaSize()
        preview_pil = Image.fromarray(merged_image)
        preview_pil.save(output_name)
        ZoomableImage(self.root,preview_pil, output_name).set_save_func(
            lambda output_name: (
                preview_pil.save(os.path.join(self.image_dir,output_name)),
                save_global_homography_to_h5(os.path.join(self.image_dir, "global_homography.hdf5"),gh,(pano_x, pano_y)),
                save_global_homography_to_txt(os.path.join(self.image_dir,"gh.txt"),gh,(pano_x, pano_y)),
                save_images_grid_to_file(os.path.join(self.image_dir, "images_grid.hdf5"), self.images_grid),
                save_matches_to_h5(os.path.join(self.image_dir, "match_list.hdf5"), self.match_list),
                print(f"Saved global homography to {os.path.join(self.image_dir, 'global_homography.hdf5')}"),
            )
        );

    def select_best_points(self, best_points: int):
        """
            Mostra i migliori best_points con il reprojection error piu' basso
        """        # Get number of points to show from input, defaulting to all points if empty
        points1 = self.canvas1.get_points()
        points2 = self.canvas2.get_points()
        try:
            n = len(points1) if not best_points else int(best_points)
            if 4 <= n <= len(points1) and len(points2)==len(points1):
                # keep only the n points with the highest reprojection error
                # indexes = sorted(range(len(self.reprojection_errors)), key=lambda i: self.reprojection_errors[i], reverse=True)[:n]
                indexes = sorted(range(len(self.canvas1.get_points())), key=lambda i: self.canvas1.get_points()[i].reprojection_error)[:n]
                indexes.reverse()
                print(f"Trovati {len(indexes)} migliori punti")
                self.canvas1.set_points([points1[i] for i in indexes])
                self.canvas2.set_points([points2[i] for i in indexes])
                self.canvas1.update_point_list()
                self.canvas2.update_point_list()
                self.fullControls.set_left_keypoints_text(self.canvas1.get_points())
                self.fullControls.set_right_keypoints_text(self.canvas2.get_points())
                self.update_reprojection_errors()
                self.history.update_state([self.canvas1.get_points(),self.canvas2.get_points()])
            else:
                self.fullControls.set_reprojection_text("There are not enough loaded points")
                print(f"Value must be between 4 and {len(self.canvas1.get_points())}")
        except ValueError:
            print("Please enter a valid number")

    def filter_error(self,threshold: float):
        # indexes = [i for i, value in enumerate(self.reprojection_errors) if value > threshold]
        # indexes.reverse()
        # for i in indexes:
        #     self.canvas1.drawn_points.pop(i)
        #     self.canvas2.drawn_points.pop(i)
        # threshold = float(threshold)
        filtered_points1 = []
        for p in self.canvas1.drawn_points:
            if p.reprojection_error <= threshold:
                filtered_points1.append(p)
        self.canvas1.set_points(filtered_points1)
        filtered_points2 = []
        for p in self.canvas2.drawn_points:
            if p.reprojection_error <= threshold:
                filtered_points2.append(p)
        self.canvas2.set_points(filtered_points2)

        print(f"filter_error: Length point_list1 {len(self.canvas1.get_points())} point_list2 {len(self.canvas2.get_points())}")
        self.canvas1.update_point_list()
        self.canvas2.update_point_list()
        self.update_reprojection_errors()
        self.update_homography_matrix()
        # self.save_homography_matrix()
        self.history.update_state([self.canvas1.get_points(),self.canvas2.get_points()])

    def on_image_click(self, event, image_id):
        """Handle mouse clicks on the images."""
        scroll_canvas = self.canvas1 if image_id == 1 else self.canvas2
        other_canvas = self.canvas2 if image_id == 1 else self.canvas1
        if event.num == 1:  # Left-click
            dot,text = scroll_canvas.draw_a_relative_point(event.x, event.y)
        elif event.num == 3:  # Right-click
            scroll_canvas.remove_closest_point(event.x,event.y, other_canvas)
        self.fullControls.set_left_keypoints_text(self.canvas1.get_points())
        self.fullControls.set_right_keypoints_text(self.canvas2.get_points())
        self.history.update_state([self.canvas1.get_points(),self.canvas2.get_points()])

    def update_homography_matrix(self):
        """Compute and update the homography matrix display."""
        self.clean_error_circle()
        point_list_2 : list[Point] = self.canvas1.get_points()
        point_list_1 : list[Point] = self.canvas2.get_points()
        src_points = np.array([(p.x, p.y) for p in point_list_1], dtype=np.float32)
        dst_points = np.array([(p.x, p.y) for p in point_list_2], dtype=np.float32)
        if len(src_points) >= 4 and len(src_points) == len(dst_points):
            homography_matrix,inliers_mask = calculate_homography(src_points,dst_points, self.ransacThreshold)
            self.H=homography_matrix
            self.inliers_mask=inliers_mask
            print(self.H)
            print(self.inliers_mask)

    def save_homography_matrix(self):
        row,col = self.imgcoord1
        good_keypoints1=[]
        good_keypoints2=[]
        for p in self.canvas1.get_points():
            good_keypoints1.append((p.x, p.y))
        for p in self.canvas2.get_points():
            good_keypoints2.append((p.x, p.y))
        src_points = np.array(good_keypoints1, dtype=np.float32)
        dst_points = np.array(good_keypoints2, dtype=np.float32)
        self.match_list[row][col][self.match_direction]['homography'] = self.H
        self.match_list[row][col][self.match_direction]['inliers_mask'] = self.inliers_mask
        self.match_list[row][col][self.match_direction]['src_pts'] = src_points
        self.match_list[row][col][self.match_direction]['dst_pts'] = dst_points
        print(f'Saved homogrphy for {row} {col} {self.match_direction}')
                
        row,col = self.imgcoord2
        mirror_match_direction = determine_match_direction(self.imgcoord2, self.imgcoord1)
        self.match_list[row][col][mirror_match_direction]['homography'] =self.H
        self.match_list[row][col][mirror_match_direction]['inliers_mask'] = self.inliers_mask
        self.match_list[row][col][mirror_match_direction]['src_pts'] = src_points
        self.match_list[row][col][mirror_match_direction]['dst_pts'] = dst_points
        print(f'Saved homogrphy for {row} {col} {mirror_match_direction}')
        return

    def clean_error_circle(self):
        """Remove the error circles from the canvas."""
        self.canvas2.get_canvas().delete("error")
        self.canvas1.get_canvas().delete("error")

    def search_point(self, point):
        """Search for a point in the canvas."""
        r1 = self.canvas1.search_point(point)
        r2 = self.canvas2.search_point(point)
        if r1 and not r2:
            self.fullControls.set_search_result("Found in first canvas")
        elif r2 and not r1:
            self.fullControls.set_search_result("Found in second canvas")
        elif not r1 and not r2:
            self.fullControls.set_search_result("Point not found")
        else:
            self.fullControls.set_search_result("Found")

    def update_reprojection_errors(self):
        """Compute and update the reprojection errors."""
        self.update_homography_matrix()
        points_list1 = self.canvas1.get_points()  # Points from first canvas
        points_list2 = self.canvas2.get_points()  # Points from second canvas
        # Check if we have enough corresponding points
        if len(points_list1) < 4 or len(points_list2) < 4:
            self.clean_error_circle()
            self.fullControls.set_reprojection_text("At least 4 points are required to compute reprojection errors.")
            return
        if len(points_list1) != len(points_list2):
            self.clean_error_circle()
            self.fullControls.set_reprojection_text(f"Mismatch in point counts: {len(points_list1)} vs {len(points_list2)}")
            return
        print("Updating reprojection errors")
        # Validate homography matrix
        if not isinstance(self.H, np.ndarray) or self.H.shape != (3, 3):
            print("Error: Invalid homography matrix")
            self.fullControls.set_reprojection_text("Invalid homography matrix")
            return
        # Extract corresponding points
        src_points = np.array([(p.x, p.y) for p in points_list2], dtype=np.float32)
        dst_points = np.array([(p.x, p.y) for p in points_list1], dtype=np.float32)
        # Transform points using homography
        src_points_homogeneous = np.hstack((src_points, np.ones((len(src_points), 1))))
        projected_homogeneous = np.dot(self.H, src_points_homogeneous.T).T
        # Handle division by zero
        mask = projected_homogeneous[:, 2] != 0
        if not all(mask):
            print(f"Warning: {sum(~mask)} points have zero z-coordinate after projection")
        # Apply normalization only to valid points
        projected_points = np.zeros_like(src_points)
        projected_points[mask] = projected_homogeneous[mask, :2] / projected_homogeneous[mask, 2:]
        # Calculate reprojection errors
        reprojection_errors = np.zeros(len(src_points))
        reprojection_errors[mask] = np.linalg.norm(projected_points[mask] - dst_points[mask], axis=1)
        # Clean previous error circles
        self.clean_error_circle()
        # Update points with error values and draw error circles
        for point1, point2, error in zip(points_list1, points_list2, reprojection_errors):
            point1.set_reprojection_error(error)
            point2.set_reprojection_error(error)
            self.canvas1.draw_reprojection_error(point1)
            self.canvas2.draw_reprojection_error(point2)
        # print("Reprojection errors:")
        # for i, (error, src, dst, proj) in enumerate(zip(reprojection_errors, src_points, dst_points, projected_points)):
        #     print(f"Point {i}: Error={error:.6f}, Src=({src[0]:.2f}, {src[1]:.2f}), Dst=({dst[0]:.2f}, {dst[1]:.2f}), Projected=({proj[0]:.2f}, {proj[1]:.2f})")
        # # Update UI with error statistics
        mean_error = np.mean(reprojection_errors[mask]) if any(mask) else float('nan')
        max_error = np.max(reprojection_errors[mask]) if any(mask) else float('nan')
        self.error_mean = mean_error
        self.error_max = max_error
        self.fullControls.set_reprojection_text(points_list1)
        # Update point list display
        self.canvas1.update_point_list()
        self.canvas2.update_point_list()
        # # Update history
        # self.history.update_state([points_list1, points_list2])

    def getH(self):
        return self.H

    def getMatchlist(self):
        return self.match_list

    def run(self):
        print("run")
          # Global error logger instance
        self.root.mainloop()

def determine_match_direction(src_coords, dest_coords):
    """Determine the match direction based on the relative position of the destination image."""
    direction_map = {
        (0, 1): "right",
        (0, -1): "left",
        (1, 0): "top",
        (-1, 0): "bottom",
        (-1, -1): "bottom_left",
        (-1, 1): "bottom_right",
        (1, -1): "top_left",
        (1, 1): "top_right"
    }
    delta_row = dest_coords[0] - src_coords[0]
    delta_col = dest_coords[1] - src_coords[1]
    return direction_map.get((delta_row, delta_col), None)

def calculate_homography(src_points, dst_points, ransacThreshold):
    """Calculate the homography matrix using RANSAC algorithm."""
    # Compute the homography matrix
    A, inliers_mask = cv2.estimateAffinePartial2D(from_=src_points, to=dst_points, method=cv2.RANSAC, ransacReprojThreshold=ransacThreshold)
    homography_matrix = np.array([[A[0, 0], A[0, 1], A[0, 2]] , [A[1, 0], A[1, 1], A[1, 2]], [0, 0, 1]])
    inliers_mask = inliers_mask.ravel().astype(bool)
    return homography_matrix, inliers_mask
