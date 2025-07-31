import pymeshlab
import os
import numpy as np
import re
from core.IO import extract_folder_indices, load_global_homo_from_h5
import cv2

class MeshFusion:
    def __init__(self, basepath):
        """Initialize the mesh loader with the base directory path."""
        self.basepath = basepath
        self.meshSet = pymeshlab.MeshSet()
        self.meshIndex = []
        # Store identity matrices for each loaded mesh
        self.transformMatrices = []  # Fixed typo in variable name
        self.max_row = 0
        self.max_col = 0
        print(f"Initializing mesh loader with basepath: {basepath}")
        self.loadMeshesFromFolder()

    def loadMeshesFromFolder(self):
        """
        Loads meshes from subfolders in the base folder, and stores them in a MeshSet.
        
        Subfolders should be named like A1, B2, etc. and contain a single mesh file
        in one of these formats: .obj, .ply, .stl, .off, .3ds, .dae.
        
        Returns:
            MeshSet: The loaded meshes
        """
        mesh_extensions = [".obj", ".ply", ".stl", ".off", ".3ds", ".dae"]
        print("\n=== Scanning for mesh subfolders ===")
        
        # Get sorted list of valid subfolders
        subfolders = sorted(
            [f.path for f in os.scandir(self.basepath) 
            if f.is_dir() and re.match(r"^([A-Z])(\d+)$", os.path.basename(f.path))],
            key=lambda x: (os.path.basename(x)[0], int(os.path.basename(x)[1:])))
        
        if not subfolders:
            print("Warning: No valid subfolders found!")
            return self.meshSet

        print(f"Found {len(subfolders)} potential mesh folders")
        
        # Determine grid dimensions
        for folder_path in subfolders:
            folder_name = os.path.basename(folder_path)
            row_index, col_index = extract_folder_indices(folder_name)
            self.max_row = max(self.max_row, row_index)
            self.max_col = max(self.max_col, col_index)
        print(f"Grid dimensions: {self.max_row+1} rows x {self.max_col+1} columns")
        # Load meshes from each folder
        success_count = 0
        for i, subfolder in enumerate(subfolders, 1):
            folder_name = os.path.basename(subfolder)
            row_index, col_index = extract_folder_indices(folder_name)
            print(f"\n[{i}/{len(subfolders)}] Processing folder: {folder_name} (Row {row_index}, Col {col_index})")
            
            # Find first valid mesh file
            mesh_files = [f for f in os.listdir(subfolder) 
                        if any(f.lower().endswith(ext) for ext in mesh_extensions)]
            
            if not mesh_files:
                print(f" - Warning: No mesh files found in {folder_name}")
                continue
                
            if len(mesh_files) > 1:
                print(f" - Note: Multiple mesh files found, using first one")
                
            mesh_path = os.path.join(subfolder, mesh_files[0])
            print(f" - Loading mesh: {mesh_files[0]}")
            
            try:
                self.meshSet.load_new_mesh(mesh_path)
                self.meshIndex.append((row_index, col_index))
                self.transformMatrices.append(np.eye(4))  # Initialize with identity matrix
                success_count += 1
                print(" - Successfully loaded mesh")
            except Exception as e:
                print(f" - Error loading mesh: {str(e)}")
                continue

        print(f"\n=== Loading complete ===")
        print(f"Successfully loaded {success_count}/{len(subfolders)} meshes")
        return self.meshSet
    def extract_2d_transform_opencv(self,H, camera_matrix=None):
        """
        Extract 2D rotation and translation from homography using OpenCV.
        
        Args:
            H: 3x3 homography matrix
            camera_matrix: 3x3 camera intrinsic matrix (optional)
            
        Returns:
            dict containing rotation and translation parameters
        """
        # Default camera matrix (identity for normalized coordinates)
        if camera_matrix is None:
            camera_matrix = np.eye(3, dtype=np.float32)
        
        # Ensure homography is float32
        H = H.astype(np.float32)
        
        try:
            # Decompose homography
            num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, camera_matrix)
            
            results = []
            for i in range(num_solutions):
                R = rotations[i]
                t = translations[i].flatten()
                
                # Convert 3D rotation matrix to 2D rotation angle
                # For planar motion, we look at the rotation around Z-axis
                rotation_angle = np.arctan2(R[1, 0], R[0, 0])
                # keep positive translation values

                # devono avere lo stesso segno
                if (H[0, 2] < 0) != ( t[0] < 0) or (H[1, 2] < 0) != (t[1] < 0):
                    #print(f"Solution {i}: Inverting translation to match homography sign")
                    continue
                # keep the rotation with x and y positive
                if (R[0, 0] < 0) or (R[1, 1] < 0):
                    #print(f"Solution {i}: Inverting rotation to keep positive x and y")
                    continue
                #print(f"Solution {i}: {H[0, 2]}, {H[1, 2]}")
                #print(f"Solution {i}: Rotation matrix:\n{R}")
                #print(f"Solution {i}: Translation vector: {t}")
                return{
                    'solution_index': i,
                    'rotation_angle': rotation_angle,
                    'rotation_degrees': np.degrees(rotation_angle),
                    'translation_x': t[0],
                    'translation_y': t[1],
                    'translation_z': t[2],
                    'rotation_matrix': R,
                    'normal_vector': normals[i].flatten()
                }
            
            return {
                'num_solutions': num_solutions,
                'solutions': results
            }
        
        except Exception as e:
            return {'error': str(e), 'num_solutions': 0}
    def homography_to_euclidean(self,homography):
        """
        Convert homography matrix to Euclidean transformation (rotation + translation)
        
        Args:
            homography: 3x3 homography matrix
        
        Returns:
            rotation_matrix: 2x2 rotation matrix
            translation: 2x1 translation vector
        """
        # Normalize the homography matrix
        H = homography / homography[2, 2]
        
        # Extract the 2x2 upper-left submatrix and translation
        A = H[:2, :2]
        t = H[:2, 2]
        
        # Use SVD to decompose A = U * S * V^T
        U, S, Vt = np.linalg.svd(A)
        
        # Pure rotation matrix (closest orthogonal matrix to A)
        R = U @ Vt
        
        # Ensure proper rotation (det(R) = 1, not -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
    
        return R, t
    
    def transform_2d_to_3d_preserve_origin(self,rotation_2d, translation_2d,panorama_height, image_height=None):
        """
        Convert 2D transformation to 3D while preserving the image origin location.
        This version handles Y-axis flip while keeping the same effective origin.
        
        Parameters:
        - rotation_2d: 2x2 numpy array representing 2D rotation matrix
        - translation_2d: 2x1 numpy array (or 1D array of length 2) representing 2D translation
        - image_height: float, height of the image (required for proper origin preservation)
        
        Returns:
        - 4x4 numpy array representing the 3D rigid transformation matrix
        """
        if image_height is None:
            raise ValueError("image_height must be provided for origin preservation")
        # Ensure inputs are numpy arrays
        R_2d = np.array(rotation_2d) 
        t_2d = np.array(translation_2d).flatten()
        print(f"R_2d: {R_2d}, t_2d: {t_2d}, panorama_height: {panorama_height}, image_height: {image_height}")
        # Validate input dimensions
        if R_2d.shape != (2, 2):
            raise ValueError("rotation_2d must be a 2x2 matrix")
        if t_2d.shape != (2,):
            raise ValueError("translation_2d must be a 2x1 vector or 1D array of length 2")
        # Create 4x4 transformation matrix
        T_3d = np.eye(4)


        # Y-axis flip matrix
        flip_y = np.array([[ 1,  0],
                        [ 0, -1]])
        
        # Apply Y-axis flip to rotation
        R_3d = flip_y @ R_2d @ flip_y
        
        # For translation, we need to account for the origin shift
        # In 2D image: origin at top-left (0, 0)
        # In 3D world: origin at bottom-left (0, image_height) after flip
        origin_offset = np.array([0, image_height])
        #print(f"Origin offset (3D): {origin_offset}")
        # Transform the translation accounting for origin change
        #print(f"Translation in 2D before flip: {t_2d}")
        #print(f"Translation in 2D after flip: {flip_y @ t_2d} , original offset: {origin_offset}, flip offset: { R_3d @ flip_y @ origin_offset}")
        t_3d = flip_y @ t_2d + R_3d @ flip_y @ origin_offset
        t_3d[1] = panorama_height + t_3d[1]  # Adjust Y translation to match panorama height
        print(f"Rotation in 3D: {R_3d}")
        print(f"Translation in 3D: {t_3d}")
        
        # Embed into 3D matrix
        T_3d[0:2, 0:2] = R_3d
        T_3d[0:2, 3] = t_3d
        return T_3d
    
    def load_transform_matrices(self,image_height=8472):
        """Convert 3x3 homography to 4x4 rigid transform with Y-up coordinates"""
        h5_path = os.path.join(self.basepath, "global_homography.hdf5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Transformation file not found: {h5_path}")
        print(f"\nConverting 3x3 homographies to 4x4 rigid transforms (Y-up)")
        gh, pano_size = load_global_homo_from_h5(h5_path)
        self.transformMatrices = []
        pano_x, pano_y = pano_size
        for i, (row, col) in enumerate(self.meshIndex):
            if row >= len(gh) or col >= len(gh[row]):
                print(f"Warning: No transform for mesh {i} at row {row}, col {col}")
                self.transformMatrices.append(np.eye(4))
                continue
            H = gh[row][col]
            T = np.eye(4)
            R,t = self.homography_to_euclidean(H)
            T = self.transform_2d_to_3d_preserve_origin(rotation_2d=R,
                translation_2d=t,panorama_height=pano_y,
                image_height=image_height
            )
            self.transformMatrices.append(T)
        return self.transformMatrices

    def _decompose_homography(self, H_2x2):
        """Helper: Decompose 2x2 homography matrix into scale and rotation"""
        # Polar decomposition (R = rotation, S = scale/skew)
        U, s, Vt = np.linalg.svd(H_2x2)
        R = U @ Vt
        S = Vt.T @ np.diag(s) @ Vt
        return S, R

    def apply_transform_matrices(self):
        """Apply loaded transformation matrices to all meshes"""
        num_mesh=len(self.meshIndex)
        # for i in range(num_mesh):
        #     print(self.transformMatrices[i])
        print(f'the project has mesh number: {self.meshSet.mesh_number()}')
        if not self.transformMatrices:
            raise ValueError("No transformation matrices loaded")
        print("\n=== Applying transformations to meshes ===")
        for i in range(len(self.meshIndex)):
            print(f"\nApplying transform to mesh {i} at index {self.meshIndex[i]}")
            if i >= self.meshSet.mesh_number():
                print(f"Warning: Mesh index {i} out of bounds, skipping")
                continue
            self.meshSet.set_current_mesh(i)
            transform = self.transformMatrices[i]
            # tranform to np ndarray
            transform = np.array(transform, dtype=float)
            print(f"Applying transform to mesh {i} {self.meshIndex[i]}: {np.array(transform[:3,3])}")
            self.meshSet.set_matrix(transformmatrix=transform,freeze=True)


    def global_alignment(self):
        """Perform global alignment of all meshes"""
        print("\n=== Performing global alignment ===")
        self.meshSet.compute_matrix_by_mesh_global_alignment(
            basemesh=0,
            onlyvisiblemeshes=True,
            ogsize=50000,
            arcthreshold=0.3,
            recalcthreshold=0.1,
            samplenum=2000,
            mindistabs=10,
            trgdistabs=0.005,
            maxiternum=77,
            samplemode=True,
            reducefactorperc=0.8,
            passhifilter=0.75,
            matchmode=True
        )
        
        # Update transformation matrices with alignment results
        new_matrices = []
        for i in range(len(self.meshIndex)):
            self.meshSet.set_current_mesh(i)
            cm = self.meshSet.current_mesh()
            matrix = cm.transform_matrix()
            
            # # Combine original transform with alignment result
            # new_matrix = np.eye(4) - matrix + np.array(self.transformMatrices[i], dtype=np.float64)
            # new_matrices.append(new_matrix)
            
            # Apply the combined transformation
            self.meshSet.set_matrix(transformmatrix=matrix, freeze=True)
        
        #self.transformMatrices = new_matrices
        print("Global alignment completed")
    def poisson_reconstruction(self):
        print("\n=== Poisson reconstruction ===")
        # self.meshSet.generate_surface_reconstruction_screened_poisson(
        #     visiblelayer = False,
        #     depth = 8,
        #     fulldepth = 5,
        #     cgdepth = 0,
        #     scale = 1.1,
        #     samplespernode =  1.5,
        #     pointweight = 4,
        #     iters = 8,
        #     confidence = False,
        #     preclean = False,
        #     threads = 16
        # )
        self.meshSet.generate_surface_reconstruction_screened_poisson(
            depth = 7,
            samplespernode =  1.5,
            pointweight = 4.0,
        )
        print("Poisson reconstruction completed")

    def save_project(self, decimate=False, decimation_percentage=0.5,mesh_name="out_merged.ply" ):
        """Save the project, optionally with decimation"""
        print("\n=== Saving project ===")
        
        # Save unmerged project
        project_path = os.path.join(self.basepath, "unmerged.mlp")
        self.meshSet.save_project(project_path)
        print(f"Saved unmerged project to {project_path}")
        
        # Merge all visible meshes
        print("Merging meshes into single mesh")
        self.meshSet.generate_by_merging_visible_meshes(
            mergevisible=True,
            deletelayer=True,
            mergevertices=True,
            alsounreferenced=False
        )
        print(f'Merged project has {self.meshSet.mesh_number()} meshes')
        self.poisson_reconstruction()
        
        # Optional decimation
        if decimate:
            print(f'Decimating mesh (target: {decimation_percentage*100}% reduction)')
            self.meshSet.meshing_decimation_quadric_edge_collapse(
                targetperc=decimation_percentage
            )
            print(f'Decimated mesh has {self.meshSet.mesh_number()} meshes')
        
        # Save final merged mesh
        output_path = os.path.join(self.basepath, mesh_name)
        self.meshSet.save_current_mesh(output_path)
        print(f"Saved merged mesh to {output_path}")
    def crop_border(self, dilation_steps=3, save_cleaned=False):
        """
        Remove mesh borders by:
        1. Selecting border vertices
        2. Expanding the selection
        3. Removing selected vertices and faces
        
        Args:
            dilation_steps (int): Number of dilation iterations to expand selection
            save_cleaned (bool): Whether to save each cleaned mesh individually
        """
        print("\n=== Cleaning mesh borders ===")
        
        # Create directory for cleaned meshes if needed
        if save_cleaned:
            cleaned_dir = os.path.join(self.basepath, "cleaned_meshes")
            os.makedirs(cleaned_dir, exist_ok=True)
        
        for i in range(len(self.meshIndex)):
            self.meshSet.set_current_mesh(i)
            original_verts = self.meshSet.current_mesh().vertex_number()
            
            print(f"\nProcessing mesh {i} ({self.meshIndex[i]})")
            print(f"Original vertices: {original_verts}")
            
            try:
                # 1. Select border vertices
                self.meshSet.compute_selection_from_mesh_border()
                
                # 2. Expand selection
                if dilation_steps > 0:
                    for _ in range(dilation_steps):
                        self.meshSet.apply_selection_dilatation()
                # 3. Remove selected vertices and faces
                self.meshSet.meshing_remove_selected_vertices_and_faces()
                
                # Report results
                cleaned_verts = self.meshSet.current_mesh().vertex_number()
                print(f"Cleaned vertices: {cleaned_verts}")
                print(f"Removed {original_verts - cleaned_verts} vertices ({((original_verts - cleaned_verts)/original_verts)*100:.1f}%)")
                
                # Save individual cleaned mesh if requested
                if save_cleaned:
                    mesh_path = os.path.join(cleaned_dir, f"mesh_{i}_cleaned.ply")
                    self.meshSet.save_current_mesh(mesh_path)
                    print(f"Saved cleaned mesh to {mesh_path}")
                    
            except Exception as e:
                print(f"Error processing mesh {i}: {str(e)}")
                continue
        
        print("\nBorder cleaning complete")
    def save_transformation_matrices(self, filename="transformation_matrices.txt"):
        """
        Save all transformation matrices to a text file in human-readable format.
        
        Args:
            filename (str): Name of the output text file (will be saved in basepath)
        """
        output_path = os.path.join(self.basepath, filename)
        
        with open(output_path, 'w') as f:
            f.write("Transformation Matrices\n")
            f.write("======================\n\n")
            
            for i, (row, col) in enumerate(self.meshIndex):
                matrix = self.transformMatrices[i]
                
                f.write(f"Mesh {i} (Row {row}, Col {col}):\n")
                
                # Write the 4x4 matrix
                np.savetxt(f, matrix, fmt='%10.6f', delimiter='\t')
                
                # Extract and write translation and rotation components
                translation = matrix[:3, 3]
                rotation = matrix[:3, :3]
                
                f.write("\nTranslation (X, Y, Z):\n")
                np.savetxt(f, [translation], fmt='%10.6f', delimiter='\t')
                
                f.write("\nRotation Matrix:\n")
                np.savetxt(f, rotation, fmt='%10.6f', delimiter='\t')
                
                # Calculate Euler angles (ZYX convention)
                sy = np.sqrt(rotation[0,0] * rotation[0,0] +  rotation[1,0] * rotation[1,0])
                singular = sy < 1e-6
                
                if not singular:
                    x = np.arctan2(rotation[2,1], rotation[2,2])
                    y = np.arctan2(-rotation[2,0], sy)
                    z = np.arctan2(rotation[1,0], rotation[0,0])
                else:
                    x = np.arctan2(-rotation[1,2], rotation[1,1])
                    y = np.arctan2(-rotation[2,0], sy)
                    z = 0
                    
                f.write("\nEuler Angles (ZYX) in degrees:\n")
                f.write(f"Roll (X): {np.degrees(x):.2f}\n")
                f.write(f"Pitch (Y): {np.degrees(y):.2f}\n")
                f.write(f"Yaw (Z): {np.degrees(z):.2f}\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Transformation matrices saved to {output_path}")
    def get_transformation_matrices(self):
        """
        Return all transformation matrices as a NumPy array of 4x4 matrices.
        
        Returns:
            np.ndarray: Array of shape (N, 4, 4) where N is number of meshes,
                    containing all transformation matrices
        """
        if not self.transformMatrices:
            raise ValueError("No transformation matrices loaded")
        
        # Convert list of matrices to numpy array
        matrices_array = np.array(self.transformMatrices, dtype=np.float64)
        
        # Verify shape is correct (N, 4, 4)
        if matrices_array.shape[1:] != (4, 4):
            raise ValueError(f"Unexpected matrix shape: {matrices_array.shape}")
        
        return matrices_array
    def get_transformation_matrices_with_indices(self):
        """
        Return transformation matrices with their grid indices.
        
        Returns:
            tuple: (matrices, indices) where:
                - matrices: np.ndarray of shape (N, 4, 4) containing 4x4 transformation matrices
                - indices: list of tuples (row, col) representing grid positions
        """
        if not self.transformMatrices:
            raise ValueError("No transformation matrices loaded")
        
        # Convert to numpy array and verify shape
        matrices = np.array(self.transformMatrices, dtype=np.float64)
        if matrices.shape[1:] != (4, 4):
            raise ValueError(f"Unexpected matrix shape: {matrices.shape}")
        
        # Return both matrices and their corresponding indices
        return matrices, self.meshIndex.copy()