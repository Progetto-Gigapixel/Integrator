"""
Core photometric stereo algorithm for normal and albedo calculation.
Python implementation of PhotometricStereo.m
"""

import numpy as np
import cv2
from scipy import linalg
from scipy.ndimage import generic_filter, gaussian_filter


class PhotometricStereo:
    """
    Photometric stereo class for computing surface normals and albedo
    from multiple images taken under different lighting conditions.
    """
    
    def __init__(self, images, light_directions, mask=None):
        """
        Initialize the photometric stereo processor.
        
        Parameters
        ----------
        images : ndarray
            Input images stacked along 3rd dimension, shape (height, width, num_images)
        light_directions : ndarray
            Light directions, shape (3, num_images)
        mask : ndarray, optional
            Binary mask of valid pixels, shape (height, width)
        """
        self.images = images
        self.light_directions = light_directions

                
        # If no mask provided, use all pixels
        if mask is None:
            self.mask = np.ones((images.shape[0], images.shape[1]), dtype=bool)
        else:
            self.mask = mask
        
        self.height, self.width = images.shape[0], images.shape[1]
        self.num_images = images.shape[2]
        
        # Verify that we have the right number of light directions
        assert light_directions.shape[1] == self.num_images, (
            f"Number of light directions ({light_directions.shape[1]}) must match "
            f"number of images ({self.num_images})"
        )
        

        # Results computed by process()
        self.normals = None
        self.albedo = None
    
    # def process_old(self):
    #     """
    #     Compute surface normals and albedo using photometric stereo.
    #     Exact implementation of the MATLAB algorithm.
    #     """
    #
    #     #print(f"Mask shape: {self.mask.shape}")
    #     #print(f"Mask : {self.mask}")
    #
    #     # Reshape images for processing
    #     [N1, N2, M] = self.images.shape
    #     N = N1 * N2
    #     I = np.reshape(self.images, (N, M)).T
    #     mask = np.reshape(self.mask, (N, M))
    #
    #     #print(f"Mask shape: {mask.shape}")
    #     #print(f"Mask : {mask}")
    #
    #     # Create mask index for efficient computation
    #     mask_index = np.zeros(N, dtype=int)
    #     for i in range(M):
    #         mask_index = mask_index * 2 + mask[:, i]
    #
    #     #print(f"mask_index: {mask_index}")
    #     unique_mask_indices = np.unique(mask_index)
    #
    #     # Estimate scaled normal vectors
    #     b = np.full((3, N), np.nan)
    #     #print("Unique mask indices:", unique_mask_indices)
    #     for idx in unique_mask_indices:
    #         #print(f"Processing mask index1: {idx}")
    #         # Find all pixels with this index
    #         pixel_idx = np.where(mask_index == idx)[0]
    #
    #         #print(f"Processing mask index2: {pixel_idx}")
    #
    #         # Find all images that are active for this index
    #         image_tag = mask[pixel_idx[0], :]
    #
    #         #print(f"Processing mask index3: {image_tag}")
    #
    #         if np.sum(image_tag) < 3:
    #             continue
    #
    #         # Create lighting matrix for active images
    #         Li = self.light_directions[:, image_tag]
    #
    #         # Create intensity matrix for these pixels and active images
    #         Ii = I[image_tag, :][:, pixel_idx]
    #
    #         # Compute scaled normal
    #         # Equivalent to MATLAB's Li' \ Ii
    #         b[:, pixel_idx] = np.linalg.lstsq(Li.T, Ii, rcond=None)[0]
    #
    #
    #     # Reshape and calculate albedo and unit normal vectors
    #     b = np.reshape(b.T, (N1, N2, 3))
    #     rho = np.sqrt(np.sum(b**2, axis=2))
    #
    #
    #     rho_replicated = np.tile(rho[:, :, np.newaxis], (1, 1, 3))
    #
    #     # Normalizza b
    #     n = b / rho_replicated
    #
    #     # Avoid division by zero
    #     '''
    #     n = np.zeros_like(b)
    #     valid_pixels = rho > 0
    #     for i in range(3):
    #         n_channel = b[:,:,i].copy()
    #         n_channel[valid_pixels] = n_channel[valid_pixels] / rho[valid_pixels]
    #         n[:,:,i] = n_channel
    #     '''
    #     self.normals = n #np.transpose(n, (2, 0, 1))  # Convert to (3, N1, N2)
    #     self.albedo = rho
    #
    #
    #     return self.albedo, self.normals

    def process(self):
        """
        Compute surface normals and albedo using photometric stereo.
        Exact implementation of the MATLAB algorithm.
        """

        #print(f"Mask shape: {self.mask.shape}")
        #print(f"Mask : {self.mask}")

        # Reshape images for processing
        [N1, N2, M] = self.images.shape
        N = N1 * N2
        I = np.reshape(self.images, (N, M)).T
        mask = np.reshape(self.mask, (N, M))

        #print(f"Mask shape: {mask.shape}")
        #print(f"Mask : {mask}")
    
        # Create mask index for efficient computation
        mask_index = np.zeros(N, dtype=int)
        for i in range(M):
            mask_index = mask_index * 2 + mask[:, i]
    
        #print(f"mask_index: {mask_index}")
        unique_mask_indices = np.unique(mask_index)
    
        # Estimate scaled normal vectors
        #print("Unique mask indices:", unique_mask_indices)
        b = np.full((3, N), np.nan)

        for idx in np.unique(unique_mask_indices):
            # Trova tutti i pixel con questo indice
            pixelIdx = np.where(mask_index == idx)[0]

            if len(pixelIdx) == 0:
                continue

            imageTag = mask[pixelIdx[0], :]

            if np.sum(imageTag) < 3:
                continue

            Li = self.light_directions[:, imageTag.astype(bool)]
            Ii = I[imageTag.astype(bool)][:, pixelIdx]

            try:
                b[:, pixelIdx] = np.linalg.lstsq(Li.T, Ii, rcond=None)[0]
            except np.linalg.LinAlgError:
                pass  #

    
        # Reshape and calculate albedo and unit normal vectors
        b = np.reshape(b.T, (N1, N2, 3))
        rho = np.sqrt(np.sum(b**2, axis=2))


        rho_replicated = np.tile(rho[:, :, np.newaxis], (1, 1, 3))

        # Normalizza b
        n = b / rho_replicated
    
        # Avoid division by zero
        '''
        n = np.zeros_like(b)
        valid_pixels = rho > 0
        for i in range(3):
            n_channel = b[:,:,i].copy()
            n_channel[valid_pixels] = n_channel[valid_pixels] / rho[valid_pixels]
            n[:,:,i] = n_channel
        '''
        self.normals = n
        self.albedo = rho

            
        return self.albedo, self.normals

    def process_matlab(self):
        """
        Run photometric stereo with given images I, shadow mask mask, and calibrated lighting L.

        Parameters:
            I: (N1, N2, M) array of intensity images
            mask: (N1, N2, M) boolean array, 0=shadow, 1=lit
            L: (3, M) array of lighting directions

        Returns:
            rho: (N1, N2) albedo map
            n: (N1, N2, 3) normal map
        """
        N1, N2, M = self.images.shape
        N = N1 * N2

        I_flat = self.images.reshape((N, M)).T  # shape (M, N)
        mask_flat = self.mask.reshape((N, M))

        # Create mask index
        mask_index = np.zeros(N, dtype=int)
        for i in range(M):
            mask_index = mask_index * 2 + mask_flat[:, i].astype(int)

        unique_mask_indices = np.unique(mask_index)

        # Estimate scaled normals
        b = np.full((3, N), np.nan)

        for idx in unique_mask_indices:
            pixel_idx = np.where(mask_index == idx)[0]
            if len(pixel_idx) == 0:
                continue
            image_tag = mask_flat[pixel_idx[0], :]
            if np.sum(image_tag) < 3:
                continue
            Li = self.light_directions[:, image_tag.astype(bool)]  # (3, M')
            Ii = I_flat[image_tag.astype(bool), :][:, pixel_idx]  # (M', N')

            # Solve least squares: Li.T * b = Ii
            try:
                b[:, pixel_idx] = np.linalg.lstsq(Li.T, Ii, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

        # Reshape b and compute rho, n
        b = b.T.reshape((N1, N2, 3))
        rho = np.linalg.norm(b, axis=2)

        n = np.zeros_like(b)
        # rho = inpaint_nan(rho)
        #rho = np.nan_to_num(rho, nan=0.0)
        # rho = remove_outliers(rho,5,3)
        rho = clean_nan_and_peaks(rho,peak_thresh=2)

        valid = rho > 1e-8
        n[valid] = b[valid] / rho[valid][..., np.newaxis]
        rho = np.clip(rho, 1e-8, np.max(rho))
        # n = self.clip_normals(n)

        return rho, n

    def clip_normals(self, n):
        n[:, :, 0] = np.clip(n[:, :, 0], -1, 1)
        n[:, :, 1] = np.clip(n[:, :, 1], -1, 1)
        n[:, :, 2] = np.clip(n[:, :, 2], 0, 1)
        return n

    def refine_light_directions(self):
        """
        Refine the light directions based on the computed normals and observed intensities.
        Equivalent to the PSRefineLight.m function.
        
        Returns
        -------
        ndarray
            Refined light directions
        """
        # Implementation of light direction refinement
        # This would be a port of the PSRefineLight.m algorithm
        # For now, return the original light directions
        return self.light_directions
    
    def evaluate_normals_by_image_error(self):
        """
        Evaluate the quality of normal estimation by reconstructing
        the images and computing the error.
        
        Returns
        -------
        float
            Mean square error between original and reconstructed images
        """
        if self.normals is None or self.albedo is None:
            raise ValueError("Normals and albedo must be computed first. Call process() method.")
        
        # Reshape for matrix operations
        normals_flat = self.normals.reshape(3, -1)
        albedo_flat = self.albedo.flatten()
        
        # Compute reconstructed images
        reconstructed_flat = albedo_flat * (self.light_directions.T @ normals_flat)
        
        # Original images in flat form
        original_flat = self.images.reshape(-1, self.num_images).T
        
        # Compute mean square error
        mask_flat = self.mask.flatten()
        error = np.mean(((original_flat[:, mask_flat] - reconstructed_flat[:, mask_flat]) ** 2))
        
        return error


# def inpaint_nan(array, size=3):
#     """
#     Sostituisce i NaN con la media dei vicini validi.
#     """
#     def local_valid_mean(values):
#         valid = values[~np.isnan(values) & (values != 0)]
#         if valid.size > 0:
#             return np.mean(valid)
#         else:
#             return 0  # o np.nan se vuoi tenere come NaN
#
#     return generic_filter(array, local_valid_mean, size=size, mode='mirror')
#
# def remove_outliers(array, threshold=3.0, size=3):
#     """
#     Rimuove valori anomali (z-score alto) sostituendoli con la media dei vicini.
#     """
#     mean = gaussian_filter(array, sigma=1)
#     std = np.std(array)
#
#     z = np.abs(array - mean) / (std + 1e-8)
#     outliers = z > threshold
#     array_clean = array.copy()
#
#     def local_mean(values):
#         return np.mean(values)
#
#     array_mean_local = generic_filter(array, local_mean, size=size, mode='mirror')
#     array_clean[outliers] = array_mean_local[outliers]
#
#     return array_clean


def fast_local_mean_2d(a, size):
    """
    Calcola la media locale 2D usando la somma cumulativa (integral image).
    Più veloce di gaussian_filter per kernel uniformi.
    """
    h, w = a.shape
    s = np.zeros((h + 1, w + 1), dtype=float)
    s[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)

    result = (
        s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]
    ) / (size * size)

    # Padding per riportare alla stessa dimensione
    pad = size // 2
    return np.pad(result, pad, mode='edge')


def clean_nan_and_peaks(array, window=3, peak_thresh=3.0):
    """
    Rimuove NaN e picchi anomali sostituendoli con la media locale veloce.
    """
    # Sostituisci temporaneamente NaN con 0
    array_fixed = np.nan_to_num(array, nan=0.0)
    mask_nan = np.isnan(array)

    # Calcola la media locale
    local_mean = fast_local_mean_2d(array_fixed, size=window)

    # Calcola dev. standard locale (grezza)
    local_std = np.abs(array_fixed - local_mean)

    # Soglia di picco: differenza troppo grande
    mask_peak = local_std > (peak_thresh * np.std(array_fixed))

    # Sostituisci NaN e picchi
    result = array.copy()
    result[mask_nan | mask_peak] = local_mean[mask_nan | mask_peak]

    return result

def inpaint_nan0(img):
    # cv2.inpaint lavora con 8bit o float32
    img_float = img.astype(np.float32)
    nan_mask = np.isnan(img)
    mask_uint8 = nan_mask.astype(np.uint8) * 255
    inpainted = img_float

    fs = 5
    kernel = np.ones((fs,fs), np.uint8)
    mask_dilation = cv2.dilate(mask_uint8, kernel, iterations=1)

    # # Calcola il gradiente lungo l'asse x e y
    # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    #
    # magnitude = cv2.magnitude(sobelx, sobely)
    # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # magnitude = magnitude.astype(np.uint8)


    if np.size(img.shape) == 2:
        inpainted = inpainted[mask_dilation] = 0 #cv2.inpaint(img_float, mask_dilation, fs, cv2.INPAINT_TELEA) # Fa caca'
    else:
        inpainted[:, :, :] = inpainted[mask_dilation] = 0 #cv2.inpaint(img_float[:, :, 0], mask_dilation, fs, cv2.INPAINT_TELEA) # Fa caca'

    # inpainted[nan_mask] = 0

    return inpainted

def compute_photometric_stereo(images, light_directions, mask=None):
    """
    Simplified function to compute photometric stereo.
    
    Parameters
    ----------
    images : ndarray
        Input images stacked along 3rd dimension, shape (height, width, num_images)
    light_directions : ndarray
        Light directions, shape (3, num_images)
    mask : ndarray, optional
        Binary mask of valid pixels, shape (height, width)
        
    Returns
    -------
    tuple
        (albedo, normals) - albedo values and normal vectors
    """
    ps = PhotometricStereo(images, light_directions, mask)
    return ps.process()