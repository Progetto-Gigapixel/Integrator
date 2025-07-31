import numpy as np
import cv2

class WallisFilter:
    """Implementation of the Wallis filter as a local grayscale transform."""
    
    def __init__(self, window_size=100, target_mean=127, target_std=50, a_factor=1.0, b_factor=0.2):
        self.window_size = window_size
        self.target_mean = target_mean
        self.target_std = target_std
        self.a_factor = a_factor
        self.b_factor = b_factor

    def apply(self, image):
        # Convert input to grayscale if it's color
        if len(image.shape) == 3:
            # Convert RGB to grayscale using standard coefficients
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Apply Wallis filter to grayscale image
        return self._apply_to_channel(gray)
            
    def _apply_to_channel(self, channel):
        # Convert to float32 for processing
        channel = channel.astype(np.float32)
        
        # Calculate local mean using sliding window
        kernel = np.ones((self.window_size, self.window_size), dtype=np.float32)
        kernel /= kernel.size
        local_mean = cv2.filter2D(channel, -1, kernel, borderType=cv2.BORDER_REFLECT)
        
        # Calculate local standard deviation
        local_variance = cv2.filter2D(channel**2, -1, kernel, borderType=cv2.BORDER_REFLECT) - local_mean**2
        local_std = np.sqrt(np.maximum(local_variance, 0))  # Ensure non-negative
        
        # Apply Wallis formula vectorized
        numerator = (self.target_std * (channel - local_mean))
        denominator = local_std + self.a_factor
        
        result = numerator / np.where(denominator < 1e-4, 1e-4, denominator) + \
                self.target_mean * self.b_factor + \
                local_mean * (1 - self.b_factor)
        
        # Clip and convert back to uint8
        return np.clip(result, 0, 255).astype(np.uint8)