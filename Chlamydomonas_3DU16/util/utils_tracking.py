'''
original code is Tracking_kalmanfilter.py (Commits on Apr 19, 2024)
Shishido add some code and modify original code for using in the project. 
'''

# Standard library imports
import time

# Third-party library imports
import cupy as cp

# from cupyx.profiler import benchmark
import numpy as np
import matplotlib.pyplot as plt
# import torch
from skimage.feature import peak_local_max

# from cucim.skimage.feature import peak_local_max    # For use with cupy
from scipy.ndimage import maximum_filter as maximum_filter_cpu
from scipy.ndimage import gaussian_filter as gaussian_filter_cpu
from cupyx.scipy.ndimage import maximum_filter as maximum_filter_gpu
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_gpu

# from cucim.skimage.util import crop    # Alternative cropping function for use with cupy
from skimage import io

# Local/application-specific imports
from util.utils import check_image_information, transform_for_napari

from filterpy.kalman import KalmanFilter


class Detect2Tracking:
    """
    A class for detecting and tracking objects in a 3D image stack.

    Attributes:
    - path: str
        The path to the image file.
    - device: torch.device
        The device (CPU or GPU) used for processing.

    Methods:
    - __init__(self, img_path)
        Initializes the Detect2Tracking object with the image path and device information.
    - read_img_single_vol(self, t, ch=1, do_check=True)
        Reads a single volume of the image and returns it.
    - plot_coordinate_on_mip(self, mip_img, coordinate)
        Plots the coordinates on the maximum intensity projection (MIP) image.
    - custom_peak_local_max(self, img, min_dist, detect_num, footprint, threshold, do_profile=False)
        Finds local maxima in an image and returns their coordinates and intensities.
    - custom_detect_roi(self, t, min_dist, detect_num, footprint, do_profile=False)
        Detects regions of interest (ROIs) in a single volume of the image and returns the coordinates and a record.
    - reshape_array(self, origin_coordinate_list, next_coordinate_array)
        Reshapes an array of coordinates to include an additional dimension.
    - record_peak(self, peak_record, peak)
        Appends a new set of peak coordinates to the existing record of peaks.
    """
    def __init__(self):
        """
        Initializes the Detect2Tracking object.

        Parameters:
        - img_path: str
            The path to the image file.
        """
        self.filters = {}
        self.observed_points = []  # 追跡された座標のリストを保持
        self.predicted_points = []  # 予測された座標のリストを保持

    def read_img_single_vol(self, t, ch=1, do_check=True):
        """
        Reads a single volume of the image and returns it.

        Parameters:
        - t: int
            The time index of the volume to read.
        - ch: int, optional
            The channel index of the volume to read.
        - do_check: bool, optional
            Whether to check the image information.

        Returns:
        - img: ndarray
            The single volume of the image.
        """
        try:
            img = io.imread(self.path + f"t{t}.aivia.tif")
            img = img[ch]
            print("Done read image!")
            if do_check:
                check_image_information(img)
            return img

        except FileNotFoundError as f:
            print(f"Image loading error occurs: {f}")

    def plot_coordinate_on_mip(self, mip_img, coordinate):
        """
        Plots the coordinates on the maximum intensity projection (MIP) image.

        Parameters:
        - mip_img: ndarray
            The maximum intensity projection (MIP) image.
        - coordinate: ndarray
            The coordinates to plot on the MIP image.
        """
        plt.imshow(mip_img, cmap="gray")
        plt.scatter(coordinate[:, 2], coordinate[:, 1], s=10, c="red")
        plt.show()
    
    def plot_coordinate_on_mip_with_num(self, mip_img, coordinate, save_flag, folder):
        """
        Plots the coordinates on the maximum intensity projection (MIP) image and labels each point with its index.

        Parameters:
        - mip_img: ndarray
            The maximum intensity projection (MIP) image.
        - coordinate: ndarray
            The coordinates to plot on the MIP image.
        """
        plt.imshow(mip_img, cmap="gray")
        plt.scatter(coordinate[:, 2], coordinate[:, 1], s=10, c="red")
        
        # 各点にインデックスを追加
        for i, (x, y) in enumerate(zip(coordinate[:, 2], coordinate[:, 1])):
            plt.text(x, y, str(i), color="red", fontsize=8, ha='right', va='bottom')
        
        if save_flag:
            plt.savefig(f"{folder}\detected.jpeg", format="jpeg")

        plt.show()

    
    def make_montage_img(image, gap=10):
        """
        Create a montage image by combining multiple images.

        Parameters:
        img (numpy.ndarray): The input image array with shape (nz, ny, nx).
        gap (int, optional): The gap size between images in the montage. Default is 10.

        Returns:
        numpy.ndarray: The montage image with shape (ny+nz+10, nx+nz+10).
        """
        # if len(img.shape)==4:
        #     nz, ny, nx = img[1].shape
        #     montage = np.zeros((ny+nz+10, nx+nz+10), dtype=img.dtype)
        #     montage[:ny, :nx] = np.max(img[1], axis=0)
        #     montage[ny+gap:, :nx] = np.max(img[1], axis=1)
        #     montage[:ny, nx+gap:] = np.max(img[1], axis=2).transpose()
        #     return montage
        # else:
        img =np.array(image)
        nz, ny, nx = img.shape[0], img.shape[1], img.shape[2]
        montage = np.zeros((ny+nz+10, nx+nz+10), dtype=img.dtype)
        montage[:ny, :nx] = np.max(img, axis=0)
        montage[ny+gap:, :nx] = np.max(img, axis=1)
        montage[:ny, nx+gap:] = np.max(img, axis=2).transpose()
        return montage
    
    def plot_coordinate_on_mip_montage(self, mip_img, coordinate):
        """
        Plots the coordinates on the maximum intensity projection (MIP) image.

        Parameters:
        - mip_img: ndarray
            The maximum intensity projection (MIP) image.
        - coordinate: ndarray
            The coordinates to plot on the MIP image.
        """
        img = self.make_montage_img(mip_img)
        
        plt.imshow(img, cmap="gray")
        plt.scatter(coordinate[:, 2], coordinate[:, 1], s=10, c="red")
        plt.scatter(coordinate[:, 2], coordinate[:, 1]+mip_img.shape[1]+10, s=10, c="red")
        plt.scatter(coordinate[:, 0]+mip_img.shape[2]+10, coordinate[:, 1], s=10, c="red")
        plt.show()

    def custom_peak_local_max(
        self, 
        img, 
        min_dist, 
        detect_num, 
        footprint, 
        threshold, 
        use_gpu, 
        do_profile=False
    ):
        """
        Finds local maxima in an image and returns their coordinates and intensities.

        Parameters:
        - img: ndarray
            Input image.
        - min_dist: int
            Minimum number of pixels separating peaks.
        - detect_num: int
            Number of peaks to detect.
        - footprint: ndarray, optional
            Structuring element used to define the neighborhood of each pixel.
        - threshold: float, optional
            Minimum intensity value for a peak to be considered valid.
        - do_profile: bool, optional
            Whether to profile the function execution time.

        Returns:
        - peaks_with_intensities: ndarray
            Array of shape (N, 3) containing the coordinates (x, y) and intensities of the detected peaks.
        """
        # Apply a maximum filter to the image with the specified size or footprint.
        # This creates a filtered image where each pixel's value is replaced by the maximum value
        # in its neighborhood defined by the footprint or the size.
        if use_gpu:
            xp = cp
            _local_max = maximum_filter_gpu(img, size=2 * min_dist + 1, footprint=footprint)
        else:
            xp = np
            _local_max = maximum_filter_cpu(img, size=2 * min_dist + 1, footprint=footprint)

        # Create a mask where true values correspond to pixels that are equal to the local maxima.
        # This is a way to identify potential peak locations.
        _peak_mask = img == _local_max

        # Initialize a mask to exclude peaks at the edges of the image.
        # This is done to avoid detecting peaks at the boundaries, where the analysis might be less reliable.
        edge_exclusion_mask = xp.ones_like(_peak_mask)
        edge_exclusion_mask[:min_dist, :, :] = 0
        edge_exclusion_mask[-min_dist:, :, :] = 0
        edge_exclusion_mask[:, :min_dist, :] = 0
        edge_exclusion_mask[:, -min_dist:, :] = 0
        edge_exclusion_mask[:, :, :min_dist] = 0
        edge_exclusion_mask[:, :, -min_dist:] = 0

        # Combine the peak mask with the edge exclusion mask to eliminate edge peaks.
        _peak_mask = xp.logical_and(_peak_mask, edge_exclusion_mask)

        # Find the coordinates of all potential peaks after applying the edge exclusion.
        peaks = xp.argwhere(_peak_mask)

        # Retrieve the intensity values of the detected peaks.
        intensities = img[_peak_mask]

        # Filter out peaks with intensities below the specified threshold.
        valid_idx = xp.where(intensities >= threshold)[0]

        # If more valid peaks are found than the desired number, select the top `detect_num` based on intensity.
        # Otherwise, use all valid peaks.
        if len(valid_idx) > detect_num:
            idx = xp.argsort(intensities[valid_idx])[::-1][:detect_num]
        else:
            idx = xp.argsort(intensities[valid_idx])[::-1]

        # Sort the peaks and their intensities based on the filtering and selection done above.
        sorted_peaks = peaks[valid_idx][idx]
        sorted_intensities = intensities[valid_idx][idx]

        # Combine the peak coordinates with their intensities for the final output.
        peaks_with_intensities = xp.concatenate(
            (sorted_peaks, sorted_intensities[:, xp.newaxis]), axis=1
        )

        # Optionally sort the resulting peaks by their spatial coordinates (useful for consistent ordering).
        peaks_idx = xp.argsort(peaks_with_intensities[:, 0])
        peaks_with_intensities = peaks_with_intensities[peaks_idx]

        return peaks_with_intensities

    def custom_detect_roi(
        self, 
        image, 
        t, 
        min_dist, 
        detect_num, 
        footprint, 
        use_gpu,
        do_profile=False
    ):
        """
        Detect regions of interest (ROIs) in a single volume of an image at time t,
        using custom peak detection settings.

        Parameters:
        - t (int): The time point (volume) of the image to analyze.
        - min_dist (int): The minimum distance between detected peaks, to avoid clustering.
        - detect_num (int): The maximum number of peaks to detect.
        - footprint (numpy.ndarray): The local region around each pixel considered for the peak.
        - do_profile (bool, optional): Flag to enable profiling information (timing).

        Returns:
        - peaks_with_intensities (numpy.ndarray): Array containing the detected peaks along with their intensities.
        - peak_record_initial (numpy.ndarray): Initial record of the detected peaks with a time index column prepended.

        The function first reads a single volume of the image at time t, then uses a custom peak detection method
        to identify the most prominent peaks based on the specified parameters. The detected peaks are then prepared
        for further analysis by adding a time index and initializing a record for tracking.
        """

        # Read a single volume of the image at the specified time point.
            ## original code
            # img = self.read_img_single_vol(t, ch=1)
            # print(f"Image shape: {img.shape}")
        ## shishido's code
        if use_gpu:
            xp = cp
            img = xp.asarray(image)
        else:
            xp = np
            img = image

        # Start timing the detection process if profiling is enabled.
            # start = time.time()

        # Detect peaks within the image using the specified parameters.
        # Returns coordinates and intensities of detected peaks.
        peaks_with_intensities = self.custom_peak_local_max(
            img, min_dist, detect_num, footprint, 200, use_gpu
        )  # shape: (track_num, 4)

        # Print the time taken for detection if profiling is enabled.
            # print(f"Detection complete at {time.time() - start} s")

        # Initialize a column of zeros to serve as the time index for each detected peak.
        time_index_column = xp.zeros((peaks_with_intensities.shape[0], 1), dtype=int)

        # Concatenate the time index column with the detected peaks and their intensities.
        peaks_with_intensities_reshaped = xp.hstack(
            (time_index_column, peaks_with_intensities)
        )

        # Initialize the peak record array with dimensions suitable for storing the detected peaks over time.
        peak_record_initial = xp.zeros(
            (
                1,
                peaks_with_intensities_reshaped.shape[0],
                peaks_with_intensities_reshaped.shape[1],
            )
        )

        # Insert the reshaped peaks_with_intensities data into the first entry of the peak record.
        peak_record_initial[0, :, :] = peaks_with_intensities_reshaped

        if use_gpu:
            peaks_with_intensities = xp.asnumpy(peaks_with_intensities)
            peak_record_initial = xp.asnumpy(peak_record_initial)

        return peaks_with_intensities, peak_record_initial
    
    def custom_peak_local_max_sort(
        self, 
        img, 
        min_dist, 
        detect_num, 
        footprint, 
        threshold, 
        use_gpu, 
        do_profile=False
    ):
        """
        Finds local maxima in an image and returns their coordinates and intensities.

        Parameters:
        - img: ndarray
            Input image.
        - min_dist: int
            Minimum number of pixels separating peaks.
        - detect_num: int
            Number of peaks to detect.
        - footprint: ndarray, optional
            Structuring element used to define the neighborhood of each pixel.
        - threshold: float, optional
            Minimum intensity value for a peak to be considered valid.
        - do_profile: bool, optional
            Whether to profile the function execution time.

        Returns:
        - peaks_with_intensities: ndarray
            Array of shape (N, 3) containing the coordinates (x, y) and intensities of the detected peaks.
        """
        # Apply a maximum filter to the image with the specified size or footprint.
        if use_gpu:
            xp = cp
            _local_max = maximum_filter_gpu(img, size=2 * min_dist + 1, footprint=footprint)
        else:
            xp = np
            _local_max = maximum_filter_cpu(img, size=2 * min_dist + 1, footprint=footprint)

        # Create a mask where true values correspond to pixels that are equal to the local maxima.
        _peak_mask = img == _local_max

        # Initialize a mask to exclude peaks at the edges of the image.
        edge_exclusion_mask = xp.ones_like(_peak_mask)
        edge_exclusion_mask[:min_dist, :, :] = 0
        edge_exclusion_mask[-min_dist:, :, :] = 0
        edge_exclusion_mask[:, :min_dist, :] = 0
        edge_exclusion_mask[:, -min_dist:, :] = 0
        edge_exclusion_mask[:, :, :min_dist] = 0
        edge_exclusion_mask[:, :, -min_dist:] = 0

        # Combine the peak mask with the edge exclusion mask to eliminate edge peaks.
        _peak_mask = xp.logical_and(_peak_mask, edge_exclusion_mask)

        # Find the coordinates of all potential peaks after applying the edge exclusion.
        peaks = xp.argwhere(_peak_mask)

        # Retrieve the intensity values of the detected peaks.
        intensities = img[_peak_mask]

        # Filter out peaks with intensities below the specified threshold.
        valid_idx = xp.where(intensities >= threshold)[0]

        # If no valid peaks exist after filtering, return an empty array.
        if len(valid_idx) == 0:
            return xp.array([])

        # Sort the valid peaks by intensity (in descending order).
        sorted_idx = xp.argsort(intensities[valid_idx])[::-1]

        # Select the top `detect_num` peaks based on the sorted intensities.
        if len(sorted_idx) > detect_num:
            selected_idx = sorted_idx[:detect_num]
        else:
            selected_idx = sorted_idx

        # Extract the coordinates and intensities of the selected peaks.
        sorted_peaks = peaks[valid_idx][selected_idx]
        sorted_intensities = intensities[valid_idx][selected_idx]

        # Combine the peak coordinates with their intensities for the final output.
        peaks_with_intensities = xp.concatenate(
            (sorted_peaks, sorted_intensities[:, xp.newaxis]), axis=1
        )

        return peaks_with_intensities


    def custom_detect_roi_sort(
        self, 
        image, 
        t, 
        min_dist, 
        detect_num, 
        footprint, 
        threshold,
        use_gpu,
        do_profile=False
    ):
        """
        Detect regions of interest (ROIs) in a single volume of an image at time t,
        using custom peak detection settings.

        Parameters:
        - t (int): The time point (volume) of the image to analyze.
        - min_dist (int): The minimum distance between detected peaks, to avoid clustering.
        - detect_num (int): The maximum number of peaks to detect.
        - footprint (numpy.ndarray): The local region around each pixel considered for the peak.
        - do_profile (bool, optional): Flag to enable profiling information (timing).

        Returns:
        - peaks_with_intensities (numpy.ndarray): Array containing the detected peaks along with their intensities.
        - peak_record_initial (numpy.ndarray): Initial record of the detected peaks with a time index column prepended.

        The function first reads a single volume of the image at time t, then uses a custom peak detection method
        to identify the most prominent peaks based on the specified parameters. The detected peaks are then prepared
        for further analysis by adding a time index and initializing a record for tracking.
        """

        # Read a single volume of the image at the specified time point.
            ## original code
            # img = self.read_img_single_vol(t, ch=1)
            # print(f"Image shape: {img.shape}")
        ## shishido's code
        if use_gpu:
            xp = cp
            img = xp.asarray(image)
        else:
            xp = np
            img = image

        # Start timing the detection process if profiling is enabled.
            # start = time.time()

        # Detect peaks within the image using the specified parameters.
        # Returns coordinates and intensities of detected peaks.
        peaks_with_intensities = self.custom_peak_local_max_sort(
            img, min_dist, detect_num, footprint, threshold, use_gpu
        )  # shape: (track_num, 4)

        # Print the time taken for detection if profiling is enabled.
            # print(f"Detection complete at {time.time() - start} s")

        # Initialize a column of zeros to serve as the time index for each detected peak.
        time_index_column = xp.zeros((peaks_with_intensities.shape[0], 1), dtype=int)

        # Concatenate the time index column with the detected peaks and their intensities.
        peaks_with_intensities_reshaped = xp.hstack(
            (time_index_column, peaks_with_intensities)
        )

        # Initialize the peak record array with dimensions suitable for storing the detected peaks over time.
        peak_record_initial = xp.zeros(
            (
                1,
                peaks_with_intensities_reshaped.shape[0],
                peaks_with_intensities_reshaped.shape[1],
            )
        )

        # Insert the reshaped peaks_with_intensities data into the first entry of the peak record.
        peak_record_initial[0, :, :] = peaks_with_intensities_reshaped

        if use_gpu:
            peaks_with_intensities = xp.asnumpy(peaks_with_intensities)
            peak_record_initial = xp.asnumpy(peak_record_initial)

        return peaks_with_intensities, peak_record_initial

    def reshape_array(self, origin_coordinate_list, next_coordinate_array):
        """
        Reshape an array of coordinates to include an additional dimension.

        This method takes an existing list of origin coordinates and a numpy array of next coordinates,
        then reshapes the next coordinate array to have an additional dimension, initializing
        the new dimension with zeros.

        Parameters:
        - origin_coordinate_list: list
            A list containing the original coordinates. The length of this list determines
            the first dimension of the new shape.
        - next_coordinate_array: numpy.ndarray
            An array containing the next set of coordinates. Its second dimension's size
            determines the second dimension of the new shape, with an additional offset.

        Returns:
        - next_coordinate_array_reshaped: numpy.ndarray
            The reshaped array with the new dimension added and initialized to zeros.
        """
        # Calculate the new shape based on the length of the original coordinate list
        # and the second dimension of the next coordinate array, with an additional dimension added.
        new_shape = (
            len(origin_coordinate_list),
            next_coordinate_array.shape[1] + 1,
        )  # (number of tracks, 4)

        # Initialize a new array with zeros based on the calculated shape.
        next_coordinate_array_reshaped = np.zeros(new_shape)

        # Copy the contents of the next coordinate array into the new array, starting from the second column,
        # leaving the first column as zeros. This effectively adds a new dimension to the array.
        next_coordinate_array_reshaped[:, 1:] = next_coordinate_array

        return next_coordinate_array_reshaped

    def record_peak(self, peak_record, peak):
        """
        Append a new set of peak coordinates to the existing record of peaks.

        This method takes an existing record of peaks and a new set of peak coordinates,
        then appends the new peaks to the record, expanding the record array.

        Parameters:
        - peak_record: numpy.ndarray
            The existing record of peaks, to which the new peaks will be appended.
        - peak: numpy.ndarray
            The new set of peak coordinates to append to the record. Expected to have one less dimension
            than the peak record, as it represents a single entry.

        Returns:
        - peak_record: numpy.ndarray
            The updated record of peaks with the new peaks appended.
        """
        # Append the new peak to the existing peak record. The 'None' in the indexing
        # adds a new axis to 'peak', making its dimensions compatible with 'peak_record' for concatenation.
        # This operation effectively adds a new set of coordinates as a new entry in the peak record.
        peak_record = np.concatenate((peak_record, peak[None, :, :]), axis=0)

        return peak_record

    def find_closest_to_center_where(self, indices, center):
        """
        Finds the index of the point in 'indices' that is closest to the 'center'.

        This function calculates the Euclidean distance from each point in 'indices'
        to the 'center' and returns the point (index) that is closest to the center.

        Parameters:
        - indices (np.ndarray): An array of indices (points), where each point is represented as [z, y, x].
        - center (np.ndarray): The center point, represented as [z, y, x].

        Returns:
        - np.ndarray: The coordinates of the point in 'indices' that is closest to the 'center'.
        """
        # Convert the list of indices to a NumPy array for efficient mathematical operations.
        indices_array = np.array(indices)
        # Calculate the Euclidean distance from each point in 'indices' to the 'center'.
        distances = np.linalg.norm(indices_array - center, axis=1)
        # Find the index of the point with the minimum distance to the 'center'.
        closest_idx = np.argmin(distances)
        # Return the coordinates of the closest point.
        return indices_array[closest_idx]

    def tracking_using_intensity(
        self,
        img,
        z_around,
        y_around,
        x_around,
        pre_coordinate_with_intensity,
        peak_record_with_intensity,
        threshold,
        sigma=1.0,
        do_profile=False,
    ):
        """
        Tracks the intensity peaks in an image by comparing them to previous frame intensities.

        This function crops regions around previous intensity peaks in the given image,
        computes the difference in intensity within these regions to the previous intensities,
        and updates the peak records based on the intensity differences.

        Parameters:
        - img (np.ndarray): The 3D image array in which to track intensity peaks.
        - z_around, y_around, x_around (int): The sizes of the regions to crop around each peak in the z, y, and x dimensions, respectively.
        - pre_coordinate_with_intensity (list): The list of coordinates with their corresponding intensities from the previous frame.
        - peak_record_with_intensity (list): The list to which the updated peak records will be added.
        - threshold (float): The intensity difference threshold used to consider if a peak has moved.
        - do_profile (bool, optional): If True, prints profiling information such as time taken.

        Returns:
        - list: The updated list of tracked coordinates with their intensities.
        - list: The updated peak record with intensities.
        """
        if do_profile:
            print("------------------")
            start = time.time()

        # Extract the shape of the image for boundary checks.
        shapex, shapey, shapez = img.shape[2], img.shape[1], img.shape[0]

        # Unpack the previous coordinates and their intensities.
        pz, py, px, pre_intensity = np.array(pre_coordinate_with_intensity).T
        if do_profile:
            print(f"pre_intensity: {pre_intensity}")

        # Determine the start and end indices for cropping around each peak, ensuring they are within the image bounds.
            # Hiroki's code
            # crop_z_start = np.maximum((pz - z_around), 1)
            # crop_z_end = np.where(
            #     crop_z_start == 1, 1 + 2 * z_around, np.minimum(pz + z_around, shapez)
            # )
            # crop_y_start = np.maximum((py - y_around), 1)
            # crop_y_end = np.where(
            #     crop_y_start == 1, 1 + 2 * y_around, np.minimum(py + y_around, shapey)
            # )
            # crop_x_start = np.maximum((px - x_around), 1)
            # crop_x_end = np.where(
            #     crop_x_start == 1, 1 + 2 * x_around, np.minimum(px + x_around, shapex)
            # )
        # Kasagi's code
        crop_z_start = np.maximum((pz - z_around).astype(np.int16), 0)
        crop_z_end = np.minimum((pz + z_around + 1).astype(np.int16), shapez)
        crop_y_start = np.maximum((py - y_around).astype(np.int16), 0)
        crop_y_end = np.minimum((py + y_around + 1).astype(np.int16), shapey)
        crop_x_start = np.maximum((px - x_around).astype(np.int16), 0)
        crop_x_end = np.minimum((px + x_around + 1).astype(np.int16), shapex)

        # Crop regions around each previous peak for the current frame.
        cropped_images = [
            img[z_start:z_end, y_start:y_end, x_start:x_end]
            for z_start, z_end, y_start, y_end, x_start, x_end in zip(
                crop_z_start,
                crop_z_end,
                crop_y_start,
                crop_y_end,
                crop_x_start,
                crop_x_end,
            )
        ]
        
        # Apply Gaussian filter to each cropped image
        smoothed_cropped_images = [
        gaussian_filter_cpu(cropped_img, sigma=sigma)
        for cropped_img in cropped_images
    ]

        # Compute the absolute difference in intensities within the cropped regions to the previous frame's intensities.
        intensity_diffs = [
            np.abs(cropped_img.astype(np.float32) - intensity.astype(np.float32))
            for cropped_img, intensity in zip(smoothed_cropped_images, pre_intensity)
        ]

        min_diff_indices = []
        for intensity_diff, cropped_img in zip(intensity_diffs, smoothed_cropped_images):
            # Find indices where the intensity difference is below the threshold.
            below_threshold = np.argwhere(intensity_diff <= threshold)
            if below_threshold.size > 0:
                # Calculate the center of the cropped region.
                center = np.array(cropped_img.shape) // 2
                # Find the index closest to the center among those below the threshold.
                closest_idx = self.find_closest_to_center_where(below_threshold, center)
                min_diff_indices.append(closest_idx)
            else:
                # If no indices are below the threshold, choose the index with the minimum intensity difference.
                min_diff_idx = np.unravel_index(
                    np.argmin(intensity_diff, axis=None), cropped_img.shape
                )
                min_diff_indices.append(np.array(min_diff_idx))

        # Convert the indices of minimum difference back to the original coordinate system and record them with their intensities.
        tracked_coordinates_with_intensity = [
            np.array(
                [
                    max(0, min(shapez-1, z_start + min_diff_idx[0])),
                    max(0, min(shapey-1, y_start + min_diff_idx[1])),
                    max(0, min(shapex-1, x_start + min_diff_idx[2])),
                    # z_start + min_diff_idx[0],
                    # y_start + min_diff_idx[1],
                    # x_start + min_diff_idx[2],
                    cropped_img[tuple(min_diff_idx)],
                ]
            )
            for (z_start, y_start, x_start), min_diff_idx, cropped_img in zip(
                zip(pz - z_around, py - y_around, px - x_around),
                min_diff_indices,
                smoothed_cropped_images,
            )
        ]

        if do_profile:
            print(
                f"tracked_coordinates_with_intensity: {len(tracked_coordinates_with_intensity)}"
            )

        # Update the peak record with the newly tracked coordinates and their intensities.
        next_coordinate_array = np.asarray(tracked_coordinates_with_intensity)
        next_coordinate_array_reshaped = self.reshape_array(
            tracked_coordinates_with_intensity, next_coordinate_array
        )
        peak_record_with_intensity = self.record_peak(
            peak_record_with_intensity, next_coordinate_array_reshaped
        )

        if do_profile:
            print(f"Total time: {time.time() - start} s")
            print("------------------")

        return tracked_coordinates_with_intensity, peak_record_with_intensity

    def initialize_kalman_filters(self, initial_coordinates):
        for idx, coord in enumerate(initial_coordinates):
            kf = KalmanFilter(dim_x=8, dim_z=4)
            kf.x[:4] = coord[:4].reshape(-1, 1)  # initial x, y, z, intensity
            kf.x[4:] = 0  # initial velocities and intensity change rate set to 0
            kf.F = np.array(
                [
                    [1, 0, 0, 0, 1, 0, 0, 0],  # x position update
                    [0, 1, 0, 0, 0, 1, 0, 0],  # y position update
                    [0, 0, 1, 0, 0, 0, 1, 0],  # z position update
                    [0, 0, 0, 1, 0, 0, 0, 1],  # intensity update
                    [1, 0, 0, 0, 1, 0, 0, 0],  # x velocity
                    [0, 1, 0, 0, 0, 1, 0, 0],  # y velocity
                    [0, 0, 1, 0, 0, 0, 1, 0],  # z velocity
                    [0, 0, 0, 1, 0, 0, 0, 1],  # intensity change rate
                ]
            )
            kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],  # measure x
                    [0, 1, 0, 0, 0, 0, 0, 0],  # measure y
                    [0, 0, 1, 0, 0, 0, 0, 0],  # measure z
                    [0, 0, 0, 1, 0, 0, 0, 0],  # measure intensity
                ]
            )
            kf.R *= 0.1  # Measurement noise
            kf.P *= 10  # Initial estimation error covariance
            kf.Q *= 0.1  # Process noise
            self.filters[idx] = kf

    def update_kalman_filters(self, new_observations):
        for idx, obs in enumerate(new_observations):
            if idx in self.filters:
                kf = self.filters[idx]
                kf.predict()
                kf.update(
                    np.array(obs[:4]).reshape(-1, 1)
                )  # Only update x, y, z, intensity
            else:
                self.filters[idx] = self.create_new_kalman_filter(obs[:4])

    def track_and_update(
        self,
        img,
        z_around,
        y_around,
        x_around,
        peak_record_with_intensity,
        threshold,
        sigma,
        do_profile=False,
    ):
        if do_profile:
            start_time = time.time()

        # Step 1: Predict next positions using Kalman Filters and collect the predicted state
        predicted_positions = []
        for kf in self.filters.values():
            kf.predict()
            predicted_positions.append(
                kf.x[:4].flatten()
            )  # Get the predicted x, y, z positions and intensity
        self.predicted_points.append(predicted_positions)  # 予測された座標を記録

        predicted_coordinates = np.array(
            predicted_positions
        )  # Convert list to array for further processing

        # Step 2: Use the predicted positions as the basis for intensity tracking
        tracked_coordinates_with_intensity, peak_record = self.tracking_using_intensity(
            img=img,
            z_around=z_around,
            y_around=y_around,
            x_around=x_around,
            pre_coordinate_with_intensity=predicted_coordinates,  # Ensure this includes x, y, z, and intensity
            peak_record_with_intensity=peak_record_with_intensity,
            threshold=threshold,
            sigma=sigma,
            do_profile=do_profile,
        )
        self.observed_points.append(
            tracked_coordinates_with_intensity
        )  # 追跡された座標を記録

        # Step 3: Update Kalman Filters with the new measurements
        for idx, (coords, kf) in enumerate(
            zip(tracked_coordinates_with_intensity, self.filters.values())
        ):
            measurement = np.array(
                [coords[0], coords[1], coords[2], coords[3]]
            ).reshape(
                -1, 1
            )  # Reshape for Kalman filter
            kf.update(measurement)

        if do_profile:
            print(f"Tracking and updating took {time.time() - start_time} seconds")

        return tracked_coordinates_with_intensity, peak_record


# if __name__ == "__main__":

#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#     device = torch.device("cpu")
#     print(device)
#     torch.no_grad()  # for memory saving

#     path = "/crest_image_data/20230809_184014tdTomato_brightaiviatif_20volume/20230809-184014tdTomato bright_"

#     img_t0 = io.imread(path + "t0.aivia.tif")
#     print(img_t0.shape)  # (channel, z, y, x)

#     d2t = Detect2Tracking(img_path=path)
#     next2, peak2 = d2t.custom_detect_roi(
#         t=0,
#         min_dist=1,
#         detect_num=10,
#         footprint=np.ones((40, 40, 40), dtype="bool"),
#         do_profile=True,
#     )
#     # initial_roi = next2.copy()
#     # d2t.plot_coordinate_on_mip(np.max(img_t0[1], axis=0), next2)
#     d2t.initialize_kalman_filters(next2)

#     for i in range(1, 20):
#         print(f"t{i}")
#         img = d2t.read_img_single_vol(i, ch=1, do_check=False)
#         d2t.initialize_kalman_filters(next2)
#         next2, peak2 = d2t.track_and_update(
#             img=img,
#             z_around=2,
#             y_around=7,
#             x_around=7,
#             peak_record_with_intensity=peak2,
#             threshold=0.5,
#         )

#     fd_img = [
#         np.max(io.imread(path + f"t{i}.aivia.tif")[1].astype(np.uint16), axis=0)
#         for i in range(20)
#     ]

#     napari_array_5d = transform_for_napari(peak2)
#     napari_array = napari_array_5d[:, :5]

#     # 各フレームごとに追跡データと画像をプロット
#     for i in range(20):
#         frame_data = napari_array[napari_array[:, 1] == i]

#         fig, ax = plt.subplots()
#         ax.imshow(fd_img[i], cmap="gray")  # 背景としての画像
#         ax.scatter(
#             frame_data[:, 4], frame_data[:, 3], c="red", s=50, edgecolor="white"
#         )  # トラッキングデータのプロット

#         # ラベルとタイトルの追加
#         ax.set_xlabel("X Position")
#         ax.set_ylabel("Y Position")
#         ax.set_title(f"Frame {i}: Tracking Overlay")

#         # グリッドの非表示
#         ax.grid(False)

#         # 画像の保存または表示
#         plt.savefig(f"{i}.png")
#         plt.close()
