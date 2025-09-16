from ..data import RadData
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk, square
from skimage.util import img_as_ubyte
from warnings import warn
"""
This file implements some measurement functions on a paired image and segmentation mask.

It makes measurements, which can be used to calculate e.g. the mean T1 value in a segmented region.

It has the following features:

1. Filter segmentation areas based on a list of integers. If not specified, handle each class separately.
2. Rejection of single pixels from the calculation based on the 3MAD method (median absolute deviation from the median).
3. Reject slices based on the minimum size of the segmented region.
4. Optionally, reject slices based on the fraction of low-entropy pixels in the segmented region.


"""


class Measurement:
    """Class to perform Zbinden-style measurements on a paired image and segmentation mask.
    
    Attributes:
        image (RadData): The input image data (e.g., T1 map).
        segmentation (RadData): The corresponding segmentation mask. May contain multiple labels, as integer values (not one-hot).
        labels (list[int], optional): List of integer labels to measure. If None, all unique labels in the segmentation are used.
        merge_labels (bool, optional): Whether to merge all specified labels into one region for measurement. Default is False.
        min_roi_size (int, optional): Minimum number of pixels for the segmented region to be considered valid. Default is 50.
                On slices where the segmented region is smaller, all measurements are skipped. (it's quite naive)
        entropy_threshold (float, optional): Threshold for local entropy. Above this value, a local entropy value is considered high,
                and below, low. Default is None, meaning no entropy-based rejection is performed.
                Lukas Zbinden found a good value at 2.1
        low_entropy_fraction (float, optional): Fraction of low-entropy pixels in the segmented region above which the slice is rejected.
                This is based on the intuition that local entropy should be fairly low over the T1/T2 map on the liver.
                It may not apply equally well to everything. Default is None, meaning no entropy-based rejection is performed.
                Lukas Zbinden found a good value at 5-10% 
                    NOTE: This value actually does not align with the motivation at all.. If we would go for a homogeneous region,
                            we would want a reasonably high fraction of low entropy pixels. I guess this helps to only reject slices with
                            artifacts. This also makes me think that they did very little qualitative evaluation.
        reject_outliers (bool, optional): Whether to reject voxel outliers based on the 3MAD method. Default is True. This
                removes those voxels from the calculation of the mean and stddev.
    """

    def __init__(self,
                 image: RadData,
                 segmentation: RadData,
                 labels: list[int] | int | None = None,
                 merge_labels: bool = False,
                 min_roi_size: int = 50,
                 entropy_threshold: float | None = None,
                 low_entropy_fraction_threshold: float | None = None,
                 reject_outliers: bool = True):
        if image.shape != segmentation.shape:
            raise ValueError(f"Image and segmentation must have the same shape. Got {image.shape} and {segmentation.shape}")
        if image.ndim != 3:
            raise ValueError("Image and segmentation must be 3D")
        self.image = image
        self.labels = self._init_labels(labels)
        self.segmentation = self._merge_labels(segmentation) if merge_labels else segmentation
        self.labels = [1] if merge_labels else self.labels  # if merging, only use label 1
        self.min_roi_size = min_roi_size
        self.entropy_threshold = entropy_threshold
        self.low_entropy_fraction_threshold = low_entropy_fraction_threshold
        self.reject_outliers = reject_outliers

    def _init_labels(self, labels: list[int] | None) -> list[int]:
        if labels is None:
            unique_labels = np.unique(self.segmentation)
            # remove background label 0
            unique_labels = unique_labels[unique_labels != 0]
            return unique_labels.tolist()
        elif isinstance(labels, int):
            return [labels]
        elif isinstance(labels, list):
            return labels
        else:
            raise ValueError("Labels must be a list of integers, a single integer, or None")
        
    def _merge_labels(self, segm_slice: RadData) -> RadData:
        merged = np.isin(segm_slice, self.labels).astype(np.int16)
        return RadData(merged, affine=segm_slice.affine, header=segm_slice.header)
    
    @staticmethod
    def _filter_by_label(segm_slice: RadData, label: int) -> RadData:
        # return a binary mask where the segmentation equals the label
        filtered = (segm_slice == label).astype(np.int16)
        return RadData(filtered, affine=segm_slice.affine, header=segm_slice.header)
    
    @staticmethod
    def _roi_size_over_threshold(segm_slice: RadData, min_size: int) -> bool:
        assert segm_slice.shape[segm_slice.slice_dim] == 1, "segm_slice must be a 2D slice"

        assert np.unique(segm_slice).size <= 2, "segm_slice must be binary"
        
        return np.sum(segm_slice) >= min_size
    
    @staticmethod
    def _zbinden_normalize(image: RadData) -> RadData:
        """Normalize the image to [0, 1] range, as done by Zbinden.

        Args:
            image (RadData): The input image slice.
        Returns:
            RadData: The normalised image slice.
        """
        img_min = np.min(image)
        img_max = np.max(image)
        if img_max - img_min == 0:
            return RadData(np.zeros_like(image), affine=image.affine, header=image.header)
        normalized = (image - img_min) / (img_max - img_min)
        return RadData(normalized, affine=image.affine, header=image.header)

    @staticmethod
    def _calculate_local_entropy(image_slice: RadData, window_size: int = 3, neighbourhood_type: str = "square") -> RadData:
        """Calculate the local entropy of the image slice using a window.

        Copied from Zbinden's code, based on skimage.
        
        Args:
            image_slice (RadData): The input image slice.
            window_size (int, optional): The size of the window to use for calculating local entropy. Default is 3.
            neighbourhood_type (str, optional): The type of neighbourhood to use. Default is "square"
                Other option is "disk"

        Returns:
            RadData: The local entropy map of the image slice.
        
        Notes:
            - This is a simple implementation using a 3x3 window. More advanced methods may be used.
            - The edges are handled by padding with zeros, which may not be ideal.
        """
        normalized_image = Measurement._zbinden_normalize(image_slice)
        # squeeze here because skimage expects 2D images
        normalized_image = img_as_ubyte(normalized_image).squeeze()
        if neighbourhood_type == 'square':
            selem = square(window_size)
            # selem = np.ones((block_size, block_size))
        elif neighbourhood_type == 'disk':
            assert window_size == 3, f"disk needs be defined differently than square, i.e. square(3) != disk(3)"
            warn("Lukas: using disk(2) but block_size == 3!")
            selem = disk(2)
        else:
            raise ValueError(f"Lukas: Unknown neighborhood type: {neighbourhood_type}")
        entropy_image = entropy(normalized_image, selem)

        return RadData(entropy_image[:, :, np.newaxis], affine=image_slice.affine, header=image_slice.header)
    
    def _entropy_check(self, image_slice: RadData, segm_slice: RadData) -> bool:
        """Check if the fraction of low-entropy pixels in the segmented region is above the threshold.

        Args:
            image_slice (RadData): The input image slice.
            segm_slice (RadData): The corresponding binary segmentation slice.
        Returns:
            bool: True if the fraction of low-entropy pixels is above the threshold, False otherwise
        """

        assert self.entropy_threshold is not None, "entropy_threshold must be set to use this function"
        assert self.low_entropy_fraction_threshold is not None, "low_entropy_fraction must be set to use this function"
        assert segm_slice.shape[segm_slice.slice_dim] == 1, "segm_slice must be a 2D slice"
        assert np.unique(segm_slice).size <= 2, "segm_slice must be binary"

        entropy_map = self._calculate_local_entropy(image_slice)
        low_entropy_mask = entropy_map <= self.entropy_threshold
        segmented_low_entropy = np.logical_and(low_entropy_mask, segm_slice.astype(bool))
        fraction_low_entropy = np.sum(segmented_low_entropy) / np.sum(segm_slice)

        return fraction_low_entropy >= self.low_entropy_fraction_threshold
    
    @staticmethod
    def _mask_by_segmentation(image_slice: RadData, segm_slice: RadData) -> RadData:
        """Mask the image slice by the segmentation slice.

        Args:
            image_slice (RadData): The input image slice.
            segm_slice (RadData): The corresponding binary segmentation slice.
        Returns:
            RadData: The masked image slice, with NaN values where the segmentation is 0.
        """
        assert segm_slice.shape == image_slice.shape, "segm_slice and image_slice must have the same shape"
        assert segm_slice.shape[segm_slice.slice_dim] == 1, "segm_slice must be a 2D slice"
        assert np.unique(segm_slice).size <= 2, "segm_slice must be binary"

        # set to nan wherever segm_slice is 0
        masked = np.where(segm_slice, image_slice, np.nan)
        return RadData(masked, affine=image_slice.affine, header=image_slice.header)
    
    @staticmethod
    def _remove_outliers(masked_image: RadData, threshold: float = 3.0, mode: str = 'mad') -> RadData:
        """Remove outliers from the intensities using the 3MAD method.

        Args:
            masked_image (RadData): The input masked image slice, with nan values where the segmentation is 0.
            threshold (float, optional): The threshold for the MAD method. Default is 3.0.
            mode (str, optional): The method to use for outlier detection. Currently only 'mad' is implemented. Default is 'mad'.
        Returns:
            RadData: The intensities with outliers removed (set to NaN).
        
        Notes:
            - The 3MAD method is robust to outliers and works well for skewed distributions.
            - Other methods may be implemented in the future.
        """
        assert mode == 'mad', "Currently only 'mad' mode is implemented"

        intensities = masked_image.flatten()
        median = np.nanmedian(intensities)
        mad = np.nanmedian(np.abs(intensities - median))
        if mad == 0:
            return RadData(intensities, affine=None, header=None)  # no outliers to remove
        lower_bound = median - threshold * mad
        upper_bound = median + threshold * mad
        filter_mask = (masked_image < lower_bound) | (masked_image > upper_bound)
        filtered = masked_image.copy()
        filtered[filter_mask] = np.nan
        return RadData(filtered, affine=masked_image.affine, header=masked_image.header)
    
    def _get_intensities_label(self, img_slice: RadData, segm_slice: RadData, label: int, debug: bool = False, debug_slice_idx: int = 0) -> np.ndarray:
        """Get the intensities for a specific label in the segmented region of the given slice.
        
        Designed to work on a single slice, for a single label.

        Args:
            img_slice (RadData): The input image slice.
            segm_slice (RadData): The corresponding segmentation slice.
            label (int): The label to extract intensities for.
            debug (bool, optional): Whether to print debug information and save intermediate results. Default is False.
            debug_slice_idx (int, optional): The index of the slice being processed, for debug-printing purposes. Default is 0.
        """
        # filter the segmentation to only include the current label
        filtered_segm = self._filter_by_label(segm_slice, label)

        if debug:
            print(f"Filtered segmentation for label {label} has {np.sum(filtered_segm)} pixels")
            filtered_segm.save(f"debug/debug_segm_label_{label}_slice_{debug_slice_idx}.nii.gz")

        # check if the size of the segmented region is above the minimum size
        if not self._roi_size_over_threshold(filtered_segm, self.min_roi_size):
            if debug:
                print(f"Skipping label {label} on slice {debug_slice_idx} due to small ROI size")
            return np.array([])  # return empty array for this label

        if debug:
            print(f"Label {label} on slice {debug_slice_idx} passed ROI size check")

        # optionally, calculate the local entropy and check the fraction of low-entropy pixels
        if self.entropy_threshold is not None and self.low_entropy_fraction_threshold is not None:
            if not self._entropy_check(img_slice, filtered_segm):
                if debug:
                    print(f"Skipping label {label} on slice {debug_slice_idx} due to high entropy fraction")
                return np.array([])  # return empty array for this label
            if debug:
                print(f"Label {label} on slice {debug_slice_idx} passed entropy check")

        # extract the intensities from the image corresponding to the segmented region
        masked_intensities = self._mask_by_segmentation(img_slice, filtered_segm)

        if debug:
            print(f"Extracted {np.sum(~np.isnan(masked_intensities))} intensities for label {label} on slice {debug_slice_idx}")
            masked_intensities.save(f"debug/debug_masked_intensities_label_{label}_slice_{debug_slice_idx}.nii.gz")

        # optionally, remove outliers using the 3MAD method
        if self.reject_outliers:
            filtered_intensities = self._remove_outliers(masked_intensities)
            if debug:
                num_outliers = len(masked_intensities[~np.isnan(masked_intensities)]) - len(filtered_intensities[~np.isnan(filtered_intensities)])
                print(f"Removed {num_outliers} outliers for label {label} on slice {debug_slice_idx}")
                filtered_intensities.save(f"debug/debug_filtered_intensities_label_{label}_slice_{debug_slice_idx}.nii.gz", set_nan_to_zero=True)
    
        # return the intensities as a 1D array, removing NaNs
        return filtered_intensities[~np.isnan(filtered_intensities)].flatten()  # return as 1D array

    
    def get_intensities_slice(self, slice_idx: int, debug: bool = False) -> dict[int, np.ndarray]:
        """Get the intensities that are in the segmented region of the given slice index.
        
        Args:
            slice_idx (int): The index of the slice to process.

        Returns:
            dict[int, np.ndarray]: A dictionary mapping each label to the intensities in that region.
                The intensities are a 1D numpy array.


        Notes:
            - Zbinden did everything per-slice, including calculating the entropy. This may make sense, since 
              the slice has a lot finer resolution than the through-plane direction.

        Steps:
            - For each label:
                - Extract the region corresponding to that label.
                - Check if the region size is above the minimum size. If not, skip this label.
                - Optionally, calculate the local entropy and check if the fraction of low-entropy pixels is above the threshold.
                  If not, skip this label.
                - Extract the intensities from the image corresponding to the segmented region.
                - Optionally, remove outliers using the 3MAD method.
                - Store the intensities in a dictionary mapping label to intensities. (and convert to a 1D array)

        Between each step, we should allow saving the intermediate results for debugging purposes.
        """

        if slice_idx < 0 or slice_idx >= self.image.shape[2]:
            raise IndexError(f"slice_idx {slice_idx} is out of bounds for image with shape {self.image.shape}")

        img_slice = self.image[:, :, slice_idx]
        segm_slice = self.segmentation[:, :, slice_idx]

        if debug:
            print(f"Processing slice {slice_idx}")
            print(f"Image slice shape: {img_slice.shape}, dtype: {img_slice.dtype}")
            print(f"Segmentation slice shape: {segm_slice.shape}, dtype: {segm_slice.dtype}, unique labels: {np.unique(segm_slice)}")
            img_slice.save(f"debug/debug_img_slice_{slice_idx}.nii.gz")
            segm_slice.save(f"debug/debug_segm_slice_{slice_idx}.nii.gz")

        results = {}

        for label in self.labels:
            if debug:
                print(f"Processing label {label} on slice {slice_idx}")
            intensities = self._get_intensities_label(img_slice, segm_slice, label, debug=debug, debug_slice_idx=slice_idx)
            results[label] = intensities

        return results
    
    def get_intensities_full(self, debug: bool = False) -> dict[int, np.ndarray]:
        """Get the intensities that are in the segmented regions for all slices.

        Args:
            debug (bool, optional): Whether to print debug information and save intermediate results. Default is False.
        Returns:
            dict[int, np.ndarray]: A dictionary mapping each label to the intensities in that region
                The intensities are a 1D numpy array.
        """

        all_results = {label: [] for label in self.labels}

        for slice_idx in range(self.image.shape[2]):
            if debug:
                print(f"Processing slice {slice_idx}/{self.image.shape[2]-1}")
            slice_results = self.get_intensities_slice(slice_idx, debug=debug)
            for label, intensities in slice_results.items():
                all_results[label].append(intensities)

        # concatenate the lists of arrays into a single array per label
        for label in all_results:
            if all_results[label]:  # only concatenate if there are any arrays
                all_results[label] = np.concatenate(all_results[label])
            else:
                all_results[label] = np.array([])  # no intensities found for this label

        return all_results
    
    def get_mean_intensities_full(self, return_std: bool = False, debug: bool = False) -> dict[int, float]:
        """Get the mean intensities that are in the segmented regions for all slices.

        Args:
            debug (bool, optional): Whether to print debug information and save intermediate results. Default is False.
        Returns:
            dict[int, float]: A dictionary mapping each label to the mean intensity in that region.
        """
        
        all_intensities = self.get_intensities_full(debug=debug)
        mean_intensities = {}
        std_intensities = {}
        for label, intensities in all_intensities.items():
            if intensities.size > 0:
                mean_intensities[label] = float(np.mean(intensities))
                if return_std:
                    std_intensities[label] = float(np.std(intensities))
            else:
                mean_intensities[label] = float('nan')  # no intensities found for this label
                if return_std:
                    std_intensities[label] = float('nan')
        if return_std:
            return mean_intensities, std_intensities

        return mean_intensities