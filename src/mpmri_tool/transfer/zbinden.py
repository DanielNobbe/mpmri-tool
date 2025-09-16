import numpy as np
import nibabel as nib
from collections import defaultdict
from ..data import RadData, get_affine_transform
from warnings import warn
from functools import cached_property
"""
In this file, the main function will be a transfer function, to transfer a segmentation from src to target image.

Or should we make it a class?
--> CUrrently it's very functional, but we could combine functions into a class.
When we start doing multiple slices, it may make sense to make a class

"""


class TransferSlice:
    """Class to handle transfer on a slice.
    
    Attributes:
        source_img: RadData: The source image from which to transfer the segmentation.
        target_img: RadData: The target image to which to transfer the segmentation.
        source_seg: RadData: The segmentation in the source image space.
        transform_matrix: np.ndarray: The affine transformation matrix from source to target. Does not need to be specified.


    
    """

    def __init__(self, source_img: RadData, source_seg: RadData, target_img: RadData):
        self.source_img = source_img
        self.target_img = target_img
        self.source_seg = source_seg
        
        self.transform_matrix = get_affine_transform(source_img, target_img)

    def get_target_slice_index(self, source_slice_index):
        # Create a homogeneous coordinate for the source slice index
        source_coord = np.array([0, 0, source_slice_index])  # Assuming x=0, y=0 for simplicity
        # TODO: this coordinate works fine if all slices are parallel, which is usually the case.
        warn("Assuming x=0, y=0 for slice coordinate transformation. This only works if slices are parallel.")

        # Apply the transformation matrix
        target_coord = nib.affines.apply_affine(self.transform_matrix, source_coord)
        
        # Extract the z-coordinate (slice index) and round to nearest integer
        target_slice_index = int(round(target_coord[2]))

        if target_slice_index < 0 or target_slice_index >= self.target_img.shape[2]:
            raise ValueError(f"Transformed slice index {target_slice_index} is out of bounds for target image with {self.target_img.shape[2]} slices.")
        
        return target_slice_index
    
    @cached_property
    def inverse_transform_matrix(self):
        return np.linalg.inv(self.transform_matrix)
    
    def get_source_slice_index(self, target_slice_index):
        # Create a homogeneous coordinate for the target slice index
        target_coord = np.array([0, 0, target_slice_index])  # Assuming x=0, y=0 for simplicity
        warn("Assuming x=0, y=0 for slice coordinate transformation. This only works if slices are parallel.")

        # Apply the inverse transformation matrix
        inv_transform_matrix = self.inverse_transform_matrix
        source_coord = nib.affines.apply_affine(inv_transform_matrix, target_coord)

        # Extract the z-coordinate (slice index) and round to nearest integer
        source_slice_index = int(round(source_coord[2]))

        if source_slice_index < 0 or source_slice_index >= self.source_img.shape[2]:
            raise ValueError(f"Transformed slice index {source_slice_index} is out of bounds for source image with {self.source_img.shape[2]} slices.")
        
        return source_slice_index

    
    # TODO:: Make an alternative using nibabel resample_from_to function
    # uses interpolation, so probably is a lot better..

    def get_raw_target_coords(self, source_segm_slice: RadData, source_slice_index: int) -> RadData:
        height, width, _ = source_segm_slice.shape
        # i_coords_src, j_coords_src = np.meshgrid(np.arange(width), np.arange(height))
        i_coords_src, j_coords_src = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        i_coords_src = i_coords_src.flatten()
        j_coords_src = j_coords_src.flatten()

        # 3D coords
        # Create source coordinate array for batch transformation
        src_coords_3d = np.column_stack([
            i_coords_src, 
            j_coords_src, 
            np.full(len(i_coords_src), source_slice_index)
        ])

        target_coords_raw = nib.affines.apply_affine(self.transform_matrix, src_coords_3d)  # Vectorized coordinate transformation
        # why are we not using numpy here?

        return target_coords_raw, (i_coords_src, j_coords_src)  # second is source_coords_raw
    
    def filter_valid_coords(self, raw_target_coords: np.ndarray, raw_source_coords: np.ndarray) -> dict[str, np.ndarray]:
        """Filter out coordinates and conflics."""
        # TODO: This could work better when incorporating the distance in the mapping somehow
        # (there must be a way to use matrices to get the distances directly)
        i_coords_src, j_coords_src = raw_source_coords
        i_raw, j_raw, k_raw = raw_target_coords.T

        target_coords_rounded = np.round(raw_target_coords).astype(int)
        # Clip coordinates to be within target image bounds
        target_coords_clipped = np.clip(target_coords_rounded, [0, 0, 0], np.array(self.target_img.shape) - 1)
        i_tgt, j_tgt, k_tgt = target_coords_clipped.T

        coord_dict = defaultdict(list)
        for idx, coord in enumerate(zip(i_tgt, j_tgt)):
            coord_dict[coord].append(idx)

        # now iterate over the dictionary, and for each list of indices, find the raw coordinate closest to the rounded coordinate
        unique_indices = []
        for coord, indices in coord_dict.items():
            if len(indices) == 1:
                unique_indices.append(indices[0])
            else:
                rounded_coord = np.array(coord)
                raw_coords = raw_target_coords[indices, :2]  # only x and y
                distances = np.linalg.norm(raw_coords - rounded_coord, axis=1)
                closest_index = indices[np.argmin(distances)]
                unique_indices.append(closest_index)
        # note: there may be an easier way if we track the distance while doing the mapping/rounding
        unique_indices = np.array(unique_indices)
        i_tgt = i_tgt[unique_indices]
        j_tgt = j_tgt[unique_indices]
        # k_tgt = k_tgt[unique_indices]  # not needed, since we are only mapping to a single slice
        i_coords_src = i_coords_src[unique_indices]
        j_coords_src = j_coords_src[unique_indices]

        return {
            "target_coords_i": i_tgt,
            "target_coords_j": j_tgt,
            "source_coords_i": i_coords_src,
            "source_coords_j": j_coords_src
        }
    # TODO: Make this return a custom type

    def create_target_segmentation(self, mapped_coords: dict[str, np.ndarray], source_segm_slice: RadData, target_img_slice: RadData) -> RadData:
        tgt_segm_slice = np.zeros(target_img_slice.shape, dtype=source_segm_slice.dtype)
        # TODO: Add logic to RadData class to handle affine matrix under slicing etc
        tgt_segm_slice = RadData(tgt_segm_slice, affine=target_img_slice.affine, header=target_img_slice.header)
        # Map the segmentation values to the target slice


        tgt_segm_slice[
            mapped_coords["target_coords_i"], mapped_coords["target_coords_j"]
        ] = source_segm_slice[mapped_coords["source_coords_i"].flatten(), mapped_coords["source_coords_j"].flatten()]

        return tgt_segm_slice

    def transfer(self, source_slice_index: int, return_image: bool = False) -> RadData:
        """Transfer the segmentation from source to target image for a given slice index.

        Args:
            src_slice_index (int): The slice index in the source image to transfer.

        It will find the closest corresponding slice in the target image, and then transfer the segmentation,
        based on world coordinates.
        """

        # Get the corresponding slice index in the target image
        tgt_slice_index = self.get_target_slice_index(source_slice_index)

        # Extract the source slice and segmentation
        # source_slice = self.source_img[:, :, source_slice_index]
        source_seg_slice = self.source_seg[:, :, source_slice_index]

        # Extract the target slice
        target_slice = self.target_img[:, :, tgt_slice_index]

        # Now we need to resample the source segmentation slice to the target slice space.
        # We can use nibabel's resample_from_to function for this, but let's first use our own baseline
        # (based on Lukas Zbinden's code)

        raw_target_coords, raw_source_coordinates = self.get_raw_target_coords(source_seg_slice, source_slice_index)

        filtered_coords = self.filter_valid_coords(raw_target_coords, raw_source_coordinates)

        target_segm_slice = self.create_target_segmentation(filtered_coords, source_seg_slice, target_slice)
        if return_image:
            return target_slice, target_segm_slice
        return target_segm_slice
    

class Transfer3D:
    """Class to handle full-volume transfer of segmentations.
    
    It uses a TransferSlice object to handle each set of slices.
    In this way, it follows Lukas Zbinden's approach.

    There will be two modes of operation:
    1. Base mode: loops through all slices in the mpmri map sequence (target),
         finds the closest slice in the native sequence (source), and transfers the segmentation.
         Note that this assumes there are fewer slices in the target sequence than the native sequence.
         It is unclear how to handle the case where there are more slices in the target sequence.
         Using a more advanced transfer method, such as resample_from_to, may be better in that case,
         since it can do interpolation.
    2. Slice-selection mode: Loop through all slices in the target sequence, and for each slice,
           find the closest slice in the source sequence. Select a number of slices around that slice (e.g., 3),
           and evaluate 

    NOTE: Lukas uses a method from itk: sitk.ImageRegistrationMethod() which is literally built to align images from
    different sources and modalities. (including in 3D). He uses this in the slice subselection logic, seemingly
    only to get a max and min intensity from it? That doesn't make sense at all, but this method would probably
    work well to align the scans. Note that there can be deformation as well, which may allow us to deal with breathing
    and other changes.

    
    Attributes:
        source_img: RadData: The source image from which to transfer the segmentation.
        target_img: RadData: The target image to which to transfer the segmentation.
        source_seg: RadData: The segmentation in the source image space.
        
    Notes:
        - Lukas implemented a number of additional checks etc. The ones that cannot be disabled are
          listed here:
        - minimum roi size: exclude regions smaller than this size. HOWEVER, his implementation is
          slightly more naive: it excludes entire slices is the number of RoI pixels on it is less than
          the threshold, typically set to 50, 75 or 100 pixels. Best results were achieved for 50
        - remove voxel outliers: Detect voxels that are outliers in terms of intensity and remove them
          The best method was using the median-absolute-deviation from the median, with a threshold of 3.
          TODO: Visualise the MAD values (I suspect only high-intensity outliers are removed)
          NOTE: This is only done for extracting the map values, averaged over e.g. the entire RoI. So
             it's not relevant for mapping the segmentation, only for extracing the values
        - optionally: entropy calculation. It calculates the local entropy for every pixel, supposedly
          using a window, and calculates the low-entropy fraction in the RoI.
            There's two relevant params:
            - tau: any local entropy value below this is considered low-entropy
            - threshold: if the fraction of low-entropy pixels in the RoI is above this, the slice is rejected
          Note that this is calculated per RoI, so it should be done per label. We could also use connected components
          but that is outside of our scope.
        --> It seems these checks are more related to extracting the map values, not to transferring the segmentation.
            We could of course reject certain slices based on their minimum roi size, but that does not seem like a logical thing
            to do, users should be able to decide that for themselves. 

    TODO: Add label mapping option
    """

    def __init__(self, source_img: RadData, source_seg: RadData, target_img: RadData, mode: str = "base"):
        self.source_img = source_img
        self.target_img = target_img
        self.source_seg = source_seg
        
        self.transfer_slice = TransferSlice(source_img, source_seg, target_img)

        if mode not in ["base", "slice-selection"]:
            raise ValueError(f"Mode must be 'base' or 'slice-selection', but is {mode}")
        self.mode = mode

    def transfer(self, return_image: bool = False) -> RadData:
        if self.mode == "base":
            return self.transfer_base(return_image=return_image)
        elif self.mode == "slice-selection":
            raise NotImplementedError("Slice-selection mode is not yet implemented")
        else:
            raise ValueError(f"Mode must be 'base' or 'slice-selection', but is {self.mode}")

    def transfer_base(self, return_image: bool = False) -> RadData:

        """Transfer the segmentation from source to target image for all slices.

        Args:
            return_image (bool): Whether to return the target image along with the segmentation.
        """

        target_segmentations = []

        for target_slice_index in range(self.target_img.shape[2]):
            # Find the closest slice in the source image
            source_slice_index = self.transfer_slice.get_source_slice_index(target_slice_index)
            # print(f"Transferring slice {target_slice_index} using source slice {source_slice_index}")

            if return_image:
                target_slice, target_segm_slice = self.transfer_slice.transfer(source_slice_index, return_image=True)
            else:
                target_segm_slice = self.transfer_slice.transfer(source_slice_index, return_image=False)
            target_segmentations.append(target_segm_slice)

        target_segm_volume = np.concat(target_segmentations, axis=2)
        # we need to manually set the affine and header, since we have not implemented that for stacking of RadData
        target_segm_volume = RadData(target_segm_volume, affine=self.target_img.affine, header=self.target_img.header)

        if return_image:
            return self.target_img, target_segm_volume
        return target_segm_volume
        
