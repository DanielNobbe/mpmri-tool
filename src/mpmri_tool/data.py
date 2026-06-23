import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Self, Literal
from warnings import warn

import numpy as np
import nibabel as nib
from nibabel import processing
from nibabel.funcs import as_closest_canonical
import itk
from scipy.ndimage import label as connected_components_label


RadDataKind = Literal['image', 'segmentation']

### TODO: REWRITE BASED ON TORCHIO (they implement the affine updates etc much better than I do)

class RadData(np.ndarray):
    """A subclass of np.ndarray to represent radiology data.

    This class is built to load from a NiFTI object (through NIBabel).

    Attributes:
        self: np.ndarray: The underlying numpy array.
        slice_dim: int: The dimension along which slices are taken (default is 2, i.e., axial slices).
            # TODO: Implement a more robust solution using RAS orientation code
        affine: np.ndarray: The affine transformation matrix from the NIfTI header.
        header: nib.Nifti1Header: The NIfTI header containing metadata.
    """

    def __new__(
            cls,
            input_data: nib.nifti1.Nifti1Image | Self | np.ndarray | str | os.PathLike | bytes,
            slice_dim: int | str = 2,
            affine=None,
            header=None,
            as_float: bool = False,
            kind: RadDataKind = "image",
            resample_to: Self | None = None,
            tempdir: str = '/tmp'
        ) -> Self:

        if kind not in ['image', 'segmentation']:
            raise ValueError(f"Invalid kind: {kind}. Must be 'image' or 'segmentation'.")
        
        if kind == 'segmentation' and as_float:
            warn("as_float=True is not typical for segmentation data. Ensure this is intended.", UserWarning)
        elif kind == 'image':
            as_float = True  # force as_float for image data

        match input_data:
            case cls():
                return input_data
            case nib.nifti1.Nifti1Image():

                if resample_to is not None:
                    input_data = processing.resample_from_to(input_data, (resample_to.shape, resample_to.affine), order=3 if as_float else 0)  # linear interpolation for images, nearest neighbor for segmentations

                if as_float:
                    obj = np.asarray(input_data.get_fdata()).view(cls)
                else:
                # obj = np.asarray(input_data.get_fdata()).view(cls)
                    obj = np.asanyarray(input_data.dataobj, dtype=int).view(cls)  # TODO: Should we load segm like this but images like float?
                obj.affine = input_data.affine
                obj.header = input_data.header
                obj.slice_dim = obj._get_slice_dim_from_name(slice_dim) if isinstance(slice_dim, str) else slice_dim
                obj.kind = kind
                return obj
            case np.ndarray():
                if affine is None or header is None:
                    raise ValueError("When input_data is a numpy array, affine and header must be provided")
                
                if resample_to is not None:
                    as_nifti = nib.Nifti1Image(input_data, affine=affine, header=header)
                    as_nifti = processing.resample_from_to(as_nifti, (resample_to.shape, resample_to.affine), order=3 if as_float else 0)
                    return cls(as_nifti, slice_dim=slice_dim, as_float=as_float, kind=kind)

                obj = input_data.view(cls)
                obj.affine = affine
                obj.header = header
                obj.slice_dim = obj._get_slice_dim_from_name(slice_dim) if isinstance(slice_dim, str) else slice_dim
                obj.kind = kind
                return obj
            case str() | os.PathLike():
                nifti_img = nib.load(input_data)
                nifti_img = as_closest_canonical(nifti_img)
                # print(f"Loaded NIfTI image from {input_data} with shape {nifti_img.shape} and affine:\n{nifti_img.affine}")
                # ors = nib.orientations.aff2axcodes(nifti_img.affine)
                # print(f"Image orientation (axcodes): {ors}")

                if resample_to is not None:
                    nifti_img = processing.resample_from_to(nifti_img, (resample_to.shape, resample_to.affine), order=3 if as_float else 0)  # linear interpolation for images, nearest neighbor for segmentations

                obj = cls(nifti_img, slice_dim=slice_dim, as_float=as_float, kind=kind)
                return obj
            case bytes():
                with NamedTemporaryFile(suffix='.nii.gz', dir=tempdir, delete=True) as tmp_file:
                    # delete=True is essential to prevent blowup of shm
                    # also, we can use os.unlink to ensure the file is deleted immediately after loading
                    tmp_file.write(input_data)
                    tmp_file.flush()
                    obj = cls(tmp_file.name, slice_dim=slice_dim, as_float=as_float, kind=kind, resample_to=resample_to)
                    os.unlink(tmp_file.name)  # this ensures the file is deleted on a crash
                return obj
            case _:
                raise TypeError(f"Input must be a Nifti1Image or RadData instance, but is {type(input_data)}")

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.affine = getattr(obj, 'affine', None)
        self.header = getattr(obj, 'header', None)
        self.slice_dim = getattr(obj, 'slice_dim', None)
        self.kind = getattr(obj, 'kind', 'image')

    def __repr__(self):
        if len(self.shape) == 0:
            # don't print affine etc for 1D outputs (e.g. max())
            return f"RadData object of kind {self.kind}: {super().__repr__()}"
        return f"RadData object of kind {self.kind}: {super().__repr__()}\nAffine:\n{self.affine}\nHeader:\n{self.header}"

    def _get_slice_indices(self, key):
        if isinstance(key, tuple):
            indices = [k.start if isinstance(k, slice) else k for k in key]
        elif isinstance(key, slice):
            indices = [key.start]
        else:
            indices = [key]
        return [(index if index is not None else 0) for index in indices]

    # handle slicing for the affine matric
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, RadData):

            # ensure it is 3 dimensional if it is not a single value
            # take into account the key, if a key is a slice, that should keep the dimension
            # expand empty dimensions
            if result.ndim < 3:
                keys_are_slices = [isinstance(k, slice) for k in key] if isinstance(key, tuple) else [isinstance(key, slice)]
                new_shape = list(result.shape)
                for index, is_slice in enumerate(keys_are_slices):
                    if not is_slice:
                        new_shape.insert(index, 1)

                result = result.reshape(new_shape)

                warn("Slicing RadData to less than 3D; this may not correctly conserve affine matrix.", UserWarning)

            # Update the affine matrix for 3D slices
            if self.ndim >= 3 and isinstance(key, tuple) and len(key) >= 3:
                slice_indices = self._get_slice_indices(key)
                new_affine = self.affine.copy()
                # new_affine[:3, 3] += np.array(slice_indices) * self.header.get_zooms()[:3]
                # result.affine = new_affine
                # the affine matrix needs to be updated.
                # a slice is basically an additional translation, where we need to do:
                # A' = M' + t'
                # where M' = M (the rotation/scaling part) and t' = t + M * s  (s is the slice offset)
                t = self.affine[:3, 3].copy()
                t_prime = t + np.dot(self.affine[:3, :3], np.array(slice_indices)[:3])
                new_affine = self.affine.copy()
                new_affine[:3, 3] = t_prime
                result.affine = new_affine
            else:
                warn("Slicing RadData did not result in a 3D volume; affine matrix not updated.", UserWarning)
                result.affine = self.affine
            result.header = self.header
        return result

    @property
    def num_slices(self) -> int:
        """Return the number of slices along the slice dimension."""
        if self.slice_dim is None:
            raise ValueError("Data does not have a defined slice dimension.")
        return self.shape[self.slice_dim]

    # TODO: Add some affine handling when creating a fully new array and inserting into it

    def save(self, filepath: str | os.PathLike, set_nan_to_zero: bool = False):
        """Save the RadData to a NIfTI file.

        Args:
            filepath (str | os.PathLike): The path to save the NIfTI file.
        """
        if self.affine is None or self.header is None:
            raise ValueError("Cannot save RadData without affine and header information.")
        
        if set_nan_to_zero:
            data_to_save = self.copy()
            data_to_save[np.isnan(data_to_save)] = 0
            nifti_img = nib.Nifti1Image(data_to_save.view(np.ndarray), affine=self.affine, header=self.header)
        else:
            nifti_img = nib.Nifti1Image(self.view(np.ndarray), affine=self.affine, header=self.header)
        nib.save(nifti_img, filepath)

    def to_nib(self) -> nib.nifti1.Nifti1Image:
        """Convert the RadData to a NIfTI image.

        Returns:
            nib.nifti1.Nifti1Image: The corresponding NIfTI image.
        """
        if self.affine is None or self.header is None:
            raise ValueError("Cannot convert RadData to NIfTI without affine and header information.")
        return nib.Nifti1Image(self.view(np.ndarray), affine=self.affine, header=self.header)
    
    def _get_slice_dim_from_name(self, name: str) -> int:
        """Get the slice dim from the plane name (e.g. 'axial', 'sagittal', 'coronal')."""
        name = name.lower()
        match name:
            case 'axial' | 'transverse':
                return 2
            case 'sagittal':
                return 0
            case 'coronal':
                return 1
            case _:
                raise ValueError(f"Unknown plane name: {name}. Expected 'axial', 'sagittal', or 'coronal'.")

    def crop_to_foreground(self) -> 'RadData':
        """Crop the RadData to the foreground (non-zero values)."""
        foreground_mask = self != 0
        if not np.any(foreground_mask):
            warn("No foreground found in RadData; returning original data.")
            return self
        coords = np.array(np.nonzero(foreground_mask))
        min_coords = coords.min(axis=1)
        max_coords = coords.max(axis=1) + 1  # add 1 to include the max index
        slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
        return self[slices]

    @staticmethod
    def _orient_slice(slice_data: np.ndarray, flip: list[int], slice_dim: int) -> np.ndarray:
        """Orient the slice data according to the specified orientation, for the purpose of creating an image.

        Args:
            slice_data (np.ndarray): The slice data to orient.
            flip (list[int]): A list of dimensions along which to flip the data.
            slice_dim (int): The dimension along which the slice is taken.

        Returns:
            np.ndarray: The oriented slice data.
        """
        # Apply flips if specified
        for dim in flip:
            slice_data = np.flip(slice_data, axis=dim)
        # always transpose first and second dims to align with how screens work, where vertical is the first dim
        slice_data = slice_data.transpose(1, 0)
        return slice_data

    def extract_slice(self, slice_index: int, slice_dim: int | None = None, flip_dims: list[int] = None, pad: int | None = None) -> 'RadData':
        """Extract a single slice along the specified dimension or self.slice_dim."""
        if slice_dim is None:
            slice_dim = self.slice_dim
        if slice_dim is None:
            raise ValueError("Data does not have a defined slice dimension.")
        
        # Create a slicing key that selects all in other dimensions and the specified index in the slice dimension
        key = [slice(None)] * self.ndim
        key[slice_dim] = slice(slice_index, slice_index + 1)  # this conserves dimension and correctly updates affine
        
        slice_arr = self[tuple(key)].squeeze() # remove the slice dimension

        # if pad is defined, we add a padding of zeros around the slice
        if pad is not None:
            slice_arr = np.pad(slice_arr, pad_width=pad, mode='constant', constant_values=0)

        return self._orient_slice(slice_arr, flip_dims, slice_dim)
    
    def as_oriented_array(self, orientation: str = 'ras') -> np.ndarray:
        """Return the data as a numpy array oriented according to the specified orientation, for the purpose of creating an image.

        Args:
            orientation (str): The orientation to use. Options are 'ras' (default), 'radiological' (i.e. looking from bottom of the feet, assuming slice axis is z axis) or 'neurological' (i.e. looking from top of the head, assuming slice axis is z axis). In the case of 'ras', the data is returned as-is, in the other cases it's transposed for orientation purposes.
        Returns:
            np.ndarray: The oriented data array.
        
        How it works:
        see: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm#:~:text=Radiological%22%20vs%20%22Neurological%22%20Orientation%20in%20Viewers,-%22Radiological
         - ras: return as is
         - radiological: flip left-right (right-left from left-right), flip front-back (anterior-posterior from posterior-anterior), keep up-down (stays inferior-superior)
         - neurological: keep left-right (stays left-right), flip front-back (anterior-posterior from posterior-anterior), 
        """

        
    def get_orientation_config(self, orientation: str, slice_dim: int) -> dict:
        """Get the orientation configuration for the specified orientation.
        
        We largely follow http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
        for the definition of the orientations of the slices. For the slice ordering, there seem to be 
        no standards, but we have tried to make it anatomically realistic.

        Short summary:
        Radiolocial orientation: left of patient on right side, anterior of patient on top or right,
        always "looking from below"
        - axial slices: (slice dim == 2)
            ordering: iterate from feet to head (no inversion from RAS+)
            flips: flip L/R (dim 0 of slice), flip A/P (dim 1 of slice)
        - sagittal slices: we choose the anterior of patient on the right (arbitrary) (slice dim == 0)
            ordering: iterate from right to left (follow graham), so invert from RAS+
            flips: no flip for A/P (dim 0 of slice), flip I/S (dim 1 of slice)
        - coronal slices: right on left, iterate front to back (slice dim == 1)
            ordering: anterior to posterior, so INVERT from RAS+
            flips: flip L/R (dim 0 of slice), flip I/S (dim 1 of slice)


        Neurological orientation: left of patient on left side, anterior of patient on top or right,
        always "looking from above"
        - axial slices: (slice dim == 2)
            ordering: iterate from head to feet (invert from RAS+)
            flips: no flip L/R (dim 0 of slice), flip A/P (dim 1 of slice)
        - sagittal slices: we choose the anterior of patient on the right (arbitrary) (slice dim == 0)
            ordering: iterate from right to left (follow graham), so invert from RAS+
            flips: no flip for A/P (dim 0 of slice), flip I/S (dim 1 of slice)
        - coronal slices: left on left, up on top, look from back (ordering also kind of arbitrary) (slice dim == 1)
            ordering: posterior to anterior, so follow RAS+
            flips: no flip L/R (dim 0 of slice), flip I/S (dim 1 of slice)

        We will output a dict with keys "invert ordering" (bool), "flip_dims" (list of dimensions to flip, where the dimensions are relative to the slice, so 0 is the first dim of the slice, 1 is the second dim of the slice) and "orientation_description" (a string description of the orientation for logging/debugging purposes).
        """

        match orientation, slice_dim:
            case 'ras', _:
                return {
                    "invert_ordering": False,
                    "flip_dims": [],
                    "orientation_description": "RAS orientation: no flips, no inversion, standard radiological orientation for axial slices, but not for sagittal and coronal slices."
                }
            case 'radiological', 2: # axial
                return {
                    "invert_ordering": False,
                    "flip_dims": [0, 1],
                    "orientation_description": "Radiological orientation for axial slices: left of patient on right side, anterior of patient on top or right, always looking from below. Flips: flip L/R (dim 0 of slice), flip A/P (dim 1 of slice). No inversion from RAS+."
                }
            case 'radiological', 0: # sagittal
                return {
                    "invert_ordering": True,
                    "flip_dims": [1],
                    "orientation_description": "Radiological orientation for sagittal slices: we choose the anterior of patient on the right (arbitrary). Ordering: iterate from right to left, so invert from RAS+. Flips: no flip for A/P (dim 0 of slice), flip I/S (dim 1 of slice)."
                }
            case 'radiological', 1: # coronal
                return {
                    "invert_ordering": True,
                    "flip_dims": [0, 1],
                    "orientation_description": "Radiological orientation for coronal slices: right on left, iterate front to back. Ordering: anterior to posterior, so INVERT from RAS+. Flips: flip L/R (dim 0 of slice), flip I/S (dim 1 of slice)."
                }
            case 'neurological', 2: # axial
                return {
                    "invert_ordering": True,
                    "flip_dims": [1],
                    "orientation_description": "Neurological orientation for axial slices: left of patient on left side, anterior of patient on top or right, always looking from above. Flips: no flip L/R (dim 0 of slice), flip A/P (dim 1 of slice). Ordering: iterate from head to feet, so invert from RAS+."
                }
            case 'neurological', 0: # sagittal
                return {
                    "invert_ordering": True,
                    "flip_dims": [1],
                    "orientation_description": "Neurological orientation for sagittal slices: we choose the anterior of patient on the right (arbitrary). Ordering: iterate from right to left, so invert from RAS+. Flips: no flip for A/P (dim 0 of slice), flip I/S (dim 1 of slice)."
                }
            case 'neurological', 1: # coronal
                return {
                    "invert_ordering": False,
                    "flip_dims": [1],
                    "orientation_description": "Neurological orientation for coronal slices: left on left, up on top, look from back (ordering also kind of arbitrary). Ordering: posterior to anterior, so follow RAS+. Flips: no flip L/R (dim 0 of slice), flip I/S (dim 1 of slice)."
                }
            case _:
                raise ValueError(f"Unknown orientation {orientation} or invalid slice_dim {slice_dim} for orientation.")  
    
    def extract_slices(self, slice_dim: int | str | None = None, slice_interval: int | None = None, num_slices: int | None = None, orientation: str = 'ras', pad: int | None = None, skip_outer_slices: bool = False) -> list['RadData']:
        """Extract all slices along the specified dimension or self.slice_dim.

        Args:
            orientation (str): The output orientation of the slices. Options are 'ras' (default),
                'radiological' (i.e. looking from bottom of the feet, assuming slice axis is z axis) or
                'neurological' (i.e. looking from top of the head, assuming slice axis is z axis).
                Note that currently, the orientation only affects the slices themselves, not their
                ordering.
        
        If slice_interval is provided, extracts every slice_interval-th slice. If num_slices is provided, extracts that many slices evenly spaced along the dimension. If neither is provided, extracts all slices. 
        In RAS orientation, the slices are ordered from inferior to superior.

        """

        if slice_interval is not None and num_slices is not None:
            raise ValueError("Cannot specify both slice_interval and num_slices. Please choose one.")

        if isinstance(slice_dim, str):
            slice_dim = self._get_slice_dim_from_name(slice_dim)
        elif slice_dim is None:
            if self.slice_dim is None:
                raise ValueError("Data does not have a defined slice dimension. Please specify slice_dim.")
            slice_dim = self.slice_dim

        orientation_config = self.get_orientation_config(orientation, slice_dim)

        # if num_slices is not None and skip_outer_slices:
        #     num_slices += 2  # we will later remove the first and last slice, so we need to extract 2 extra slices to account for this

        if slice_interval is not None:
            step_size = slice_interval
        elif num_slices is not None:
            step_size = max(1, self.shape[slice_dim] // num_slices)
        else:
            step_size = 1

        if orientation_config["invert_ordering"]:
            slice_start = self.shape[slice_dim] - 1 if not skip_outer_slices else self.shape[slice_dim] - 1 - step_size
            slice_end = -1 if not skip_outer_slices else -1 + step_size
            slice_range = range(slice_start, slice_end, -1 * step_size)
        else:
            slice_start = 0 if not skip_outer_slices else step_size
            slice_end = self.shape[slice_dim] if not skip_outer_slices else self.shape[slice_dim] - step_size
            slice_range = range(slice_start, slice_end, step_size)
        
        slices = [self.extract_slice(i, slice_dim, orientation_config["flip_dims"]) for i in slice_range]  # TODO: Implement multiprocessing?

        if len(slices) == 0:
            warn("No slices were extracted. Please check your slice_interval and num_slices parameters.")
        if num_slices is not None and len(slices) > num_slices:
            warn(f"Extracted more slices than requested num_slices={num_slices}. Returning the first {num_slices} slices (out of {len(slices)}).")
            slices = slices[:num_slices]

        
        return slices
    
    def to_2d(self) -> 'RadData':
        """Convert the RadData to a 2D array, similar to squeezing.
        
        Only allowed if the RadData has exactly one slice along the slice_dim.
        """
        if self.slice_dim is None:
            raise ValueError("Data does not have a defined slice dimension.")
        if self.shape[self.slice_dim] != 1:
            raise ValueError(f"Cannot convert to 2D; slice dimension {self.slice_dim} has size {self.shape[self.slice_dim]}, expected 1.")
        
        new_shape = list(self.shape)
        new_shape.pop(self.slice_dim)
        new_data = self.reshape(new_shape)
        
        # Update affine: remove the slice dimension
        new_affine = self.affine.copy()
        # This is a simplification; in reality, removing a dimension from the affine is more complex
        # Here we just set the translation part for the removed dimension to zero
        new_affine = np.delete(new_affine, self.slice_dim, axis=0)
        new_affine = np.delete(new_affine, self.slice_dim, axis=1)
        
        return RadData(new_data, slice_dim=None, affine=new_affine, header=self.header)
    
    # def stack(self, other_items: list['RadData'], dim: int = 3) -> 'RadData':
    #     """Stack this RadData with other RadData items along the given dimension.
        
    #     Args:
    #         other_items (list[RadData]): List of RadData items to stack with this one.
    #         dim (int): Dimension along which to stack. Defaults to 4th dimension (the default channel dim).
        
    #     Returns:
    #         RadData: Stacked RadData item.
    #     """
    #     return stack([self] + other_items, dim=dim)
    
    def normalize(self, method: str = 'mean-std', rescale: float | int | None = None) -> 'RadData':
        """Normalize the RadData using the specified method.

        Note: Does not operate in-place; returns a new RadData item.
        
        Args:
            method (str): Normalization method. Options are 'mean-std' or 'min-max'.
            rescale (float | int | None): Factor by which to rescale the normalized data.

        Returns:
            RadData: Normalized RadData item.
        """
        if method == 'mean-std':
            mean = np.mean(self)
            std = np.std(self)
            normalized_data = (self - mean) / std
        elif method == 'min-max':
            min_val = np.min(self)
            max_val = np.max(self)
            normalized_data = (self - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if rescale is not None:
            normalized_data *= rescale

            # also cast to type of rescale if rescale is int or float
            if isinstance(rescale, int | float):
                normalized_data = normalized_data.astype(type(rescale))

        return RadData(normalized_data, slice_dim=self.slice_dim, affine=self.affine, header=self.header)
    
    def to_itk(self) -> itk.ImageBase:
        """Convert the RadData to an ITK image.

        Returns:
            itk.ImageBase: The corresponding ITK image.
        """

        # print(f"Dtype of RadData: {self.dtype}")

        if self.affine is None or self.header is None:
            raise ValueError("Cannot convert RadData to ITK image without affine and header information.")

        array = np.ascontiguousarray(self.view(np.ndarray).transpose())

        # perm = [2, 1, 0]
        # new_affine = np.zeros_like(self.affine)
        # new_affine[:3, :3] = self.affine[np.ix_(perm, perm)]
        # new_affine[:3, 3] = self.affine[perm, 3]
        # new_affine[3, 3] = 1
        
        # Create ITK image from numpy array
        itk_image = itk.image_from_array(array, is_vector=True if self.ndim > 3 else False)  # TODO: Deal with the case of 2D vector image
        
        # Set spacing, origin, and direction based on affine
        spacing = np.linalg.norm(self.affine[:3, :3], axis=0)
        
        # Direction is the rotation/scaling part of the affine
        direction = self.affine[:3, :3] / spacing
        direction[0:2, 0:2] *= -1  # flip x and y to match ITK coordinate system
        # above is an empirical result
        itk_direction = itk.GetMatrixFromArray(direction)
        
        # Origin is translation part
        origin = self.affine[:3, 3]
        origin[0:2] *= -1  # flip x and y to match ITK coordinate system, empirically found
        
        # Set spacing, origin, and direction on ITK image
        itk_image.SetSpacing(spacing.tolist())
        itk_image.SetOrigin(origin.tolist())
        itk_image.SetDirection(itk_direction)
        
        return itk_image
    
    @staticmethod  # TODO: Make this into its own mixin
    def _get_mapping_array(sample_to_target_map):
        max_sample_value = max(sample_to_target_map.keys())
        mapping_array = np.zeros(max_sample_value + 1, dtype=np.int32)
        for sample_value, target_value in sample_to_target_map.items():
            mapping_array[sample_value] = target_value
        return mapping_array
    
    def map_labels(self, label_map: dict[int, int]) -> 'RadData':
        """Map labels in the RadData according to the provided label map.

        Note: only works for the 'segmentation' kind RadData. # TODO: Split into separate subclass
        
        Args:
            label_map (dict[int, int]): A dictionary mapping old labels to new labels.
        Returns:
            RadData: A new RadData item with labels mapped.
        """

        if self.kind != 'segmentation':
            raise ValueError("Label mapping is only supported for RadData of kind 'segmentation'.")
        
        mapping_array = self._get_mapping_array(label_map)

        mapped_data = self.copy()
        flat_data = mapped_data.flatten()
        mapped_flat_data = mapping_array[flat_data]
        mapped_data = mapped_flat_data.reshape(self.shape)

        return RadData(mapped_data, slice_dim=self.slice_dim, affine=self.affine, header=self.header, kind=self.kind)

    def filter_by_label(self, label: int | list[int]) -> 'RadData':
        """Filter the RadData to only include the specified label(s). Will map the label(s) to 1 and all other values to 0.

        Note: only works for the 'segmentation' kind RadData. # TODO: Split into separate subclass
        
        Args:
            label (int | list[int]): The label(s) to filter by.
        """
        if self.kind != 'segmentation':
            raise ValueError("Label filtering is only supported for RadData of kind 'segmentation'.")

        if isinstance(label, int):
            label = {label}

        label_map = {i.item(): (1 if i in label else 0) for i in self.unique()}  # TODO: Stop wrapping the numpy array, or fix the slicing behaviour for low-level functions
        return self.map_labels(label_map)
    
    def _split_connected_components(self) -> 'RadData':
        labeled_array, num_features = connected_components_label(self)
        return RadData(labeled_array, slice_dim=self.slice_dim, affine=self.affine, header=self.header, kind=self.kind)
    
    def split_into_objects(self, method='connected') -> 'RadData':
        """Split the RadData into separate objects based on the specified method.
        Only operated on 'segmentation' kind RadData, and requires that the segmentation is binary.
        If your segmentation is not binary, consider using filter_by_label.

        Note: only works for the 'segmentation' kind RadData. # TODO: Split into separate subclass
        
        Args:
            method (str): The method to use for splitting. Options are 'connected' (connected components) or 'watershed' (watershed transform).
            
        Returns:
            RadData: A new RadData item with separate objects labeled with unique integers.
        """

        if self.kind != 'segmentation':
            raise ValueError("Object splitting is only supported for RadData of kind 'segmentation'.")

        if method == 'connected':
            return self._split_connected_components()
        elif method == 'watershed':
            raise NotImplementedError("Watershed splitting is not yet implemented, we found out it's actually quite complex to implement it.")
        else:
            raise ValueError(f"Unknown splitting method: {method}")

    def unique(self, *args, **kwargs):
        """Return the unique values in the RadData, similar to np.unique."""
        return np.unique(self.view(np.ndarray), *args, **kwargs)




def get_affine_transform(source_img: RadData, target_img: RadData) -> np.ndarray:
    # TODO: Make this part of the RadData class
    # Get the affine matrices
    source_affine = source_img.affine
    target_affine = target_img.affine
    
    # Compute the transformation matrix from source to target
    transform_matrix = np.linalg.inv(target_affine).dot(source_affine)
    
    return transform_matrix


# def stack(items: list[RadData], dim: int = 3, normalize: bool = False) -> RadData:
#     """Stack multiple RadData items along the given dimension.
    
#     Args:
#         items (list[RadData]): List of RadData items to stack.
#         dim (int): Dimension along which to stack. Defaults to 4th dimension (the default channel dim).
    
#     Returns:
#         RadData: Stacked RadData item.
#     """

#     shapes = [item.shape for item in items]
#     if not all(shape == shapes[0] for shape in shapes):
#         raise ValueError("All items must have the same shape to be stacked. If you need to stack items with varying shapes that occupy the same space, please consider implementing resampling or upsampling.")

#     # validate that the max dimension for each item is either less than dim, or only a length of one along dim
#     for item in items:
#         if dim < item.ndim and item.shape[dim] > 1:
#             raise ValueError(f"Cannot stack along dimension {dim} for item with shape {item.shape} (ndim={item.ndim}).")

#     # expand dimensions as necessary, mostly relevant if we need to add more than one dim
#     expanded_items = [item.reshape(item.shape + (1,) * (dim - item.ndim + 1)) if item.ndim < dim else item for item in items]

#     if normalize:
#         expanded_items = [item.normalize() for item in expanded_items]

#     stacked_array = np.stack(expanded_items, axis=dim)
    
#     # Create new affine and header
#     # For simplicity, we take the affine and header from the first item
#     new_affine = items[0].affine
#     new_header = items[0].header.copy()
#     warn("RadData.stack uses the affine and header from the first item. Please ensure this is appropriate for your use case.")
    
#     # Update header to reflect new shape
#     new_shape = list(items[0].shape)
#     new_shape.insert(dim, len(items))
#     new_header.set_data_shape(tuple(new_shape))
    
#     return RadData(stacked_array, slice_dim=items[0].slice_dim, affine=new_affine, header=new_header)