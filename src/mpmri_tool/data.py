import os

import numpy as np
import nibabel as nib
from typing import Self

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

    def __new__(cls, input_data: nib.nifti1.Nifti1Image | Self | np.ndarray | str | os.PathLike, slice_dim: int = 2, affine=None, header=None) -> Self:
        match input_data:
            case cls():
                return input_data
            case nib.nifti1.Nifti1Image():
                # obj = np.asarray(input_data.get_fdata()).view(cls)
                obj = np.asanyarray(input_data.dataobj).view(cls)  # TODO: Should we load segm like this but images like float?
                obj.affine = input_data.affine
                obj.header = input_data.header
                obj.slice_dim = slice_dim
                return obj
            case np.ndarray():
                if affine is None or header is None:
                    raise ValueError("When input_data is a numpy array, affine and header must be provided")
                obj = input_data.view(cls)
                obj.affine = affine
                obj.header = header
                obj.slice_dim = slice_dim
                return obj
            case str() | os.PathLike():
                nifti_img = nib.load(input_data)
                obj = cls(nifti_img)
                return obj
            case _:
                raise TypeError(f"Input must be a Nifti1Image or RadData instance, but is {type(input_data)}")

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.affine = getattr(obj, 'affine', None)
        self.header = getattr(obj, 'header', None)
        self.slice_dim = getattr(obj, 'slice_dim', None)

    def __repr__(self):
        return super().__repr__() + f"\nAffine:\n{self.affine}\nHeader:\n{self.header}"

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

            # Update the affine matrix for 3D slices
            if self.ndim == 3 and isinstance(key, tuple) and len(key) == 3:
                slice_indices = self._get_slice_indices(key)
                new_affine = self.affine.copy()
                # new_affine[:3, 3] += np.array(slice_indices) * self.header.get_zooms()[:3]
                # result.affine = new_affine
                # the affine matrix needs to be updated.
                # a slice is basically an additional translation, where we need to do:
                # A' = M' + t'
                # where M' = M (the rotation/scaling part) and t' = t + M * s  (s is the slice offset)
                t = self.affine[:3, 3].copy()
                t_prime = t + np.dot(self.affine[:3, :3], np.array(slice_indices))
                new_affine = self.affine.copy()
                new_affine[:3, 3] = t_prime
                result.affine = new_affine
            else:
                result.affine = self.affine
            result.header = self.header
        return result

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



def get_affine_transform(source_img: RadData, target_img: RadData) -> np.ndarray:
    # TODO: Make this part of the RadData class
    # Get the affine matrices
    source_affine = source_img.affine
    target_affine = target_img.affine
    
    # Compute the transformation matrix from source to target
    transform_matrix = np.linalg.inv(target_affine).dot(source_affine)
    
    return transform_matrix