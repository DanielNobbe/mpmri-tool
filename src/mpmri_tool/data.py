import os
from typing import Self
from warnings import warn

import numpy as np
import nibabel as nib
import itk


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

    def __new__(cls, input_data: nib.nifti1.Nifti1Image | Self | np.ndarray | str | os.PathLike, slice_dim: int = 2, affine=None, header=None, as_float: bool = False) -> Self:
        match input_data:
            case cls():
                return input_data
            case nib.nifti1.Nifti1Image():
                if as_float:
                    obj = np.asarray(input_data.get_fdata()).view(cls)
                else:
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
                obj = cls(nifti_img, slice_dim=slice_dim, as_float=as_float)
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
        if len(self.shape) == 0:
            # don't print affine etc for 1D outputs (e.g. max())
            return super().__repr__()
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
    
    def extract_slice(self, slice_index: int, slice_dim: int | None = None) -> 'RadData':
        """Extract a single slice along the specified dimension or self.slice_dim."""
        if slice_dim is None:
            slice_dim = self.slice_dim
        if slice_dim is None:
            raise ValueError("Data does not have a defined slice dimension.")
        
        # Create a slicing key that selects all in other dimensions and the specified index in the slice dimension
        key = [slice(None)] * self.ndim
        key[slice_dim] = slice(slice_index, slice_index + 1)  # this conserves dimension and correctly updates affine
        
        return self[tuple(key)]
    
    def extract_slices(self, slice_dim: int | None = None) -> list['RadData']:
        """Extract all slices along the specified dimension or self.slice_dim."""
        if slice_dim is None:
            slice_dim = self.slice_dim
        if slice_dim is None:
            raise ValueError("Data does not have a defined slice dimension.")
        
        # always slice from low to high z coordinate
        first_slice_coord = nib.affines.apply_affine(self.affine, [0, 0, 0])[slice_dim]
        last_slice_indices = [0] * self.ndim
        last_slice_indices[slice_dim] = self.shape[slice_dim] - 1
        last_slice_coord = nib.affines.apply_affine(self.affine, last_slice_indices)[slice_dim]

        if last_slice_coord < first_slice_coord:
            warn("Data affine indicates that slice order is descending. Slices will be extracted in descending order.")
            slice_range = range(self.shape[slice_dim] - 1, -1, -1)
        else:
            slice_range = range(self.shape[slice_dim])
        
        slices = [self.extract_slice(i, slice_dim) for i in slice_range]  # TODO: Implement multiprocessing?
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
    
    def normalize(self, method: str = 'mean-std') -> 'RadData':
        """Normalize the RadData using the specified method.

        Note: Does not operate in-place; returns a new RadData item.
        
        Args:
            method (str): Normalization method. Options are 'mean-std' or 'min-max'.
        
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