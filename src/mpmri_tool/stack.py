from mpmri_tool.data import RadData

import nibabel as nib
from nibabel.processing import resample_from_to
import itk

from warnings import warn
import warnings
warnings.formatwarning = warnings._formatwarning_orig
import os

import numpy as np
import tempfile
import gzip
import shutil


class StackMisalignedError(Exception):
    """Custom exception for misaligned stacks."""
    pass

class Stacker:
    """Class to stack multiple 3D images into a 4D image.
    
    Useful when multiple co-registered 3D images (e.g., different MRI sequences) need to be stacked into a single object.
    Note that NIFTI files generally have poor support for 4D images,
    although our RadData class can handle them naturally.
    We recommend saving the files in NRRD format; when using the
    '.seq.nrrd' extension, applications like 3DSlicer will
    automatically recognise them as a 4D sequence.
    """

    def __init__(self):
        pass

    @staticmethod
    def _to_tuple(arr: np.ndarray) -> tuple:
        """Convert a numpy array to a tuple.

        Args:
            arr (np.ndarray): The numpy array to convert.

        Returns:
            tuple: The converted tuple.
        """

        return tuple(arr.tolist())

    def _get_corners(self, item: RadData) -> dict[str, list[float]]:
        """Get the physical coordinates of the corners of a RadData item.

        Args:
            item (RadData): The RadData object for which to get the corner coordinates.

        Returns:
            dict[str, list[float]]: A dictionary mapping corner identifiers to their physical coordinates.
        """

        corners = {}
        for i in [0, item.shape[0] - 1]:
            for j in [0, item.shape[1] - 1]:
                for k in [0, item.shape[2] - 1]:
                    corners[(i,j,k)] = self._to_tuple(nib.affines.apply_affine(item.affine, [i, j, k]))
        return corners
    
    def _check_corners_match(self, corners_list: list[dict[str, list[float]]], tol: float = 1e-3, sort: bool = False) -> bool:
        """Check if all corners in the list are the same within a tolerance.

        Args:
            corners_list (list[dict[str, list[float]]]): List of corner dictionaries to compare.
            tol (float): Tolerance for comparison, by default 1e-3.
            sort (bool): If True, will sort corners by their world
                coordinates. This is useful if the ordering of
                corners differs between items, and allows
                validating that they occupy the same space even if
                orientations differ.

        Returns:
            bool: True if all corners are the same within the
            tolerance, False otherwise.
        """

        if sort:
            corners_list = [dict(sorted(corners.items(), key=lambda item: item[1])) for corners in corners_list] # automatically breaks ties by next dimensions

        reference_corners = corners_list[0]

        # if sort:
        #     # print all corners
        #     print("Reference corners (sorted):")
        #     for key, coord in reference_corners.items():
        #         print(f"  Corner {key}: {coord}")
            
        #     for i, corners in enumerate(corners_list[1:]):
        #         print(f"Item {i+1} corners (sorted):")
        #         for key, coord in corners.items():
        #             print(f"  Corner {key}: {coord}")

        for idx, corners in enumerate(corners_list[1:]):
            for ref_coord, curr_coord in zip(reference_corners.values(), corners.values()):
                # ref_coord = reference_corners[key]
                # curr_coord = corners[key]
                if any(abs(r - c) > tol for r, c in zip(ref_coord, curr_coord)):
                    # print(f"Corners do not match for item {idx+1}: reference {ref_coord}, current {curr_coord}")  # debug print
                    
                    return False
        return True
    
    def _check_shapes_match(self, items: list[RadData]) -> bool:
        """Check if all RadData items have the same shape.

        Args:
            items (list[RadData]): List of RadData objects to compare.

        Returns:
            bool: True if all items have the same shape, False otherwise.
        """

        reference_shape = items[0].shape

        for item in items[1:]:
            if item.shape != reference_shape:
                return False
        return True
    
    def _check_all_float(self, items: list[RadData]) -> bool:
        """Check if all RadData items have float data type.

        Args:
        items (list[RadData]): List of RadData objects to check.

        Returns:
            bool: True if all items have float data type, False otherwise.
        """

        for item in items:
            if not np.issubdtype(item.dtype, np.floating):
                return False
        return True
    
    def _check_ext_nrrd(self, filename: str):
        """Check if the filename has a NRRD extension, and print warnings if it doesn't.

        Args:
            filename (str): The filename to check.

        Returns:
            bool: True if the filename ends with '.nrrd' or '.seq.nrrd', False otherwise.
        """

        if filename.endswith('.gz'):
            # disregard .gz for extension check
            filename = filename[:-3]

        if filename.endswith('.seq.nrrd'):
           return
        elif filename.endswith('.nrrd'):
            warn("We recommend saving stacked 4D images with the '.seq.nrrd' extension, so that applications like 3DSlicer automatically recognise them as sequences.")
            return
        else:
            ext = os.path.splitext(filename)[1]
            warn(f"We recommend using the NRRD format to save stacked 4D images. The provided filename has extension '{ext}', which may not be supported properly by some applications, or may not support 4D volumes well.")

        return filename.endswith('.nrrd') or filename.endswith('.seq.nrrd')
    
    def _stack_itk(self, items: list[RadData], filename: str):
        """Stack multiple RadData 3D volumes into a single 4D NRRD file using ITK.
        
        Args:
            items (list[RadData]): List of RadData objects, each representing a 3D image.
            filename (str): The output filename for the stacked 4D image.
        """


        # convert each 3D subvolume to ITK image
        imgs = [item.to_itk() for item in items]

        # stack along 4th dimension using JoinSeries
        join_type = itk.ComposeImageFilter[itk.Image[itk.D,3], itk.VectorImage[itk.D,3]]
        join = join_type.New()
        for i, img in enumerate(imgs):
            join.SetInput(i, img)
        img4d = join.GetOutput()

        # save the 4D image
        itk.imwrite(img4d, filename)

    def stack_to_file(self, items: list[RadData], filename: str, corners_tol: float = 1e-3, ignore_slice_mismatch: bool = False):
        """Stack multiple RadData 3D items into a single 4D NRRD file.

        Note that stacking to a 4D RadData object is almost trivial,
        while we need this custom class to save it to an NRRD file
        properly, which is not trivial. 
        Note that it would be possible to save a 4D RadData object to
        NRRD too using similar methods. This would use a single affine
        matrix, which is OK if everything is resampled to the same
        space.

        Args: items (list[RadData]): List of RadData objects, each representing a 3D image.

        """

        if not self._check_all_float(items):
            raise TypeError("Not all input RadData items have float data type. We use ITK for stacking to a file, and it has static typing, so we require float64 data. Use RadData(..., as_float=True) when loading the items to ensure they have float data type.")

        if not self._check_shapes_match(items):
            raise ValueError("Input RadData items do not have the same shape. If shapes are close, stacking could work well with resampling. Consider resampling items before stacking, or implementing a more flexible shape check in the Stacker class.")  # e.g., allow small differences in shape and resample accordingly

        corners_list = [self._get_corners(item) for item in items]

        corners_and_orientation_match = self._check_corners_match(corners_list, sort=False, tol=corners_tol)

        if not corners_and_orientation_match:
            corners_match = self._check_corners_match(corners_list, sort=True, tol=corners_tol)
            if not corners_match:

                raise StackMisalignedError("Input RadData items do not occupy the same physical space.")
            else:

                # resample all items to the first one

                if ignore_slice_mismatch:
                    # set the affine offset in z-direction to 0, so that resampling ignores slice position differences
                    new_items = []
                    for item in items:
                        item_affine = np.eye(4)
                        item_affine[:2, :2] = item.affine[:2, :2]
                        item_affine[:2, 3] = item.affine[:2, 3]
                        new_item = item.copy()
                        new_item.affine = item_affine
                        new_items.append(new_item)
                    items = new_items
                
                reference_item = items[0]
                resampled_items = [reference_item]

                for item in items[1:]:
                    resampled_nib = resample_from_to(item.to_nib(), reference_item.to_nib())
                    resampled_items.append(RadData(resampled_nib, as_float=True))
                items = resampled_items

                warn("Input RadData items have different orientations. They have been resampled to match the first item's orientation before stacking. This may not always work well if applying it to segmentation masks or other discrete images, since it uses spline interpolation. Typically, this operation only does a rotation.", UserWarning)

        self._check_ext_nrrd(filename)

        if os.path.splitext(filename)[1] == '.gz':
            warn("Saving stacked NRRD file with gzip compression. 3DSlicer and other tools may not be natively compatible with nrrd.gz files.", UserWarning)
            # save temp file, then gzip it
            with tempfile.NamedTemporaryFile(suffix='.nrrd', delete=False) as tmpfile:
                self._stack_itk(items, tmpfile.name)
                with open(tmpfile.name, 'rb', compressionlevel=5) as f_in:
                    with gzip.open(filename, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        else:

            self._stack_itk(items, filename)
        
