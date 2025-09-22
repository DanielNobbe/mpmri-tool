import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from ..data import RadData, get_affine_transform


class Transfer3D:
    def __init__(self, native_img: RadData, native_segm: RadData, target_img: RadData, mode='base', order: int = 0):
        self.native_img = native_img
        self.native_segm = native_segm
        self.target_img = target_img
        self.mode = mode
        self.order = order
        assert mode in ['base'], "Mode must be 'base'"

    def transfer(self, return_image=False, debug=False):
        native_img = self.native_img.to_nib()
        native_segm = self.native_segm.to_nib()
        target_img = self.target_img.to_nib()

        # use nibabel.processing.resample_from_to

        if self.mode == 'base':
            resampled_segm = resample_from_to(native_segm, target_img, order=self.order)  # Nearest neighbor for segmentation
            if return_image:
                resampled_img = resample_from_to(native_img, target_img, order=3)  # Higher order for image
        
            resampled_segm_data = RadData(resampled_segm.get_fdata(), affine=resampled_segm.affine, header=resampled_segm.header)
            if return_image:
                resampled_img_data = RadData(resampled_img.get_fdata(), affine=resampled_img.affine, header=resampled_img.header)
                return resampled_img_data, resampled_segm_data
            return resampled_segm_data
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented.")
