from mpmri_tool.transfer.zbinden import TransferSlice, Transfer3D
from mpmri_tool.data import RadData
from mpmri_tool.measure.zbinden import Measurement

def full_pipeline():
    native_img = RadData("/Users/daniel/data/zbinden/P3/toy-version/inphase/mr/ISP5003_1002_121842.245000_0000.nii.gz")
    native_seg = RadData("/Users/daniel/data/zbinden/P3/toy-version/inphase/segm/ISP5003_1002_121842.245000.nii.gz")

    target_img = RadData("/Users/daniel/data/zbinden/P3/toy-version/maps/ISP5003_122656.909000_t1map_longt1_mbh_moco_t1.nii.gz")

    # transfer = TransferSlice(native_img, native_seg, target_img)

    # slice_idx = 40

    # tgt_slice, tgt_segm = transfer.transfer(slice_idx, return_image=True)

    # tgt_slice.save("tgt_slice.nii.gz")
    # tgt_segm.save("tgt_segm.nii.gz")


    transfer = Transfer3D(native_img, native_seg, target_img, mode='base')
    tgt_segm = transfer.transfer(return_image=False)

    tgt_segm.save("debug/tgt_segm_3d.nii.gz")
    target_img.save("debug/tgt_img_3d.nii.gz")

    measure = Measurement(target_img, tgt_segm, 
                      labels=[1,2,3, 4, 5, 6, 7, 8, 9, 10, 11], merge_labels=True, 
                      min_roi_size=50, entropy_threshold=4.5,
                      low_entropy_fraction_threshold=0.05)
    
    results = measure.get_intensities_full(debug=True)

    values = measure.get_mean_intensities_full(return_std=True, debug=True)


    print("Mean intensities and stddevs for each label:")
    means, stddevs = values
    for label in means:
        print(f"Label {label}: Mean = {means[label]:.2f}, StdDev = {stddevs[label]:.2f}")


def main():
    native_img = RadData("/Users/daniel/data/zbinden/P3/toy-version/inphase/mr/ISP5003_1002_121842.245000_0000.nii.gz")
    native_seg = RadData("/Users/daniel/data/zbinden/P3/toy-version/inphase/segm/ISP5003_1002_121842.245000.nii.gz")

    target_img = RadData("/Users/daniel/data/zbinden/P3/toy-version/maps/ISP5003_122656.909000_t1map_longt1_mbh_moco_t1.nii.gz")

    # transfer = TransferSlice(native_img, native_seg, target_img)

    # slice_idx = 40

    # tgt_slice, tgt_segm = transfer.transfer(slice_idx, return_image=True)

    # tgt_slice.save("tgt_slice.nii.gz")
    # tgt_segm.save("tgt_segm.nii.gz")


    transfer = Transfer3D(native_img, native_seg, target_img, mode='base')
    tgt_segm = transfer.transfer(return_image=False)

    tgt_segm.save("tgt_segm_3d.nii.gz")
    target_img.save("tgt_img_3d.nii.gz")


if __name__ == "__main__":
    full_pipeline()