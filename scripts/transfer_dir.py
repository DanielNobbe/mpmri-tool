import os
import shutil 
from itertools import repeat
from multiprocessing import get_context

from tqdm import tqdm

from mpmri_tool.transfer.zbinden import TransferSlice, Transfer3D
from mpmri_tool.transfer.resample import Transfer3D as Transfer3DResample
from mpmri_tool.data import RadData
from mpmri_tool.measure.zbinden import Measurement

def old():
    native_img = RadData("/Users/daniel/data/zbinden/P3/toy-version/inphase/mr/ISP5003_1002_121842.245000_0000.nii.gz")
    native_seg = RadData("/Users/daniel/data/zbinden/P3/toy-version/inphase/segm/ISP5003_1002_121842.245000.nii.gz")

    target_img = RadData("/Users/daniel/data/zbinden/P3/toy-version/maps/ISP5003_122656.909000_t1map_longt1_mbh_moco_t1.nii.gz")

    # transfer = TransferSlice(native_img, native_seg, target_img)

    # slice_idx = 40

    # tgt_slice, tgt_segm = transfer.transfer(slice_idx, return_image=True)

    # tgt_slice.save("tgt_slice.nii.gz")
    # tgt_segm.save("tgt_segm.nii.gz")

    os.makedirs("debug", exist_ok=True)

    transfer_ss = Transfer3D(native_img, native_seg, target_img, mode='slice-selection')
    tgt_img, tgt_segm_ss = transfer_ss.transfer(return_image=True, debug=True)
    tgt_img.save("debug/transfer_tgt_img_3d_ss.nii.gz")
    tgt_segm_ss.save("debug/transfer_tgt_segm_3d_ss.nii.gz")

    transfer_base = Transfer3D(native_img, native_seg, target_img, mode='base')
    tgt_segm_base = transfer_base.transfer(return_image=False, debug=True)
    tgt_segm_base.save("debug/transfer_tgt_segm_3d_base.nii.gz")

    transfer_resample = Transfer3DResample(native_img, native_seg, target_img, order=0)
    tgt_segm_res = transfer_resample.transfer(return_image=False, debug=True)
    tgt_segm_res.save("debug/transfer_tgt_segm_3d_res.nii.gz")

    transfer_resample_3 = Transfer3DResample(native_img, native_seg, target_img, order=3)
    tgt_segm_3 = transfer_resample_3.transfer(return_image=False, debug=True)
    tgt_segm_3.save("debug/transfer_tgt_segm_3d_res_order3.nii.gz")


def process_subjects(subject_id, files, output_dir: str, native_dir: str):
    if "native_img" not in files or "native_segm" not in files or "maps" not in files:
        print(f"Skipping subject {subject_id} as it is missing native image, segmentation, or maps.")
        return
    print(f"Processing subject {subject_id}...")
    native_img = RadData(files["native_img"])
    native_segm = RadData(files["native_segm"])

    # create output directory for this subject
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)

    # copy the inphase files for this subject into a subdirectory
    inphase_output_dir = os.path.join(subject_output_dir, "inphase")
    if not os.path.exists(inphase_output_dir):
        os.makedirs(inphase_output_dir, exist_ok=True)
    shutil.copy(files["native_img"], os.path.join(inphase_output_dir, subject_id + "_mr.nii.gz"))
    shutil.copy(files["native_segm"], os.path.join(inphase_output_dir, subject_id + "_segm.nii.gz"))
    

    transferred_dir = os.path.join(subject_output_dir, "transferred")
    os.makedirs(transferred_dir, exist_ok=True)

    for map_type, map_file in files["maps"].items():
        print(f"  Transferring segmentation to map type {map_type}...")

        # first, copy this map file to the output directory
        target_img_output_path = os.path.join(transferred_dir, map_type + ".nii.gz")
        shutil.copy(map_file, target_img_output_path)

        target_img = RadData(map_file)

        # transfer using slice selection
        transfer_ss = Transfer3D(native_img, native_segm, target_img, mode='slice-selection')
        target_segm_ss = transfer_ss.transfer(return_image=False, debug=False)
        target_segm_ss_output_path = os.path.join(transferred_dir, map_type + "_segm_ss.nii.gz")
        target_segm_ss.save(target_segm_ss_output_path)

        # transfer using base method
        transfer_base = Transfer3D(native_img, native_segm, target_img, mode='base')
        target_segm_base = transfer_base.transfer(return_image=False, debug=False)
        target_segm_base_output_path = os.path.join(transferred_dir, map_type + "_segm_base.nii.gz")
        target_segm_base.save(target_segm_base_output_path)


def main():
    """Transfer segmentations from native to target images using different methods.
    
    With this script, we run the transfer on a full set of images and segmentations.

    Note: Directories are structured as follows:
    - There is a `inphase` directory containing the native images and segmentations.
        - `mr` subdirectory contains the native MR images. The filenames follow the pattern `ISP{subject_id}_*.nii.gz`.
           The numbers in * are not relevant for us
        - `segm` subdirectory contains the native segmentations. The filenames follow the pattern `ISP{subject_id}_*.nii.gz`.
    - There is a `maps` directory containing the target images. The filenames follow the pattern `ISP{subject_id}_*_{sequence_type}.nii.gz`.
        - The sequence_type can be:
            - t2map_flash_moco_t2
            - t1map_longt1_mbh_moco_t1
            - t1map_shortt1_4sl_mbh_15min_moco_t1

    As output, we will create a folder for each subject, with a folder `inphase` that is is a symlink to the original
    inphase folder for that subject, and a folder `transferred` that contains the transferred segmentations and the target images.
    """

    native_dir = "/Users/daniel/data/zbinden/P3/inphase"
    native_segm_dir = os.path.join(native_dir, "segm")
    native_img_dir = os.path.join(native_dir, "mr")

    maps_dir = "/Users/daniel/data/zbinden/P3/maps"

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    map_types = [
        "t2map_flash_moco_t2",
        "t1map_longt1_mbh_moco_t1",
        "t1map_shortt1_4sl_mbh_15min_moco_t1"
    ]

    # now open the directories, and find the relevant files
    native_img_files = [f for f in os.listdir(native_img_dir) if f.endswith(".nii.gz")]
    native_segm_files = [f for f in os.listdir(native_segm_dir) if f.endswith(".nii.gz")]
    map_files = [f for f in os.listdir(maps_dir) if f.endswith(".nii.gz")]

    # create a mapping from subject id to files
    subjects = {}
    for f in native_img_files:
        subject_id = f.split("_")[0]  # Extract the subject id from the filename
        if subject_id not in subjects:
            subjects[subject_id] = {}
        subjects[subject_id]["native_img"] = os.path.join(native_img_dir, f)
    for f in native_segm_files:
        subject_id = f.split("_")[0]  # Extract the subject id from the filename
        if subject_id not in subjects:
            subjects[subject_id] = {}
        subjects[subject_id]["native_segm"] = os.path.join(native_segm_dir, f)
    for f in map_files:
        parts = f.split("_")
        subject_id = parts[0]  # Extract the subject id from the filename
        # find one of the map types in the filename, and add it under that key to the dict
        for map_type in map_types:
            if map_type in f:
                if subject_id not in subjects:
                    raise ValueError(f"Found map file {f} for subject {subject_id}, but no native image/segmentation found.")
                if "maps" not in subjects[subject_id]:
                    subjects[subject_id]["maps"] = {}
                subjects[subject_id]["maps"][map_type] = os.path.join(maps_dir, f)
                break

    print(f"Found {len(subjects)} subjects with native images and segmentations.")
    print(f"Found map files for {sum(1 for s in subjects.values() if 'maps' in s)} subjects.")

    # now, for each subject, perform the transfer for each map type
    
    args = zip(subjects.keys(), subjects.values(), repeat(output_dir), repeat(native_dir))
    with tqdm(total=len(subjects)) as pbar:
        with get_context("spawn").Pool() as pool:
            for _ in pool.starmap(process_subjects, args):
                pbar.update()

    # for subject_id, files in tqdm(subjects.items()):
    #     if "native_img" not in files or "native_segm" not in files or "maps" not in files:
    #         print(f"Skipping subject {subject_id} as it is missing native image, segmentation, or maps.")
    #         continue
    #     print(f"Processing subject {subject_id}...")
    #     native_img = RadData(files["native_img"])
    #     native_segm = RadData(files["native_segm"])

    #     # create output directory for this subject
    #     subject_output_dir = os.path.join(output_dir, subject_id)
    #     os.makedirs(subject_output_dir, exist_ok=True)
    #     # create a symlink to the inphase directory
    #     inphase_symlink = os.path.join(subject_output_dir, "inphase")
    #     if not os.path.exists(inphase_symlink):
    #         os.symlink(os.path.abspath(native_dir), inphase_symlink)

    #     transferred_dir = os.path.join(subject_output_dir, "transferred")
    #     os.makedirs(transferred_dir, exist_ok=True)

    #     for map_type, map_file in files["maps"].items():
    #         print(f"  Transferring segmentation to map type {map_type}...")

    #         # first, copy this map file to the output directory
    #         target_img_output_path = os.path.join(transferred_dir, map_type + ".nii.gz")
    #         shutil.copy(map_file, target_img_output_path)

    #         target_img = RadData(map_file)

    #         # transfer using slice selection
    #         transfer_ss = Transfer3D(native_img, native_segm, target_img, mode='slice-selection')
    #         target_segm_ss = transfer_ss.transfer(return_image=False, debug=False)
    #         target_segm_ss_output_path = os.path.join(transferred_dir, map_type + "_segm_ss.nii.gz")
    #         target_segm_ss.save(target_segm_ss_output_path)

    #         # transfer using base method
    #         transfer_base = Transfer3D(native_img, native_segm, target_img, mode='base')
    #         target_segm_base = transfer_base.transfer(return_image=False, debug=False)
    #         target_segm_base_output_path = os.path.join(transferred_dir, map_type + "_segm_base.nii.gz")
    #         target_segm_base.save(target_segm_base_output_path)

    print(f"Transferring completed for {len(subjects)} subjects. Results are in the '{output_dir}' directory.")

    





if __name__ == "__main__":
    main()
    # full_pipeline()