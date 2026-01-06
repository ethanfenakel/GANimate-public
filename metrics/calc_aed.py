import os
import glob
import numpy as np
from scipy.io import loadmat
import argparse


def calc_aed_apd(gen_path, orig_path, start_idx=80, end_idx=144):

    orig_file_names = os.listdir(orig_path)

    all_diffs = []
    for video_name in orig_file_names:

        gen_vid_path = os.path.join(gen_path, video_name + '.mp4')
        if not os.path.exists(gen_vid_path):
            print(f'video {video_name} does not exist in generated videos... skipping')
            continue
        orig_coeffs_path = os.path.join(orig_path, video_name, 'coeffs')
        orig_mat_files = sorted(glob.glob(f'{orig_coeffs_path}/**/*.mat', recursive=True))
        if len(orig_mat_files) == 0:
            print('Error!!')
            continue
        orig_mat = loadmat(orig_mat_files[0])["coeff"]

        gen_coeffs_path = os.path.join(gen_path,video_name+'.mp4','coeffs')
        gen_mat_files = sorted(glob.glob(f'{gen_coeffs_path}/**/*.mat', recursive=True))
        curr_diffs = []
        curr_mats = []
        for i, mat_path in enumerate(gen_mat_files):
            gen_mat = loadmat(mat_path)["coeff"]
            curr_mats.append(gen_mat)



            curr_diff = np.linalg.norm((orig_mat[:min(401,gen_mat.shape[0])]-gen_mat)[:,start_idx:end_idx], axis=1)
            curr_diffs.append(curr_diff)

            print('j')

        gen_mat_avg = (np.mean(curr_mats, axis=0))
        diff = np.linalg.norm((orig_mat[:min(401,gen_mat_avg.shape[0])]-gen_mat_avg)[:,start_idx:end_idx], axis=1)


        all_diffs.append(diff)

    aed = np.nanmean(np.array(all_diffs))
    return aed


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run AED and APD")
    parser.add_argument("--orig_path", type=str, required=True, help="orig_path_of_videos")
    parser.add_argument("--gen_path", type=str, required=True, help="gen_path_of_videos")
    parser.add_argument("--start_idx", type=int, default=80,required=True, help="start_idx for expression 80 for pose 224")
    parser.add_argument("--end_idx", type=int, default=144, required=True,
                        help="end_idx for expression 144 for pose 227")



    args = parser.parse_args()

    score = calc_aed_apd(args.gen_path, args.orig_path)
    print(f'{score}')

    # pose: 224:227
    # expression: 80:144





