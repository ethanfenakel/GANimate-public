from cdfvd import fvd
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run FVD")
    parser.add_argument("--orig_path", type=str, required=True, help="orig_path_of_videos")
    parser.add_argument("--gen_path", type=str, required=True, help="gen_path_of_videos")

    args = parser.parse_args()

    orig_path = args.orig_path
    gen_path = args.gen_path
    torch.cuda.empty_cache()
    fvd = fvd.cdfvd('videomae', ckpt_path=None, half_precision=True)
    print('hi')
    res = {}
    for vid_name in os.listdir(orig_path):
        # torch.cuda.empty_cache()
        # fvd = fvd.cdfvd('videomae', ckpt_path=None, half_precision=True)
        real_videos = fvd.load_videos(os.path.join(orig_path,vid_name), data_type='image_folder', sequence_length=20)
        generated_videos = fvd.load_videos(os.path.join(gen_path,vid_name), data_type='image_folder', sequence_length=20)
        fvd.compute_real_stats(real_videos)
        fvd.compute_fake_stats(generated_videos)
        fvd_score = fvd.compute_fvd_from_stats()
        res[vid_name] = fvd_score

    score = np.mean(np.array(list(res.values())))
    print(f'{score}')
