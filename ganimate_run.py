
import torch
import os
import ganimate
import utils
import dlib
import numpy as np
import argparse
from LMKalmanFilter import KalmanFilter
import face_alignment
from PTI.utils.alignment import get_landmark



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run GANimate")
    parser.add_argument("--lmks_save_dir_path", type=str, default='./aligned_lmks',
                        help="Path to landmarks save directory")
    parser.add_argument("--res_dir", type=str, default='./output',
                        help="Path to output frames dir")
    parser.add_argument("--W_path", type=str, required=True, help="W of first frame")
    parser.add_argument("--G_path", type=str, required=True, help="G of generator after PTI")
    parser.add_argument("--PERSON", type=str, default='demo', help="Name of person to animate")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--save", type=bool, default=True, help="save landmarks")
    parser.add_argument("--max_frames", type=int, default=179, help="Max frames to animate")
    parser.add_argument("--display_lmks", type=bool, default=True, help="Display the landmarks ")
    parser.add_argument("--init_handles_kf_dir", type=str, help="Handle KF coefficients")
    parser.add_argument("--init_targets_kf_dir", type=str, help="Target KF coefficients")
    parser.add_argument("--target_lmks_dir", type=str, help="Target landmarks used for animation")

    args = parser.parse_args()
    PERSON = args.PERSON
    fps = args.fps
    max_frames = args.max_frames
    display_lmks = args.display_lmks
    init_handles_kf_dir = args.init_handles_kf_dir
    init_targets_kf_dir = args.init_targets_kf_dir
    target_lmks_dir = args.target_lmks_dir
    predictor = dlib.shape_predictor('./PTI/pretrained_models/align.dat')

    ## Default to CPU if no GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device='cuda')

    save_dir_path = args.lmks_save_dir_path
    os.makedirs(save_dir_path, exist_ok=True)
    res_dir = args.res_dir
    os.makedirs(res_dir, exist_ok=True)

    current_animate_dir = os.path.join(res_dir, args.PERSON)
    os.makedirs(current_animate_dir, exist_ok=True)

    W = torch.load(args.W_path)

    G = ganimate.load_model(args.G_path, device=device)

    W = W.cpu()
    W = W.detach().numpy()
    img, F0 = ganimate.generate_image(W, G, network_pkl=None, device=device)

    img.save(os.path.join(current_animate_dir, 'frame_0.jpg'))

    i = 0

    save = args.save
    if save:
        os.makedirs(os.path.join(current_animate_dir, 'filtered_targets'), exist_ok=True)
        os.makedirs(os.path.join(current_animate_dir, 'original_targets'), exist_ok=True)

    for _ in range(max_frames):
        if _ == 0:
            handles = list(get_landmark(os.path.join(current_animate_dir, 'frame_0.jpg'), predictor))[48:68]

            initial_states_handles = [np.load(
                os.path.join(init_handles_kf_dir, f'handles_kalman_res/state_{i}.npy'))
                for i in range(20)]
            initial_covariances_handles = [np.load(
                os.path.join(init_handles_kf_dir, f'handles_kalman_res/cov_{i}.npy'))
                for i in range(20)]

            kf_handles = [KalmanFilter(n_dim_state=6, n_dim_obs=2, initial_state=np.concatenate(
                (np.array([handles[i][0], handles[i][1]]), initial_states_handles[i][2:])),
                                       initial_covariance=initial_covariances_handles[i], dt=(1 / fps) / 7) for i in
                          range(20)]

            filtered_handles = handles
        else:
            filtered_handles = new_handles
            handles = new_handles

        generated_handles = list(get_landmark(os.path.join(current_animate_dir, f'frame_{_}.jpg'), predictor))[48:68]
        all_lmks = list(get_landmark(os.path.join(current_animate_dir, f'frame_{_}.jpg'), predictor))

        targets = list(np.load(os.path.join(target_lmks_dir, f'original_targets/frame{_}.npy')))
        if _ == 0:
            initial_states_targets = [np.load(
                os.path.join(init_targets_kf_dir, f'targets_kalman_res/state_{i}.npy'))
                for i in range(20)]
            initial_covariances_targets = [np.load(
                os.path.join(init_targets_kf_dir, f'targets_kalman_res/cov_{i}.npy'))
                for i in range(20)]

            kf_targets = [KalmanFilter(n_dim_state=6, n_dim_obs=2, initial_state=np.concatenate(
                (np.array([targets[i][0], targets[i][1]]), initial_states_targets[i][2:])),
                                       initial_covariance=initial_covariances_targets[i], dt=1 / fps) for i in
                          range(20)]
            for kf in kf_targets:
                kf.predict()

            filtered_kf_states = [kf.update(np.array([targets[i][0], targets[i][1]])) for i, kf in
                                  enumerate(kf_targets)]
            filtered_targets = targets
        else:
            for kf in kf_targets:
                kf.predict()

            filtered_kf_states = [kf.update(np.array([targets[i][0], targets[i][1]])) for i, kf in
                                  enumerate(kf_targets)]
            filtered_targets = [filtered_state[:2].astype(int) for filtered_state in filtered_kf_states]

        if save:
            np.save(os.path.join(current_animate_dir, f'filtered_targets/frame{_}.npy'),
                    # np.array(filtered_targets))
                    np.array(targets))
            np.save(os.path.join(current_animate_dir, f'original_targets/frame{_}.npy'),
                    np.array(targets))

        mask = ganimate.create_mask_lip(generated_handles, _, img)
        mask_nose = ganimate.create_mask_nose(all_lmks, _, img)
        mask_face = ganimate.create_mask_face(all_lmks, _, img)

        if display_lmks:
            if _ > 1:
                targets_to_display = list(np.load(
                    os.path.join(current_animate_dir, f'filtered_targets/frame{0 + _ - 1}.npy')))
            else:
                targets_to_display = targets
            utils.draw_handle_target_points(img, filtered_handles, targets_to_display)
            img.save(os.path.join(current_animate_dir, f'frame_{_}_lm.jpg'))
        new_W, new_handles = ganimate.optimize(
            W=W,
            G=G,
            handle_points=filtered_handles,
            target_points=targets,
            r1=3,
            tolerance=2,
            max_iter=5,
            lr=2e-3,
            multiplier=1.0,
            mask=mask,
            chin_mask=mask_nose,
            face_mask=mask_face,
            predictor=predictor,
            handle_kf_filters=kf_handles,
            dir_save_name=current_animate_dir,
            frame_num=_,
            orig_F0=F0,
            target_resolution=1024,
            device=device
        )
        new_img, new_F0 = ganimate.generate_image(new_W, G, network_pkl=None, device=device)
        new_img.save(os.path.join(current_animate_dir, f'frame_{_ + 1}.jpg'))
        W = new_W.copy()
        img = new_img
