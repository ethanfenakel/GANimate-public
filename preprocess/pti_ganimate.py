import os
import argparse
from PTI.configs import paths_config, hyperparameters, global_config
from PTI.utils.align_data import pre_process_images
from PTI.scripts.run_pti import run_PTI


if __name__ == '__main__':
    CODE_DIR = 'PTI'
    current_directory = os.getcwd()
    save_path = os.path.join(os.path.dirname(current_directory), CODE_DIR, 'pretrained_models')
    os.makedirs(save_path, exist_ok=True)

    parser = argparse.ArgumentParser(description="Run PTI Preprocessing")
    parser.add_argument("--image_dir_name", type=str, default='demo',required=True, help="image_dir_name")
    parser.add_argument("--image_name", type=str, default='demo_woman',required=True, help="image_name")
    parser.add_argument("--device", type=str, default='cuda', help="device")
    parser.add_argument("--input_data_path", type=str, required=True,
                        help="input_data_path")

    args = parser.parse_args()
    image_dir_name = args.image_dir_name
    image_name = args.image_name
    global_config.device = args.device
    paths_config.e4e = './PTI/pretrained_models/e4e_ffhq_encode.pt'
    paths_config.input_data_id = image_dir_name

    paths_config.input_data_path = args.input_data_path


    paths_config.stylegan2_ada_ffhq = './PTI/pretrained_models.ffhq.pkl'
    paths_config.checkpoints_dir = './PTI'
    paths_config.style_clip_pretrained_mappers = './PTI/pretrained_models'
    hyperparameters.use_locality_regularization = False

    os.makedirs(f'./{image_dir_name}_original', exist_ok=True)
    os.makedirs(f'./{image_dir_name}_processed', exist_ok=True)

    pre_process_images(args.input_data_path)

    model_id, embedding_dir_path = run_PTI(use_wandb=False, use_multi_id_training=False)

    print(f'model_id is {model_id}')
    print(f'embedding_dir_path is {embedding_dir_path}')
    print('done')

