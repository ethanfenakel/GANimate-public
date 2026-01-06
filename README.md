# GANimate
![GANimate GIF](https://github.com/ethanfenakel/GANimate-public/blob/main/new.gif)

**GANimate** is an ultra-efficient talking-face animation framework that generates realistic lip-synchronized facial motion from a single static portrait and 2D lip landmarks.

The method is designed for human–computer interaction (HCI) scenarios and emphasizes efficiency, modularity, and temporal stability.

---

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ethanfenakel/GANimate-public.git
cd GANimate-public

git clone https://github.com/danielroich/PTI.git
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git

pip install -r requirements.txt

```

---

### Pretrained Models

| Path | Description |
|------|-------------|
| [FFHQ StyleGAN](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl) | StyleGAN2-ADA model trained on FFHQ with 1024×1024 output resolution |
| [Dlib Alignment](https://drive.google.com/file/d/1HKmjg6iXsWr4aFPuU0gBXPGR83wqMzq7/view?usp=sharing) | Dlib alignment used for image preprocessing |
| [FFHQ e4e Encoder](https://drive.google.com/file/d/1ALC5CLA89Ouw40TwvxcwebhzWXM5YSCm/view?usp=sharing) | Pretrained e4e encoder |

---
### Run GANimate
```bash

python ganimate_run.py --res_dir
./output
--W_path
./input/demo_W.pt
--G_path
./input/model_demo_W.pt
--init_handles_kf_dir
./input
--init_targets_kf_dir
./input
--target_lmks_dir
./input

```

---
### Run Preprocess to obtain W and G
```bash
python pti_ganimate.py 
--image_name demo
```




