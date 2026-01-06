import os
import sys
import time
from typing import List, Optional, Tuple
import numpy as np
import PIL
import torch
import utils
from PTI.utils.alignment import get_landmark
import torchvision
from PIL import Image, ImageDraw
from insightface.app import FaceAnalysis

stylegan2_dir = os.path.abspath("stylegan2")
sys.path.insert(0, stylegan2_dir)


def load_model(
    network_pkl: str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqdog.pkl",
    device: torch.device = torch.device("cuda"),
    fp16: bool = True,
) -> torch.nn.Module:
    """
    Loads a pretrained StyleGAN2-ADA generator network from a pickle file.

    Args:
        network_pkl (str): The URL or local path to the network pickle file.
        device (torch.device): The device to use for the computation.
        fp16 (bool): Whether to use half-precision floating point format for the network weights.

    Returns:
        The pretrained generator network.
    """
    print('Loading networks from "%s"...' % network_pkl)
    # with dnnlib.util.open_url(network_pkl) as f:
    #     # chkpt = legacy.load_network_pkl(f, force_fp16=fp16)
    G = torch.load(network_pkl)
    G = G.to(device).eval()
    # G = chkpt["G_ema"].to(device).eval()
    for param in G.parameters():
        param.requires_grad_(False)

    # Create a new attribute called "activations" for the Generator class
    # This will be a list of activations from each layer
    G.__setattr__("activations", None)

    # Forward hook to collect features
    def hook(module, input, output):
        G.activations = output

    # Apply the hook to the 7th layer (256x256)
    for i, (name, module) in enumerate(G.synthesis.named_children()):
        if i == 6:
            print("Registering hook for:", name)
            module.register_forward_hook(hook)

    return G


def generate_W(
    _G: torch.nn.Module,
    seed: int = 0,
    network_pkl: Optional[str] = None,
    truncation_psi: float = 1.0,
    truncation_cutoff: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """
    Generates a latent code tensor in W+ space from a pretrained StyleGAN2-ADA generator network.

    Args:
        _G (torch.nn.Module): The generator network, with underscore to avoid streamlit cache error
        seed (int): The random seed to use for generating the latent code.
        network_pkl (Optional[str]): The path to the network pickle file. If None, the default network will be used.
        truncation_psi (float): The truncation psi value to use for the mapping network.
        truncation_cutoff (Optional[int]): The number of layers to use for the truncation trick. If None, all layers will be used.
        device (torch.device): The device to use for the computation.

    Returns:
        The W+ latent as a numpy array of shape [1, num_layers, 512].
    """
    G = _G
    torch.manual_seed(seed)
    z = torch.randn(1, G.z_dim).to(device)
    num_layers = G.synthesis.num_ws
    if truncation_cutoff == -1:
        truncation_cutoff = None
    elif truncation_cutoff is not None:
        truncation_cutoff = min(num_layers, truncation_cutoff)
    W = G.mapping(
        z,
        None,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
    )
    return W.cpu().numpy()


def forward_G(
    G: torch.nn.Module,
    W: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through the generator network.

    Args:
        G (torch.nn.Module): The generator network.
        W (torch.Tensor): The latent code tensor of shape [batch_size, latent_dim, 512].
        device (torch.device): The device to use for the computation.

    Returns:
        A tuple containing the generated image tensor of shape [batch_size, 3, height, width]
        and the feature maps tensor of shape [batch_size, num_channels, height, width].
    """
    if not isinstance(W, torch.Tensor):
        W = torch.from_numpy(W).to(device)

    img = G.synthesis(W, noise_mode="const", force_fp32=True)

    return img, G.activations[0]


def generate_image(
    W,
    _G: Optional[torch.nn.Module] = None,
    network_pkl: Optional[str] = None,
    class_idx=None,
    device=torch.device("cuda"),
) -> Tuple[PIL.Image.Image, torch.Tensor]:
    """
    Generates an image using a pretrained generator network.

    Args:
        W (torch.Tensor): A tensor of latent codes of shape [batch_size, latent_dim, 512].
        _G (Optional[torch.nn.Module]): The generator network. If None, the network will be loaded from `network_pkl`.
        network_pkl (Optional[str]): The path to the network pickle file. If None, the default network will be used.
        class_idx (Optional[int]): The class index to use for conditional generation. If None, unconditional generation will be used.
        device (str): The device to use for the computation.

    Returns:
        A tuple containing the generated image as a PIL Image object and the feature maps tensor of shape [batch_size, num_channels, height, width].
    """
    if _G is None:
        assert network_pkl is not None
        _G = load_model(network_pkl, device)
    G = _G

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise Exception(
                "Must specify class label with --class when using a conditional network"
            )
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print("warn: --class=lbl ignored when running on an unconditional network")

    ## Generate image
    img, features = forward_G(G, W, device)

    img = utils.tensor_to_PIL(img)

    return img, features

def get_handles_insightface(img, device):
    landmark_model_insightface = '/home/ethanf/workspace/repos/DragGAN/1k3d68.onnx'
    app = FaceAnalysis(model_dir=landmark_model_insightface)
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)

    faces = app.get(np.array(img))
    handles = faces[0]['landmark_3d_68'].astype(int)[48:68]
    return (
        torch.tensor(handles[:,:2], device=device).flip(-1).float())

def create_mask_lip(landmarks, i, img, image_size=1024):
    # landmarks0 = landmarks[0:17] # chin
    landmarks1 = landmarks[0:12] #outer mouth
    landmarks2 = landmarks[12:20] #inner mouth
    lm_tuple = []
    mask = Image.new(mode='RGB', size=(image_size, image_size))
    for item in landmarks1:
        lm_tuple.append(tuple(item))
    for it in landmarks2:
        lm_tuple.append((tuple(it)))
    masked_image = ImageDraw.Draw(mask)
    masked_image.polygon(lm_tuple, fill=(255, 255, 255))
    if img is not None:
        img_new = img.copy()
        masked_img = ImageDraw.Draw(img_new)
        masked_img.polygon(lm_tuple, fill=(255, 255, 255))
        # img_new.save(f'/home/ethanf/workspace/repos/DragGAN/obama_normalized/{}/frame_{i}_mask.jpg')
    mask = torchvision.transforms.functional.pil_to_tensor(mask).bool()[0, :, :]
    return mask

def create_mask_nose(landmarks, i, img, image_size=1024):

    nose = landmarks[29:35]
    radius = np.linalg.norm(nose[1]-nose[4]) * 1.5
    center = nose[1]

    mask = Image.new('L', (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    left_up = (center[0] - radius, center[1] - radius)
    right_down = (center[0] + radius, center[1] + radius)
    draw.ellipse([left_up, right_down], fill=255)

    mask = torchvision.transforms.functional.pil_to_tensor(mask).bool()[0, :, :]
    return mask

def create_mask_nose(landmarks, i, img, image_size=1024):

    nose = landmarks[29:35]
    radius = np.linalg.norm(nose[1]-nose[4]) * 1.5
    center = nose[1]

    mask = Image.new('L', (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    left_up = (center[0] - radius, center[1] - radius)
    right_down = (center[0] + radius, center[1] + radius)
    draw.ellipse([left_up, right_down], fill=255)

    mask = torchvision.transforms.functional.pil_to_tensor(mask).bool()[0, :, :]
    return mask

def create_mask_face(landmarks, i, img, image_size=1024):


    radius = np.linalg.norm(landmarks[30]-landmarks[8])
    center = landmarks[30]

    mask = Image.new('L', (image_size, image_size), 0)
    draw = ImageDraw.Draw(mask)
    left_up = (center[0] - radius, center[1] - radius)
    right_down = (center[0] + radius, center[1] + radius)
    draw.ellipse([left_up, right_down], fill=255)

    mask = torchvision.transforms.functional.pil_to_tensor(mask).bool()[0, :, :]
    return mask


def optimize(
    W: np.ndarray,
    G: torch.nn.Module,
    handle_points: List[Tuple[int, int]],
    target_points: List[Tuple[int, int]],
    r1: int = 3,
    tolerance: int = 2,
    max_iter: int = 10,
    lr: float = 0.1,
    multiplier: float = 1.0,
    lambda_: float = 150,
    device: torch.device = torch.device("cuda"),
    mask = None,
    chin_mask = None,
    face_mask = None,
    orig_F0 = None,
    predictor = None,
    handle_kf_filters=None,
    frame_num = None,
    dir_save_name = None,
    loss_limit: int = 3,
    target_resolution: int = 1024,
) -> np.ndarray:
    """
    Optimizes the latent code tensor W to generate an image that matches the target points.

    Args:
        W (np.ndarray): The initial latent code tensor of shape [1, num_layers, 512].
        G (torch.nn.Module): The generator network.
        handle_points (List[Tuple[int, int]]): The initial handle points as a list of (x, y) tuples.
        target_points (List[Tuple[int, int]]): The target points as a list of (x, y) tuples.
        r1 (int): The radius of the motion supervision loss.
        r2 (int): The radius of the point tracking.
        d (int): The tolerance for the handle points to reach the target points.
        max_iter (int): The maximum number of optimization iterations.
        lr (float): The learning rate for the optimizer.
        multiplier (float): The speed multiplier for the motion supervision loss.
        lambda_ (float): The weight of the motion supervision loss.
        device (torch.device): The device to use for the computation.
        empty: The st.empty object to display the intermediate images.
        display_every (int): The number of iterations between displaying intermediate images.
        target_resolution (int): The target resolution for the generated image.

    Returns:
        The optimized latent code tensor W as a numpy array of shape [1, num_layers, 512].
    """
    img, F0 = forward_G(G, W, device)
    F0_resized = torch.nn.functional.interpolate(
        F0,
        size=(target_resolution, target_resolution),
        mode="bilinear",
        align_corners=True,
    ).detach()

    orig_F0_resized = torch.nn.functional.interpolate(orig_F0, size=(target_resolution, target_resolution), mode = "bilinear",align_corners=True).detach()

    # Convert handle/target points to tensors and reorder to [y, x]
    handle_points: torch.tensor = (
        torch.tensor(handle_points, device=device).flip(-1).float()
    )
    handle_points_0 = handle_points.clone()
    target_points: torch.tensor = (
        torch.tensor(target_points, device=device).flip(-1).float()
    )

    W = torch.from_numpy(W).to(device).float()
    W.requires_grad_(False)

    # Only optimize the first 6 layers of W
    W_layers_to_optimize = W[:, :6].clone()
    W_layers_to_optimize.requires_grad_(True)

    optimizer = torch.optim.Adam([W_layers_to_optimize], lr=lr)
    prev_loss = np.inf
    loss_counter = 0
    for i in range(max_iter):
        start = time.perf_counter()

        # # Check if the handle points have reached the target points
        if torch.allclose(handle_points, target_points, atol=tolerance):
            break

        optimizer.zero_grad()

        # Detach only the unoptimized layers
        W_combined = torch.cat([W_layers_to_optimize, W[:, 6:].detach()], dim=1)

        # Run the generator to get the image and feature maps
        curr_img, F = forward_G(G, W_combined, device)


        ## Bilinear interpolate F to be same size as img
        F_resized = torch.nn.functional.interpolate(
            F,
            size=(target_resolution, target_resolution),
            mode="bilinear",
            align_corners=True,
        )

        # Compute the motion supervision loss
        loss, all_shifted_coordinates = motion_supervision(
            F_resized,
            F0_resized,
            handle_points,
            target_points,
            r1,
            lambda_,
            device,
            multiplier=multiplier,
        )

        if mask is not None:
            mask = mask.logical_not()
            mask = mask.to(device)
            loss_fix = torch.nn.functional.l1_loss(F_resized*mask, orig_F0_resized*mask)
            loss += lambda_ * loss_fix
        if chin_mask is not None:
            chin_mask = chin_mask.logical_not()
            chin_mask = chin_mask.to(device)
            chin_loss = torch.nn.functional.l1_loss(F_resized*chin_mask, orig_F0_resized*chin_mask)
            loss += 200 * chin_loss
        if face_mask is not None:
            face_mask = face_mask.logical_not()
            face_mask = face_mask.to(device)
            face_loss = torch.nn.functional.l1_loss(F_resized*face_mask, orig_F0_resized*face_mask)
            loss += 200 * face_loss

        # Compute the L2 regularization term

        l2_regularization = 100 * torch.nn.functional.mse_loss(W_layers_to_optimize, W[:, :6])
        # print(f' the l2_mse regularization is {l2_regularization.item()}')
        # Add the regularization term to the loss
        loss += l2_regularization

        loss.backward()

        optimizer.step()

        print(
            f"{i}\tLoss: {loss.item():0.2f}"
        )

        if prev_loss <= loss:
            loss_counter += 1
        if loss_counter >= loss_limit:
            break

        curr_img_pil = utils.tensor_to_PIL(curr_img)
        os.makedirs(os.path.join(dir_save_name,'tmp_dir/'),exist_ok=True)
        curr_img_pil.save(os.path.join(dir_save_name,'tmp_dir/tmp.jpg'))
        handle_points_obs = list(get_landmark(os.path.join(dir_save_name,'tmp_dir/tmp.jpg'), predictor))[48:68]
        if not os.path.exists(os.path.join(dir_save_name,'original_handles/')):
            os.mkdir(os.path.join(dir_save_name,'original_handles/'))
        np.save(os.path.join(dir_save_name,'original_handles/frame{frame_num}_{i}.npy'),
                np.array(handle_points_obs))
        for kf in handle_kf_filters:
            kf.predict()
        filtered_kf_states = [kf.update(np.array([handle_points_obs[i][0], handle_points_obs[i][1]])) for i, kf in enumerate(handle_kf_filters)]
        filtered_handles = [filtered_state[:2].astype(int) for filtered_state in filtered_kf_states]
        if not os.path.exists(os.path.join(dir_save_name,'filtered_handles/')):
            os.mkdir(os.path.join(dir_save_name,'filtered_handles/'))
        np.save(os.path.join(dir_save_name,f'filtered_handles/frame{frame_num}_{i}.npy'),
                np.array(filtered_handles))
        # filtered_handles = handle_points_obs

        handle_points = (
        torch.tensor(filtered_handles, device=device).flip(-1).float())

        prev_loss = loss.item()

    return torch.cat([W_layers_to_optimize, W[:, 6:]], dim=1).detach().cpu().numpy(), handle_points.flip(-1).cpu().long().numpy().tolist()

def motion_supervision(
    F: torch.Tensor,
    F0: torch.Tensor,
    handle_points: torch.Tensor,
    target_points: torch.Tensor,
    r1: int = 3,
    lambda_: float = 20.0,
    device: torch.device = torch.device("cuda"),
    multiplier: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Computes the motion supervision loss and the shifted coordinates for each handle point.

    Args:
        F (torch.Tensor): The feature map tensor of shape [batch_size, num_channels, height, width].
        F0 (torch.Tensor): The original feature map tensor of shape [batch_size, num_channels, height, width].
        handle_points (torch.Tensor): The handle points tensor of shape [num_handle_points, 2].
        target_points (torch.Tensor): The target points tensor of shape [num_handle_points, 2].
        r1 (int): The radius of the circular mask around each handle point.
        lambda_ (float): The weight of the reconstruction loss for the unmasked region.
        device (torch.device): The device to use for the computation.
        multiplier (float): The multiplier to use for the direction vector.

    Returns:
        A tuple containing the motion supervision loss tensor and a list of shifted coordinates
        for each handle point, where each element in the list is a tensor of shape [num_points, 2].
    """
    n = handle_points.shape[0]  # Number of handle points
    loss = 0.0
    all_shifted_coordinates = []  # List of shifted patches

    for i in range(n):
        # Compute direction vector
        target2handle = target_points[i] - handle_points[i]
        d_i = target2handle / (torch.norm(target2handle) + 1e-7) * multiplier
        if torch.norm(d_i) > torch.norm(target2handle):
            d_i = target2handle

        # Compute the mask for the pixels within radius r1 of the handle point
        mask = utils.create_circular_mask(
            F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r1
        ).to(device)
        # mask = utils.create_square_mask(F.shape[2], F.shape[3], center=handle_points[i].tolist(), radius=r1).to(device)

        # Find indices where mask is True
        coordinates = torch.nonzero(mask).float()  # shape [num_points, 2]

        # Shift the coordinates in the direction d_i
        shifted_coordinates = coordinates + d_i[None]
        all_shifted_coordinates.append(shifted_coordinates)

        h, w = F.shape[2], F.shape[3]

        # Extract features in the mask region and compute the loss
        F_qi = F[:, :, mask]  # shape: [C, H*W]

        # Sample shifted patch from F
        normalized_shifted_coordinates = shifted_coordinates.clone()
        normalized_shifted_coordinates[:, 0] = (
            2.0 * shifted_coordinates[:, 0] / (h - 1)
        ) - 1  # for height
        normalized_shifted_coordinates[:, 1] = (
            2.0 * shifted_coordinates[:, 1] / (w - 1)
        ) - 1  # for width
        # Add extra dimensions for batch and channels (required by grid_sample)
        normalized_shifted_coordinates = normalized_shifted_coordinates.unsqueeze(
            0
        ).unsqueeze(
            0
        )  # shape [1, 1, num_points, 2]
        normalized_shifted_coordinates = normalized_shifted_coordinates.flip(
            -1
        )  # grid_sample expects [x, y] instead of [y, x]
        normalized_shifted_coordinates = normalized_shifted_coordinates.clamp(-1, 1)

        # Use grid_sample to interpolate the feature map F at the shifted patch coordinates
        F_qi_plus_di = torch.nn.functional.grid_sample(
            F, normalized_shifted_coordinates, mode="bilinear", align_corners=True
        )
        # Output has shape [1, C, 1, num_points] so squeeze it
        F_qi_plus_di = F_qi_plus_di.squeeze(2)  # shape [1, C, num_points]

        loss += torch.nn.functional.l1_loss(F_qi.detach(), F_qi_plus_di)

    return loss, all_shifted_coordinates


