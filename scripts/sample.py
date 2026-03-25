import argparse
import os
import random
import sys
from ssl import OP_NO_TLSv1

import nibabel as nib

sys.path.append(".")
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torchsummary import summary

from guided_diffusion import dist_util, logger
from guided_diffusion.data.brats_dataset import BRATSDataset, BRATSDataset3D
from guided_diffusion.data.custom_dataset import CustomDataset, CustomDataset3D
from guided_diffusion.data.isic_dataset import ISICDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.utils import staple

seed = 10
th.manual_seed(seed)
# th.cuda.manual_seed_all(seed) # Comment out to prevent early CUDA init
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == "ISIC":
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_test = transforms.Compose(tran_list)
        ds = ISICDataset(args, args.data_dir, transform_test, mode="Test")
        args.in_ch = 4

    elif args.data_name == "BRATS":
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_test = transforms.Compose(tran_list)
        ds = BRATSDataset(args.data_dir, transform_test, test_flag=True)
        args.in_ch = 5

    elif args.data_name == "BRATS3D":
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_test = transforms.Compose(tran_list)
        ds = BRATSDataset3D(args.data_dir, transform_test, test_flag=True)
        args.in_ch = 5

    elif any(Path(args.data_dir).glob("*/*.nii.gz")):
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_test = transforms.Compose(tran_list)
        print("Your current directory (3D NIfTI detected):", args.data_dir)
        ds = CustomDataset3D(args, args.data_dir, transform_test, mode="Test")
        args.in_ch = 4

    else:
        tran_list = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()]
        transform_test = transforms.Compose(tran_list)
        print("Your current directory (2D images):", args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_test, mode="Test")
        args.in_ch = 4

    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module." in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict
            break

    model.load_state_dict(new_state_dict)

    # --- FIX 8: The "Done Deal" Patch (All previous fixes + Output Wrapper) ---
    if args.use_fp16:
        print("Applying Fix 8: Flash Attn + SS_Former Patch + Input/Output Wrappers...")
        import math

        import torch.nn as nn
        import torch.nn.functional as F

        import guided_diffusion.nn as nn_module
        from guided_diffusion.models.base_blocks import QKVAttention
        from guided_diffusion.models.condition_net import SS_Former

        # 1. Monkey Patch: Force timestep_embedding to return FP16
        _orig_timestep_embedding = nn_module.timestep_embedding

        def half_timestep_embedding(*args, **kwargs):
            return _orig_timestep_embedding(*args, **kwargs).half()

        nn_module.timestep_embedding = half_timestep_embedding

        # 2. Monkey Patch: Flash Attention (prevent OOM)
        def efficient_qkv_forward(self, qkv):
            bs, width, length = qkv.shape
            ch = width // (3 * self.n_heads)
            q, k, v = qkv.chunk(3, dim=1)
            q = q.reshape(bs, self.n_heads, ch, length).transpose(-1, -2)
            k = k.reshape(bs, self.n_heads, ch, length).transpose(-1, -2)
            v = v.reshape(bs, self.n_heads, ch, length).transpose(-1, -2)
            out = F.scaled_dot_product_attention(q, k, v)
            out = out.transpose(-1, -2).reshape(bs, -1, length)
            return out

        QKVAttention.forward = efficient_qkv_forward

        # 3. Monkey Patch: SS_Former (FFT Float32 -> MLP Half)
        def patched_ss_former_forward(self, x, anchor_cond, semantic_cond):
            b, c, h, w = x.shape
            if anchor_cond.shape[-2:] != x.shape[-2:]:
                anchor_cond = F.interpolate(
                    anchor_cond, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            if semantic_cond.shape[-2:] != x.shape[-2:]:
                semantic_cond = F.interpolate(
                    semantic_cond, size=x.shape[-2:], mode="bilinear", align_corners=False
                )

            q = self.q_conv(x)
            k = self.k_conv(semantic_cond)
            v = self.v_conv(anchor_cond)

            q_fft = self.nbp_filter(q)
            k_fft = self.nbp_filter(k)

            scale = 1 / math.sqrt(c)
            weight = th.einsum("bchw,bchw->bhw", q_fft * scale, k_fft * scale)
            weight = th.softmax(weight.view(b, -1), dim=-1).view(b, h, w)
            attn = th.einsum("bhw,bchw->bchw", weight, v.float())

            attn = attn.permute(0, 2, 3, 1)
            mlp_out = self.mlp(attn.half())  # Cast to Half for MLP
            mlp_out = mlp_out.permute(0, 3, 1, 2)

            return self.proj_out(mlp_out) + x

        SS_Former.forward = patched_ss_former_forward

        # 4. Define Wrappers
        class HalfInputWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x, *args, **kwargs):
                return self.module(x.half(), *args, **kwargs)

        class OutputCastingWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x, *args, **kwargs):
                return self.module(x.half(), *args, **kwargs).float()

        # 5. Convert Model Weights
        def robust_fp16_convert(m):
            classname = m.__class__.__name__
            if isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                m.weight.data = m.weight.data.half()
                if m.bias is not None:
                    m.bias.data = m.bias.data.half()
            elif isinstance(m, nn.Linear):
                m.weight.data = m.weight.data.half()
                if m.bias is not None:
                    m.bias.data = m.bias.data.half()
            elif isinstance(m, nn.GroupNorm):
                if "GroupNorm32" not in classname:
                    m.weight.data = m.weight.data.half()
                    if m.bias is not None:
                        m.bias.data = m.bias.data.half()

        model.apply(robust_fp16_convert)

        # 6. Apply Wrappers
        if hasattr(model, "hwm"):
            print("  -> Wrapping hwm (Input: Float32 -> Half).")
            model.hwm = HalfInputWrapper(model.hwm)

        if hasattr(model, "out"):
            print("  -> Wrapping out (Input: Float32 -> Half -> Float32).")
            model.out = OutputCastingWrapper(model.out)

        print("  -> Fix 8 Applied Successfully.")

    # --- END FIX 8 ---

    model.to(dist_util.dev())
    model.eval()

    logger.log(f"Processing {len(datal)} batches...")

    for batch_idx in range(len(datal)):
        try:
            batch, m, path = next(data)
        except StopIteration:
            break

        if isinstance(batch, (list, tuple)):
            b, b_2_5d = batch
        else:
            b = batch
            b_2_5d = None

        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)

        if args.data_name == "ISIC":
            slice_ID = path[0].split("_")[-1].split(".")[0]
        elif args.data_name in ["BRATS", "BRATS3D"]:
            slice_ID = path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split(".nii")[0]
        else:
            slice_ID = Path(path[0]).stem

        logger.log(f"Sampling batch {batch_idx+1}/{len(datal)} - {slice_ID}...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):
            model_kwargs = {}
            if b_2_5d is not None:
                model_kwargs["x_2_5d"] = b_2_5d.to(dist_util.dev())

            start.record()

            sample_fn = (
                diffusion.p_sample_loop_known
                if not args.use_ddim
                else diffusion.ddim_sample_loop_known
            )

            # Disable gradient to save VRAM
            with th.no_grad():
                sample, x_noisy, org, cal, cal_out = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    img,
                    step=args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            # -------------------------------------------

            end.record()
            th.cuda.synchronize()
            print(f"Time for sample {i+1}: {start.elapsed_time(end):.2f}ms")

            # Move cal_out to CPU tensor safely
            if isinstance(cal_out, th.Tensor):
                co = cal_out.detach().cpu()
            else:
                co = th.tensor(cal_out)

            if hasattr(args, "version") and args.version == "new":
                enslist.append(sample[:, -1, :, :].detach().cpu())
            else:
                enslist.append(co)

            if args.debug:
                if args.data_name == "ISIC":
                    # ISIC visualization (detach + CPU for safety)
                    o = org[:, :-1, :, :].detach().cpu()
                    c = cal.repeat(1, 3, 1, 1).detach().cpu()
                    s = sample[:, -1, :, :].detach().cpu()
                    b_sz, h, w = s.size()
                    ss = s.clone()
                    ss = ss.view(s.size(0), -1)
                    ss -= ss.min(1, keepdim=True)[0]
                    ss /= ss.max(1, keepdim=True)[0]
                    ss = ss.view(b_sz, h, w)
                    ss = ss.unsqueeze(1).repeat(1, 3, 1, 1)
                    tup = (ss, o, c)

                elif args.data_name in ["BRATS", "BRATS3D"]:
                    # Handle mixed-shape mask list (BRATS3D specific)
                    # BRATS3D loader returns list [slice_mask_4D, volume_mask_5D],
                    # we only need the 4D tensor for visualization.
                    if isinstance(m, list):
                        # Find the 4D tensor (B, C, H, W) = slice mask
                        valid_masks = [x for x in m if isinstance(x, th.Tensor) and x.ndim == 4]

                        if len(valid_masks) > 0:
                            m_tensor = valid_masks[0]
                        elif len(m) > 0 and isinstance(m[0], th.Tensor):
                            m_tensor = m[0]
                        else:
                            m_tensor = th.tensor(m)
                    elif isinstance(m, th.Tensor):
                        m_tensor = m
                    else:
                        m_tensor = th.tensor(m)

                    # Move all tensors to CPU
                    m_cpu = m_tensor.detach().cpu()
                    s_cpu = sample.detach().cpu()
                    org_cpu = org.detach().cpu()
                    c_cpu = cal.detach().cpu()
                    co_cpu = co

                    # Format mask dimensions (take channel 0 from [B, C, H, W])
                    if m_cpu.ndim == 4:
                        m_vis = m_cpu[:, 0, :, :].unsqueeze(1)
                    elif m_cpu.ndim == 3:
                        m_vis = m_cpu.unsqueeze(1)
                    else:
                        m_vis = m_cpu

                    # Format sample dimensions
                    s_vis = s_cpu[:, -1, :, :].unsqueeze(1)

                    # Format original image channels
                    o1 = org_cpu[:, 0, :, :].unsqueeze(1)
                    o2 = org_cpu[:, 1, :, :].unsqueeze(1)
                    o3 = org_cpu[:, 2, :, :].unsqueeze(1)
                    o4 = org_cpu[:, 3, :, :].unsqueeze(1)

                    def norm_img(x):
                        mx = x.max()
                        return x / mx if mx > 0 else x

                    tup = (
                        norm_img(o1),
                        norm_img(o2),
                        norm_img(o3),
                        norm_img(o4),
                        m_vis,
                        s_vis,
                        c_cpu,
                        co_cpu,
                    )
                    # -----------------------------------------------------

                compose = th.cat(tup, 0)
                vutils.save_image(
                    compose,
                    fp=os.path.join(args.out_dir, f"{slice_ID}_output{i}.jpg"),
                    nrow=1,
                    padding=10,
                )

        ensres = staple(th.stack(enslist, dim=0)).squeeze(0)
        vutils.save_image(
            ensres, fp=os.path.join(args.out_dir, f"{slice_ID}_output_ens.jpg"), nrow=1, padding=10
        )

    logger.log("Sampling complete!")


def create_argparser():
    defaults = dict(
        data_name="BRATS3D",
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,
        gpu_dev="0",
        out_dir="./results/",
        multi_gpu=None,
        debug=False,
        version="new",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
