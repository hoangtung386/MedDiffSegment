import argparse
import gc
import os
import sys

sys.path.append(".")

from collections import OrderedDict

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.utils as vutils

from guided_diffusion import dist_util, logger
from guided_diffusion.data.brats_dataset import BRATSDataset3D
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.utils import staple


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    # Memory optimization settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

    # Clear initial cache
    if th.cuda.is_available():
        th.cuda.empty_cache()
        gc.collect()

    logger.log("Creating data loader...")
    tran_list = [
        transforms.Resize((args.image_size, args.image_size)),
    ]
    transform_test = transforms.Compose(tran_list)
    ds = BRATSDataset3D(args.data_dir, transform_test, test_flag=True)
    args.in_ch = 5

    # Force batch size = 1 for low memory
    datal = th.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    logger.log("Detecting model version from checkpoint...")

    # Load checkpoint to detect version
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")

    # Auto-detect version based on checkpoint keys
    checkpoint_keys = list(state_dict.keys())

    # Check for version-specific keys
    has_ss_former = any("ss_former" in k for k in checkpoint_keys)
    has_sea = any("sea" in k for k in checkpoint_keys)
    has_hwm = any("hwm" in k for k in checkpoint_keys)

    if has_ss_former and has_sea:
        detected_version = "medsegdiff-v2"
        logger.log("Detected: MedSegDiff-V2 (with SS-Former and SEA)")
    elif has_hwm:
        detected_version = "new"
        logger.log("Detected: New version (with HWM)")
    else:
        detected_version = "v1"
        logger.log("Detected: V1 version")

    # Override version if specified by user, otherwise use detected
    if args.version == "auto":
        args.version = detected_version
        logger.log(f"Using auto-detected version: {args.version}")
    else:
        logger.log(f"Using user-specified version: {args.version}")

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load model weights with partial loading support
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module." in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Try to load with strict matching first
    try:
        model.load_state_dict(new_state_dict, strict=True)
        logger.log("✓ Loaded checkpoint with strict matching")
    except RuntimeError as e:
        logger.log("⚠ Strict loading failed, attempting partial loading...")

        # Use partial loading
        if hasattr(model, "load_part_state_dict"):
            model.load_part_state_dict(new_state_dict)
            logger.log("✓ Loaded checkpoint with partial matching (load_part_state_dict)")
        else:
            # Manual partial loading
            model_dict = model.state_dict()
            matched_dict = {}
            mismatched = []

            for k, v in new_state_dict.items():
                if k in model_dict:
                    if v.shape == model_dict[k].shape:
                        matched_dict[k] = v
                    else:
                        mismatched.append(f"{k}: {v.shape} vs {model_dict[k].shape}")

            model_dict.update(matched_dict)
            model.load_state_dict(model_dict)
            logger.log(f"✓ Loaded {len(matched_dict)}/{len(new_state_dict)} layers")

            if mismatched:
                logger.log(f"⚠ {len(mismatched)} layers had size mismatch (skipped)")
    model.to(dist_util.dev())

    if args.use_fp16:
        if hasattr(model, "convert_to_fp16"):
            logger.log("Using FP16 precision...")
            model.convert_to_fp16()
        else:
            logger.log("Warning: Model doesn't support convert_to_fp16(), using FP32")
            logger.log("(MedSegDiffV2 uses self.dtype for FP16, no explicit conversion needed)")
            args.use_fp16 = False  # Disable FP16 flag

    model.eval()

    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False

    logger.log(f"Processing {len(datal)} batches...")

    processed_count = 0
    for batch_idx, (batch, m, path) in enumerate(datal):
        try:
            # Extract data
            if isinstance(batch, (list, tuple)):
                b, b_2_5d = batch
            else:
                b = batch
                b_2_5d = None

            # Create noisy input
            c = th.randn_like(b[:, :1, ...])
            img = th.cat((b, c), dim=1)

            # Extract slice ID
            slice_ID = path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split(".nii")[0]

            logger.log(f"Sampling batch {batch_idx+1}/{len(datal)} - {slice_ID}...")

            # Process with memory optimization
            with th.no_grad():
                enslist = []

                for i in range(args.num_ensemble):
                    model_kwargs = {}
                    if b_2_5d is not None:
                        model_kwargs["x_2_5d"] = b_2_5d.to(dist_util.dev())

                    # Run sampling
                    sample_fn = diffusion.p_sample_loop_known
                    sample, x_noisy, org, cal, cal_out = sample_fn(
                        model,
                        (1, 3, args.image_size, args.image_size),
                        img,
                        step=args.diffusion_steps,
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs,
                    )

                    co = th.tensor(cal_out)
                    enslist.append(co)

                    # Clear cache after each ensemble iteration
                    del sample, x_noisy, org, cal, cal_out, co
                    if th.cuda.is_available():
                        th.cuda.empty_cache()

                # Compute ensemble result
                ensres = staple(th.stack(enslist, dim=0)).squeeze(0)

                # Save result
                output_path = os.path.join(args.out_dir, f"{slice_ID}_output_ens.jpg")
                vutils.save_image(ensres, fp=output_path, nrow=1, padding=10)

                # Clean up
                del enslist, ensres, img, b, c
                if b_2_5d is not None:
                    del b_2_5d

            # Clear cache after each batch
            if th.cuda.is_available():
                th.cuda.empty_cache()
                gc.collect()

            processed_count += 1

            # Log memory usage every 10 batches
            if batch_idx % 10 == 0 and th.cuda.is_available():
                memory_allocated = th.cuda.memory_allocated() / 1024**3
                memory_reserved = th.cuda.memory_reserved() / 1024**3
                logger.log(
                    f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
                )

        except Exception as e:
            logger.log(f"Error processing batch {batch_idx}: {str(e)}")
            # Try to recover by clearing memory
            if th.cuda.is_available():
                th.cuda.empty_cache()
                gc.collect()
            continue

    logger.log(f"Sampling complete! Processed {processed_count}/{len(datal)} batches.")


def create_argparser():
    defaults = dict(
        data_name="BRATS3D",
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,  # Fixed at 1
        use_ddim=False,
        model_path="",
        num_ensemble=1,  # Reduced for memory
        gpu_dev="0",
        out_dir="./results/",
        multi_gpu=None,
        debug=False,
        version="auto",  # Auto-detect version from checkpoint
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
