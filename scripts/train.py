import argparse
import sys

import yaml

sys.path.append("../")
sys.path.append("./")
import logging
from pathlib import Path

import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.data.brats_dataset import BRATSDataset, BRATSDataset3D
from guided_diffusion.data.custom_dataset import CustomDataset, CustomDataset3D
from guided_diffusion.data.isic_dataset import ISICDataset
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.train_util import TrainLoop

logger_vis = logging.getLogger(__name__)
try:
    from visdom import Visdom

    _viz = Visdom(port=8850)
    USE_VISDOM = True
except ImportError:
    USE_VISDOM = False
    logger_vis.warning("visdom not installed - visualization disabled.")

import torchvision.transforms as transforms


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config")

    # Defaults base on model_and_diffusion_defaults
    defaults = dict(
        data_name="BRATS3D",
        data_dir="../dataset/brats2020/training",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None,  # "/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev="0",
        multi_gpu=None,  # "0,1,2"
        out_dir="./results/",
        # New loss weights for MedSegDiffV2Loss
        seg_loss_weight=1.0,
        cal_loss_weight=0.5,
        bce_loss_weight=1.0,
    )
    defaults.update(model_and_diffusion_defaults())

    args, unknown = parser.parse_known_args()

    if args.config:
        cfg = load_config(args.config)

        # Flatten the YAML structure mapping back to flat defaults
        if "data" in cfg:
            defaults["data_name"] = cfg["data"].get("name", defaults["data_name"])
            defaults["data_dir"] = cfg["data"].get("dir", defaults["data_dir"])

        if "model" in cfg:
            for k, v in cfg["model"].items():
                if k in defaults:
                    defaults[k] = v

        if "diffusion" in cfg:
            for k, v in cfg["diffusion"].items():
                if k in defaults:
                    defaults[k] = v

        if "training" in cfg:
            for k, v in cfg["training"].items():
                if k in defaults:
                    defaults[k] = v

    # Let CLI overrides have final say
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == "ISIC":
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_train)
        args.in_ch = 4
    elif args.data_name == "BRATS":
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    elif args.data_name == "BRATS3D":
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_train = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir, transform_train, test_flag=False)
        args.in_ch = 5
    elif any(Path(args.data_dir).glob("**.nii.gz")):
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
        ]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ", args.data_dir)
        ds = CustomDataset3D(args, args.data_dir, transform_train)
        args.in_ch = 4
    else:
        tran_list = [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
        transform_train = transforms.Compose(tran_list)
        print("Your current directory : ", args.data_dir)
        ds = CustomDataset(args, args.data_dir, transform_train)
        args.in_ch = 4

    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model, device_ids=[int(id) for id in args.multi_gpu.split(",")])
        model.to(device=th.device("cuda", int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, maxt=args.diffusion_steps
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        # Pass new loss weights
        seg_loss_weight=args.seg_loss_weight,
        cal_loss_weight=args.cal_loss_weight,
        bce_loss_weight=args.bce_loss_weight,
    ).run_loop()


if __name__ == "__main__":
    main()
