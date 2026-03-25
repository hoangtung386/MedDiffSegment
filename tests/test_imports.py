try:
    from guided_diffusion.models import UNetMedSegDiffV2, UNetV1, UNetNew, EncoderUNetModel, SuperResModel, ResBlock, AttentionBlock, GenericUNet, SS_Former
    print("models imports work")
    from guided_diffusion.data.brats_dataset import BRATSDataset, BRATSDataset3D
    from guided_diffusion.data.isic_dataset import ISICDataset
    from guided_diffusion.data.custom_dataset import CustomDataset, CustomDataset3D
    from guided_diffusion.data.btcv_dataset import BTCVLoader
    print("data imports work")
    import yaml
    print("yaml import works")

    # Check if train.py can be compiled successfully
    import py_compile
    py_compile.compile('scripts/train.py')
    print("train.py compiled successfully.")
    print("All imports tested successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
