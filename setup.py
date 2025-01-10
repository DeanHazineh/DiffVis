from setuptools import setup, find_packages

setup(
    name="diffvis",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "dflat-opt==4.1.0",
        "OpenEXR",
        "torch",
        "torchvision",
        "xformers",
        "pytorch-lightning",
        "matplotlib",
        "einops",
        "omegaconf",
        "numpy",
        "natsort",
        "tqdm",
        "pandas",
        "opencv-python",
        "scipy",
        "h5py",
        "ipykernel",
        "scikit-image",
        "torchmetrics",
    ],
)
