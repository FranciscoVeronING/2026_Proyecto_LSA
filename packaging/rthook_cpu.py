# PyInstaller: el motor empaquetado no usa GPU.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.pop("LSA_USE_GPU", None)
