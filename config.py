# 'psnr', 'ssim', 'niqe', 'lpips'
import numpy
from PIL import Image
isMetrics = False
calc_metrics = ['psnr', 'ssim', 'niqe']
no_refer_metrics = ['niqe', 'lpips']
save_path_root = "./res/"
# gpus = ["0"]
models_name = []
models_scale = []
models_param = []
datasets_name = []
datasets_path = []
real_datasets_name = []
real_datasets_path = []
finishFlag = Image.fromarray(numpy.zeros(1).astype(numpy.uint8))
