import numpy as np
import torch
from torch.backends import cudnn
from tqdm import tqdm
from ESRGAN.arch import EsrganGenerator
import common.utils as utils
from PIL import Image
import os
import config
from common.imresize import imresize
import UnAI.interplot

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
model_name = ""
scale = 1
root_outputs_dir = ""
weight_file = ""
device = 'cpu'


def run_on_reference(model):
    for idx, img_dir in enumerate(config.datasets_path):
        dataset = config.datasets_name[idx]
        outputs_dir = root_outputs_dir + dataset + '/'
        utils.mkdirs(outputs_dir)
        img_dir = img_dir + '/'
        img_list = os.listdir(img_dir)
        with tqdm(total=len(img_list)) as t:
            t.set_description(f"Run {model_name}x{scale} on {dataset}: ")
            for img_name in img_list:
                image = utils.loadIMG_crop(img_dir + img_name, scale)
                if image.mode == 'L':
                    image = image.convert('RGB')
                hr_image = np.array(image)

                lr_image = imresize(hr_image, 1. / scale, 'bicubic')

                lr = lr_image.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
                lr /= 255.
                lr = torch.from_numpy(lr).to(device).unsqueeze(0)
                with torch.no_grad():
                    sr = model(lr)
                sr = sr.mul(255.0).cpu().numpy().squeeze(0)
                sr = np.clip(sr, 0.0, 255.0).transpose([1, 2, 0])
                # GPU tensor -> CPU tensor -> numpy
                sr = np.array(sr).astype(np.uint8)
                sr = Image.fromarray(sr)  # hw -> wh
                if image.mode == 'L':
                    sr = sr.convert('L')
                sr.save(outputs_dir + img_name)
                t.update(1)
    if config.isMetrics:
        utils.calc_metrics_of_model_on_datasets(config.calc_metrics, model_name, scale, root_outputs_dir,
                                                config.datasets_name, config.datasets_path)


def run_on_no_reference(model):
    for idx, img_dir in enumerate(config.real_datasets_path):
        dataset = config.real_datasets_name[idx]
        outputs_dir = root_outputs_dir + dataset + '/'
        # outputs_dir = config.save_path_root
        utils.mkdirs(outputs_dir)
        img_list = os.listdir(img_dir)
        with tqdm(total=len(img_list)) as t:
            t.set_description(f"Run {model_name}x{scale} on {dataset}: ")
            for img_name in img_list:
                image = utils.loadIMG_crop(img_dir + img_name, scale)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                lr_image = np.array(image)
                sub_imgs, x_num = utils.divide_lr(lr_image, max_size=512)
                sr_imgs = []
                for sub_lr in sub_imgs:
                    lr = sub_lr.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
                    lr /= 255.
                    lr = torch.from_numpy(lr).to(device).unsqueeze(0)
                    with torch.no_grad():
                        sr = model(lr)
                    sr = sr.mul(255.0).cpu().numpy().squeeze(0)
                    sr = np.clip(sr, 0.0, 255.0).transpose([1, 2, 0])
                    # GPU tensor -> CPU tensor -> numpy
                    sr = np.array(sr).astype(np.uint8)
                    sr_imgs.append(sr)
                sr = utils.catch_sub_imgs(sr_imgs, x_num,
                                          (lr_image.shape[0]*scale, lr_image.shape[1]*scale, lr_image.shape[2]))
                sr = Image.fromarray(sr.astype(np.uint8))  # hw -> wh
                # if image.mode == 'L':
                sr = sr.convert(image.mode)
                sr.save(outputs_dir + img_name)
                config.finishFlag.save(outputs_dir + img_name.replace('.', "_finish."))
                # sr.save(outputs_dir + f'{model_name}_x{scale}.png')
                t.update(1)
    if config.isMetrics:
        utils.calc_metrics_of_model_on_datasets(config.no_refer_metrics, model_name, scale,
                                                root_outputs_dir, config.real_datasets_name, config.real_datasets_path)


def run(config_index):
    global scale, root_outputs_dir, weight_file, device, model_name
    scale = int(config.models_scale[config_index])
    model_name = config.models_name[config_index]
    root_outputs_dir = config.save_path_root + model_name + '/' + f'x{scale}' + '/'
    weight_file = config.models_param[config_index]
    if not os.path.exists(weight_file):
        print(f'Weight file not exist!:\n{weight_file}\n')
        return 0
    for dataset_path in config.datasets_path:
        if not os.path.exists(dataset_path):
            print(f"Image file not exist!:\n{dataset_path}\n")
            return 0

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = EsrganGenerator().to(device)
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['params_ema'])
    model.eval()
    if config.datasets_name:
        run_on_reference(model)
    if config.real_datasets_name:
        # UnAI.interplot.run_on_no_reference(scale, 'bicubic')
        run_on_no_reference(model)
