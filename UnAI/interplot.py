import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from common import utils
from common.imresize import imresize


def run_on_no_reference(scale, method):
    root_outputs_dir = config.save_path_root + method + '/' + f'x{scale}' + '/'
    for idx, img_dir in enumerate(config.real_datasets_path):
        dataset = config.real_datasets_name[idx]
        outputs_dir = root_outputs_dir + dataset + '/'
        # outputs_dir = config.save_path_root
        utils.mkdirs(outputs_dir)
        img_dir = img_dir + '/'
        img_list = os.listdir(img_dir)
        with tqdm(total=len(img_list)) as t:
            t.set_description(f"Run {method}x{scale} on {dataset}: ")
            for img_name in img_list:
                image = utils.loadIMG_crop(img_dir + img_name, scale)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                lr_image = np.array(image)
                sub_imgs, x_num = utils.divide_lr(lr_image, max_size=2048)
                sr_imgs = []
                for sub_lr in sub_imgs:
                    sr = imresize(sub_lr, scale, method)
                    sr_imgs.append(sr)
                sr = utils.catch_sub_imgs(sr_imgs, x_num,
                                          (lr_image.shape[0] * scale, lr_image.shape[1] * scale, lr_image.shape[2]))
                sr = Image.fromarray(sr.astype(np.uint8))  # hw -> wh
                sr = sr.convert(image.mode)
                sr.save(outputs_dir + img_name)
                config.finishFlag.save(outputs_dir + img_name.replace('.', "_finish."))
                t.update(1)
    if config.isMetrics:
        utils.calc_metrics_of_model_on_datasets(config.no_refer_metrics, method, scale,
                                                root_outputs_dir, config.real_datasets_name, config.real_datasets_path)


def run(config_index):
    scale = int(config.models_scale[config_index])
    method = config.models_name[config_index]
    for dataset_path in config.datasets_path:
        if not os.path.exists(dataset_path):
            print(f"Image file not exist!:\n{dataset_path}\n")
            return 0
    if config.real_datasets_name:
        run_on_no_reference(scale, method)
