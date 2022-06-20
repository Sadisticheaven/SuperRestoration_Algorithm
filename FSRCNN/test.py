import numpy as np
import torch
import common.utils as utils
from FSRCNN.arch import N2_10_4
from common.imresize import imresize
import config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
in_size = {2: 14, 3: 11, 4: 10}


def get_result(model, lr, scale, image_mode):
    bic_image = imresize(lr, scale, 'bicubic')
    bic_y, bic_ycbcr = utils.preprocess(bic_image, device, image_mode)
    lr_y, _ = utils.preprocess(lr, device, image_mode)

    with torch.no_grad():
        sr = model(lr_y)
    if scale != 3:
        sr = sr[..., 1:, 1:]
    sr = sr + bic_y
    sr = sr.clamp(0.0, 1.0)
    sr = sr.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    if image_mode == 'L':
        sr = np.clip(sr, 0.0, 255.0).astype(np.uint8)  # chw -> hwc
    else:
        bic_ycbcr = bic_ycbcr.mul(255.0).cpu().numpy().squeeze(0).transpose([1, 2, 0])
        sr = np.array([sr, bic_ycbcr[..., 1], bic_ycbcr[..., 2]]).transpose([1, 2, 0])  # chw -> hwc
        sr = np.clip(utils.ycbcr2rgb(sr), 0.0, 255.0).astype(np.uint8)
    return sr


def run(config_index):
    scale = int(config.models_scale[config_index])
    out_size = scale * in_size[scale]
    model = N2_10_4(scale, in_size[scale], out_size)
    from common.test_template import run
    run(config_index, model, get_result)