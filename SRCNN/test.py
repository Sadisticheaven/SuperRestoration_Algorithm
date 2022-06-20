import numpy as np
import torch
import common.utils as utils
from SRCNN.arch import SRCNN
from common.imresize import imresize

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_result(model, lr, scale, image_mode):
    bic_image = imresize(lr, scale, 'bicubic')
    bic_y, bic_ycbcr = utils.preprocess(bic_image, device, image_mode)

    with torch.no_grad():
        SR = model(bic_y).clamp(0.0, 1.0)
    SR = SR.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    if image_mode == 'L':
        sr = np.clip(SR, 0.0, 255.0).astype(np.uint8)  # chw -> hwc
    else:
        bic_ycbcr = bic_ycbcr.mul(255.0).cpu().numpy().squeeze(0).transpose([1, 2, 0])
        sr = np.array([SR, bic_ycbcr[..., 1], bic_ycbcr[..., 2]]).transpose([1, 2, 0])  # chw -> hwc
        sr = np.clip(utils.ycbcr2rgb(sr), 0.0, 255.0).astype(np.uint8)

    return sr


def run(config_index):
    model = SRCNN(padding=True)
    from common.test_template import run
    run(config_index, model, get_result)
