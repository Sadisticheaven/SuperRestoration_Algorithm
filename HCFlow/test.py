import numpy as np
import torch
import common.test_template
from HCFlow.HCFlow_SR_model import HCFlowSRModel
import os
import config
import SRFlow.options as options

heat = 0.5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_result(model, lr, scale, image_mode):
    lr = lr.astype(np.float32).transpose([2, 0, 1])  # hwc -> chw
    lr /= 255.
    lr = torch.from_numpy(lr).to(device).unsqueeze(0)

    data = {'LQ': lr}
    model.feed_data(data, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    sr = visuals['SR', heat, 0]

    sr = sr.mul(255.0).cpu().numpy()
    sr = np.clip(sr, 0.0, 255.0).transpose([1, 2, 0])
    # GPU tensor -> CPU tensor -> numpy
    sr = np.array(sr).astype(np.uint8)
    return sr


def run(config_index):
    scale = int(config.models_scale[config_index])
    model_name = config.models_name[config_index]
    weight_file = config.models_param[config_index]
    if not os.path.exists(weight_file):
        print(f'Weight file not exist!:\n{weight_file}\n')
        return 0
    for dataset_path in config.datasets_path:
        if not os.path.exists(dataset_path):
            print(f"Image file not exist!:\n{dataset_path}\n")
            return 0

    opt = options.parse('D:/Document/PycharmProj/SuperRestoration_Algorithm/HCFlow/test_SR_DF2K_4X_HCFlow.yml', is_train=False)
    opt['gpu_ids'] = '0'
    opt = options.dict_to_nonedict(opt)

    model = HCFlowSRModel(opt, [heat])
    checkpoint = torch.load(weight_file)
    model.netG.load_state_dict(checkpoint)

    common.test_template.scale = scale
    common.test_template.model_name = model_name
    common.test_template.root_outputs_dir = config.save_path_root + model_name + '/' + f'x{scale}' + '/'

    if config.datasets_name:
        from common.test_template import run_on_reference
        run_on_reference(model, get_result)
    if config.real_datasets_name:
        # UnAI.interplot.run_on_no_reference(scale, 'bicubic')
        from common.test_template import run_on_no_reference
        run_on_no_reference(model, get_result)
