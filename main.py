import argparse
import config
from common.gpu import Monitor, proc_exist, lock, unlock


def run_model_on_datasets(model_idx):
    # # 等待gpu资源
    # monitor = Monitor(10)
    # while monitor.gpus[0].memoryUtil > 0.5:
    #     pass
    # monitor.stop()
    model_name = config.models_name[model_idx]
    if model_name == 'SRCNN':
        import SRCNN.test as test
    elif model_name == 'FSRCNN':
        import FSRCNN.test as test
    elif model_name == 'SRGAN':
        import SRGAN.test as test
    elif model_name == 'BSRGAN':
        import BSRGAN.test as test
    elif model_name.find('ESRGAN') > -1:
        import ESRGAN.test as test
    elif model_name == 'HCFlow':
        import HCFlow.test as test
    elif model_name == 'SRFlow':
        import SRFlow.test as test
    elif model_name.find('SwinIR') > -1:
        import SwinIR.test as test
    elif model_name == 'bicubic':
        import UnAI.interplot as test
    else:
        print(f'算法{model_name}不存在！')
        return 1
    test.run(model_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="在给定数据集上运行指定算法。")
    parser.add_argument("--save_path_root", type=str, default="./save_path/", help="运行结果的保存根路径。")
    # parser.add_argument("--gpus", nargs='+', type=str, help="使用的gpu列表")
    parser.add_argument("--models_name", nargs='+', type=str, default=[
        # "SRCNN",
        "HCFlow"
    ], help="算法的名称列表。")
    parser.add_argument("--models_scale", nargs='+', type=str, default=["4", "4"], help="算法的放大倍率列表。")
    parser.add_argument("--models_param", nargs='+', type=str, help="算法的参数路径列表。", default=[
        # "./SRWeightFiles/SRCNNx4.pth",
        "./SRWeightFiles/SR_DF2K_X4_HCFlow++.pth"
    ])
    parser.add_argument("--datasets_name", nargs='+', type=str, default=[
        # "Set5", "Set14"
    ], help="数据集名称列表。")
    parser.add_argument("--datasets_path", nargs='+', type=str, help="数据集路径列表。", default=[
        # "F:/Document/py27Proj/SuperRestoration/datasets/Set5/",
        # "F:/Document/py27Proj/SuperRestoration/datasets/Set14/"
    ])
    parser.add_argument("--real_datasets_name", nargs='+', type=str, default=["2022-06-20_17-14-37"], help="真实数据集名称列表。")
    parser.add_argument("--real_datasets_path", nargs='+', type=str, help="真实数据集路径列表。", default=[
        # "F:/Document/user_data/root/realImage/2022-03-24_17-45-04/",
        "D:/Document/user_data/root/realImage/2022-06-20_17-14-37/",
    ])
    # parser.add_argument("--real_datasets_name", nargs='+', type=str, default=[], help="真实数据集名称列表。")
    # parser.add_argument("--real_datasets_path", nargs='+', type=str, help="真实数据集路径列表。", default=[])
    lock()
    args = parser.parse_args()
    config.save_path_root = args.save_path_root
    config.models_name = args.models_name
    config.models_scale = args.models_scale
    config.models_param = args.models_param
    config.datasets_name = args.datasets_name
    config.datasets_path = args.datasets_path
    config.real_datasets_name = args.real_datasets_name
    config.real_datasets_path = args.real_datasets_path

    if config.real_datasets_name:
        config.models_name.append("bicubic")
        config.models_param.append("")
        config.models_scale.append(config.models_scale[0])

    for idx, model_name in enumerate(config.models_name):
        print(model_name)
        try:
            run_model_on_datasets(model_idx=idx)
        except Exception as e:
            unlock()
            import traceback
            traceback.print_exc(e)
        else:
            unlock()
    print("Finish.")
