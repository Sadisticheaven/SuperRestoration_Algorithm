import argparse
import config
import SRCNN.test
import ESRGAN.test
import UnAI.interplot
from common.gpu import Monitor, proc_exist, lock, unlock


def run_model_on_datasets(model_idx):
    # # 等待gpu资源
    # monitor = Monitor(10)
    # while monitor.gpus[0].memoryUtil > 0.5:
    #     pass
    # monitor.stop()
    model_name = config.models_name[model_idx]
    if model_name == 'SRCNN':
        run = SRCNN.test.run
    elif model_name == 'ESRGAN':
        run = ESRGAN.test.run
    elif model_name == 'RealESRGAN':
        run = ESRGAN.test.run
    elif model_name == 'bicubic':
        run = UnAI.interplot.run
    else:
        print(f'算法{model_name}不存在！')
        return 1
    run(model_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="在给定数据集上运行指定算法。")
    parser.add_argument("--save_path_root", type=str, default="./save_path/", help="运行结果的保存根路径。")
    # parser.add_argument("--gpus", nargs='+', type=str, help="使用的gpu列表")
    parser.add_argument("--models_name", nargs='+', type=str, default=[
        "SRCNN",
        # "RealESRGAN"
    ], help="算法的名称列表。")
    parser.add_argument("--models_scale", nargs='+', type=str, default=["4", "4"], help="算法的放大倍率列表。")
    parser.add_argument("--models_param", nargs='+', type=str, help="算法的参数路径列表。", default=[
        "F:/Document/py27Proj/SuperRestoration/SRCNN/weight_file/SRCNN_x4_lr=e-1_batch=512/x4/best.pth",
        # "F:/Document/py27Proj/SuperRestoration/weight_file/RealESRGAN_x4plus.pth"
    ])
    parser.add_argument("--datasets_name", nargs='+', type=str, default=[
        # "Set5", "Set14"
    ], help="数据集名称列表。")
    parser.add_argument("--datasets_path", nargs='+', type=str, help="数据集路径列表。", default=[
        # "F:/Document/py27Proj/SuperRestoration/datasets/Set5/",
        # "F:/Document/py27Proj/SuperRestoration/datasets/Set14/"
    ])
    parser.add_argument("--real_datasets_name", nargs='+', type=str, default=["2022-03-24_17-45-04"], help="真实数据集名称列表。")
    parser.add_argument("--real_datasets_path", nargs='+', type=str, help="真实数据集路径列表。", default=[
        "F:/Document/user_data/root/realImage/2022-03-24_17-45-04/",
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
