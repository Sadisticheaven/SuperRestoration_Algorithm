import csv

from tqdm import tqdm

import common.utils as utils
import common.metrics as metrics
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def test_metrics(model, dataset, scale, crop, sr_path, gt_path, save_path, metrics_to_calc):
    csv_path = save_path + 'metrics.csv'
    if not os.path.exists(csv_path):
        csv_file = open(csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(('name', 'psnr', 'niqe', 'ssim', 'lpips'))
    else:
        csv_file = open(csv_path, 'a', newline='')
        writer = csv.writer(csv_file)

    sr_lists = os.listdir(sr_path)
    my_metrics = dict()
    for metric in metrics_to_calc:
        my_metrics[metric] = utils.AverageMeter()

    row_name = model + '_on_' + dataset + f'_x{scale}'
    with tqdm(total=len(sr_lists)) as t:
        t.set_description(f"Test Metrics for {row_name}: ")
        for imgName in sr_lists:
            SR = utils.loadIMG_crop(sr_path + imgName, scale)
            GT = utils.loadIMG_crop(gt_path + imgName, scale)
            my_metrics = metrics.calc_metric(SR, GT, my_metrics, crop=crop)
            t.update(1)
    avg_psnr = 0
    avg_niqe = 0
    avg_ssim = 0
    avg_lpips = 0
    if 'psnr' in my_metrics:
        avg_psnr = my_metrics['psnr'].avg
    if 'niqe' in my_metrics:
        avg_niqe = my_metrics['niqe'].avg
    if 'ssim' in my_metrics:
        avg_ssim = my_metrics['ssim'].avg
    if 'lpips' in my_metrics:
        avg_lpips = my_metrics['lpips'].avg

    writer.writerow((row_name, avg_psnr, avg_niqe, avg_ssim, avg_lpips))

