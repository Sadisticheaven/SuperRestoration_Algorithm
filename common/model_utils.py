import torch


def save_checkpoint(model, optimizer, epoch, epoch_losses, epoch_psnr, best_psnr, best_epoch, outputs_dir, csv_writer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
             'loss': epoch_losses.avg, 'psnr': epoch_psnr.avg, 'best_psnr': best_psnr, 'best_epoch': best_epoch}
    torch.save(state, outputs_dir + f'latest.pth')
    csv_writer.writerow((state['epoch'], state['loss'], state['psnr']))

    if epoch_psnr.avg > best_psnr:
        best_epoch = epoch
        best_psnr = epoch_psnr.avg
        state['best_psnr'] = best_psnr
        state['best_epoch'] = best_epoch
        torch.save(state, outputs_dir + 'best.pth')
    return best_epoch, best_psnr
