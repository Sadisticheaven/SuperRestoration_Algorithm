from SwinIR.arch import SwinIR
import config


def run(config_index):
    scale = int(config.models_scale[config_index])
    model_name = config.models_name[config_index]

    if model_name.find('classicalSR') > -1:
        upsampler = 'pixelshuffle'
    else:
        upsampler = 'nearest+conv'

    if model_name[-1] is 'M':
        resi_connection = '1conv'
        depths = [6, 6, 6, 6, 6, 6]
        num_heads = [6, 6, 6, 6, 6, 6]
        embed_dim = 180
    else:
        resi_connection = '3conv'
        depths = [6, 6, 6, 6, 6, 6, 6, 6, 6]
        num_heads = [8, 8, 8, 8, 8, 8, 8, 8, 8]
        embed_dim = 240

    model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                   img_range=1., depths=depths, embed_dim=embed_dim,
                   num_heads=num_heads, mlp_ratio=2, upsampler=upsampler,
                   resi_connection=resi_connection)
    from common.test_template import run
    run(config_index, model)
