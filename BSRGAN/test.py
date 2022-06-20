from BSRGAN.arch import RRDBNet as BSRGAN
import config


def run(config_index):
    scale = int(config.models_scale[config_index])
    model = BSRGAN(sf=scale)
    from common.test_template import run
    run(config_index, model)

