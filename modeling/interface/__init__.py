from .xdecoder_qnn import *
from .seem_v0 import *
from .seem_v1 import *
from .seem_demo import *
from .build import *

def build_decoder(config, *args, **kwargs):
    model_name = config['MODEL']['DECODER']['NAME']

    model_name = model_name + '_qnn'
    if not is_model(model_name):
        raise ValueError(f'Unkown model: {model_name}')

    return model_entrypoints(model_name)(config, *args, **kwargs)