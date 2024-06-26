from .counterfactual_cgan import CounterfactualCGAN
from .lungs_cgan import LungsCGAN
from .counterfactual_inpainting_cgan import CounterfactualInpaintingCGAN
from .counterfactual_inpainting_cgan_v2 import CounterfactualInpaintingCGANV2

def build_gan(opt, **kwargs):
    assert 'kind' in opt, 'No architecture type specified in the model configuration'
    if opt.kind == 'lungs_cgan': # has trroubles
        return LungsCGAN(opt=opt, **kwargs)
    elif opt.kind == 'counterfactual_lungs_cgan':
        return CounterfactualCGAN(opt=opt, **kwargs)
    elif opt.kind == 'inpainting_counterfactual_cgan':
        return CounterfactualInpaintingCGAN(opt=opt, **kwargs)
    elif opt.kind == 'inpainting_counterfactual_cgan_v2':
        return CounterfactualInpaintingCGANV2(opt=opt, **kwargs)
    else:
        raise ValueError(f'Invalid architecture type: {opt.kind}')
