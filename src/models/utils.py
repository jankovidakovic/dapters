import logging
import torch

logger = logging.getLogger(__name__)

def maybe_compile(model, args):
    if args.use_torch_compile:
        logger.warning(f"Compiling the model")
        model = torch.compile(model)
        logger.warning(f"Model successfully compiled.")
    return model


def set_device(model, args):
    model = model.to(args.device)
    logger.info(f"Model loaded successfully on device: {model.device}")
    return model