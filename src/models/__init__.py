from src.models.bottleneck_adapters import get_adapter_model
from src.models.transformer_encoder import get_transformer_model
from src.models.utils import maybe_compile, set_device


def setup_model(args):
    # initialize model
    if hasattr(args, "adapters"):
        model = get_adapter_model(args)
    else:
        model = get_transformer_model(args)

    # easy

    model = maybe_compile(model, args)
    model = set_device(model, args)

    return model
