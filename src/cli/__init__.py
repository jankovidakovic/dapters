import importlib


def parse_args(experiment_type: str, use_adapters: bool = False):
    # dynamically import the module
    module = importlib.import_module(f"src.cli.{experiment_type}")
    parser = module.get_parser(use_adapters=use_adapters)
    return parser.parse_args()