import importlib


def parse_args(experiment_type: str):
    # dynamically import the module
    module = importlib.import_module(f"src.cli.{experiment_type}")
    parser = module.get_parser()
    return parser.parse_args()