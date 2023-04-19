from dataclasses import dataclass


@dataclass
class HiddenRepresentationConfig:
    name: str
    source_dataset: str
    processed_datasets: list[str]
    cls_representations: list[str]