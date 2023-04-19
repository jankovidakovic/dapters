from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class HiddenRepresentationConfig:
    name: str
    source_dataset: str
    processed_datasets: list[str]
    cls_representations: list[str]


@dataclass
class Domain:
    representations: np.ndarray
    pca_representations: np.ndarray = field(init=False)
    centroid: np.ndarray
    cluster_ids: np.ndarray
    covmat: np.ndarray = field(init=False)

    def apply_pca(self, pca: PCA):
        self.pca_representations = pca.transform(self.representations)


@dataclass
class DomainCollection:
    domains: list[Domain]
    pca_dim: int

    def __post_init__(self):
        all_representations = np.concatenate(
            (domain.representations for domain in self.domains)
        )
        covmat = np.cov(all_representations, rowvar=False)

        # PCA
        pca = PCA(n_components=self.pca_dim)
        pca.fit(all_representations)
        for domain in self.domains:
            domain.apply_pca(pca)
            domain.covmat = covmat
