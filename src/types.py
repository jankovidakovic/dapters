import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA


logger = logging.getLogger(__name__)


@dataclass
class HiddenRepresentationConfig:
    name: str
    source_dataset: str
    processed_datasets: list[str]
    cls_representations: list[str]
    model_name_or_path: str


@dataclass
class Domain:
    name: str
    representations: np.ndarray
    pca_representations: np.ndarray = field(init=False)
    centroid: np.ndarray = field(init=False)
    cluster_ids: Optional[np.ndarray]  # do I have this? I dont think so -> doesnt matter anyways

    def __post_init__(self):
        self.centroid = np.mean(self.representations, axis=0)

    def apply_pca(self, pca: PCA):
        self.pca_representations = pca.transform(self.representations)


@dataclass
class DomainCollection:
    domains: list[Domain]
    pca_dim: int
    representation_covmat: np.ndarray = field(init=False)
    joint_centroid: np.ndarray = field(init=False)

    def __post_init__(self):
        logger.warning(f"Initializing domain collection of {len(self.domains)} domains.")
        all_representations = np.concatenate(
            [domain.representations for domain in self.domains],
            axis=0
        )
        logger.warning(f"Concatenated domain representations into shape {all_representations.shape}")

        self.representation_covmat = np.cov(all_representations, rowvar=False)
        logger.warning(f"Computed covariance matrix of shape {self.representation_covmat.shape}")

        self.joint_centroid = np.mean(all_representations, axis=0)
        logger.warning(f"Computed joint centroid.")

        # PCA
        pca = PCA(n_components=self.pca_dim)
        pca.fit(all_representations)
        logger.warning(F"PCA with {self.pca_dim} components fit on all representations.")
        logger.warning(f"Total variance explained: {np.sum(pca.explained_variance_ratio_) * 100:.4f}%")
        for domain in self.domains:
            domain.apply_pca(pca)

        logger.warning(f"Applied PCA and set covmat for all domains")

    def __getitem__(self, item):
        return self.domains[item]


    def __len__(self):
        return len(self.domains)


    def __iter__(self):
        yield from self.domains
