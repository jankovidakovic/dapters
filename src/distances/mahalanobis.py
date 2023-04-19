from scipy.spatial.distance import mahalanobis


def mahalanobis_distance(covmat):
    def apply(source, target):
        return mahalanobis(source, target, covmat)
    return apply
