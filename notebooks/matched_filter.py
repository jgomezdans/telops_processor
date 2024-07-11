"""
Functions for hyperspectral matched filter detection using a local clutter model.

The functions here implement a typical match filter where the measured signal
is assumed to be a linear combination of noise and background plus added clutter.
To account for variability, we first do some clustering on a compressed spectral
space and estimate the background signal per pixel.
"""

__author__ = "Jose Gomez-Dans"

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple
from osgeo import gdal
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


def perform_pca(data: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform PCA on the data cube along the n_bands dimension.

    Args:
    data (np.ndarray): The input data cube with dimensions (n_bands, y, x).
    n_components (int): The number of principal components to retain.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - transformed_data: The transformed data with dimensions (n_components, y * x).
        - explained_variance_ratio: The variance explained by each principal component.
    """
    n_bands, y, x = data.shape
    reshaped_data = data.reshape(n_bands, y * x).T  # Reshape to (y * x, n_bands)

    # Standardize the data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(reshaped_data)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(standardized_data)
    explained_variance_ratio = pca.explained_variance_ratio_

    transformed_data = pca_transformed.T  # Transpose back to (n_components, y * x)

    return transformed_data, explained_variance_ratio


def perform_clustering(transformed_data: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Perform clustering on the transformed data.

    Args:
    transformed_data (np.ndarray): The data after PCA transformation with
            dimensions (n_components, y * x).
    n_clusters (int): The number of clusters for the clustering algorithm.

    Returns:
    np.ndarray: The cluster labels for each point with dimensions (y * x).
    """
    n_components, n_points = transformed_data.shape
    reshaped_data = transformed_data.T  # Reshape to (n_points, n_components)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reshaped_data)
    labels = kmeans.labels_

    return labels


def compute_mean_vector(R: np.ndarray) -> np.ndarray:
    """Compute the mean vector of the radiance vectors."""
    return np.mean(R, axis=0)


def compute_covariance_matrix(R: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Compute the covariance matrix of the radiance vectors."""
    centered_R = R - u
    return np.dot(centered_R.T, centered_R) / R.shape[0]


def compute_inverse_via_svd(C: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute the pseudoinverse of the covariance matrix using SVD with
    thresholding.

    Parameters:
        C (np.ndarray): Covariance matrix.
        threshold (float): Eigenvalue threshold below which eigenvalues are
                    set to zero.

    Returns:
        np.ndarray: Stabilized pseudoinverse of the covariance matrix.
    """
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    # Apply threshold: zero out small eigenvalues or replace with a small value
    S_inv = np.array([1 / s if s > threshold else 0 for s in S])
    S_inv_diag = np.diag(S_inv)
    return np.dot(Vt.T, np.dot(S_inv_diag, U.T))


def compute_filter_vector(C_inv: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the filter vector q."""
    numerator = np.dot(C_inv, b)
    denominator = np.sqrt(np.dot(b.T, numerator))
    return numerator / denominator


def compute_projections(R: np.ndarray, q: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Compute the projections q^T * r_i for each r_i in R, and also
    subtract background vector U"""
    return R @ q - U @ q


def match_filter_image(
    telops_fname: Path | str,
    ref_signal_fname: Path | str,
    n_components: int,
    n_clusters: int,
    retval_all: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply matched filter to an image using TELOPS data and a reference signal.
    The algorithm performs first a PCA to reduce dimensionality, and then uses
    that to do a simple K-means clustering. The user has to provide the
    number of PCA components (usually 10-15 is enough) and the number of
    clusters (trial and error?).

    Args:
        telops_fname (Path | str): The file path or name of the TELOPS file.
        ref_signal_fname (Path | str): The path or name of the reference signal.
        n_components (int): The number of components to keep during PCA.
        n_clusters (int): The number of clusters for performing clustering.
        retval_all (bool): Return internal state that might be useful for
                debugging etc

    Returns:
        np.ndarray: The matched filter image.
        np.ndarray: The clusters map (0 to n_clusters + 1)

    Raises:
        IOError: If the reference spectrum file or TELOPS file doesn't exist.
    """
    ref_signal_fname = Path(ref_signal_fname)
    if not ref_signal_fname.exists():
        raise IOError("Reference spectrum doesn't exist!")
    telops_fname = Path(telops_fname)
    if not telops_fname.exists():
        raise IOError("TELOPS file doesn't exist!")
    ref_signal = np.loadtxt(ref_signal_fname, skiprows=1, delimiter=",")
    wv = ref_signal[:, 0]  # noqa
    b = ref_signal[:, 1]

    g = gdal.Open(telops_fname)
    data = g.ReadAsArray()
    n_bands, ny, nx = data.shape

    # Perform PCA
    transformed_data, explained_variance_ratio = perform_pca(data, n_components)
    print("Explained variance ratio:", explained_variance_ratio.sum())

    # Perform Clustering
    labels = perform_clustering(transformed_data, n_clusters).reshape((ny, nx))
    X = data.reshape((n_bands, -1))
    matched_filter = np.zeros_like(data[0])
    filters = []
    for cluster in range(n_clusters + 1):
        passer = labels.flatten() == cluster
        if passer.sum() > 1:
            R = X[:, passer].T
            u = compute_mean_vector(R)
            C = compute_covariance_matrix(R, u)
            C_inv = compute_inverse_via_svd(C, threshold=1e-6)
            q = compute_filter_vector(C_inv, b)
            filters.append(q)
            projections = compute_projections(R, q, u)
            matched_filter[labels == cluster] = projections
        else:
            matched_filter[labels == cluster] = np.nan

    if not retval_all:
        return matched_filter, labels
    else:
        return (
            matched_filter,
            labels,
            {"wv": wv, "ref_signal": b, "filter": np.array(filters)},
        )


if __name__ == "__main__":
    telops_fname = "2_Bacton_M_1200/2_Bacton_M_1200_Data_20240625_122135513.radiance.sc"
    reference_fname = "CH4_ref_spectra_subsampled.csv"
    matched_filter = match_filter_image(telops_fname, reference_fname, 15, 6)
    plt.imshow(matched_filter)
    plt.colorbar()
