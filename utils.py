import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors


def normalize_point_cloud(point_cloud):
    """
    Normalizes a 3D point cloud to fit within a unit sphere and ensure that each dimension
    has zero mean and unit variance.
    
    Args:
        point_cloud (torch.Tensor): A tensor of shape (N, 3) representing a 3D point cloud.
    
    Returns:
        A normalized point cloud as a torch.Tensor of shape (N, 3).
    """
    # Convert the point cloud to a NumPy array
    point_cloud_np = point_cloud.cpu().numpy()

    # Center the point cloud by subtracting the mean along each dimension
    point_cloud_centered = point_cloud_np - np.mean(point_cloud_np, axis=0)

    # Scale the point cloud to fit within a unit sphere
    scale = np.max(np.sqrt(np.sum(point_cloud_centered**2, axis=1)))
    point_cloud_normalized = point_cloud_centered / scale

    # Normalize each dimension to have zero mean and unit variance
    point_cloud_normalized_mean = np.mean(point_cloud_normalized, axis=0)
    point_cloud_normalized_std = np.std(point_cloud_normalized, axis=0)
    point_cloud_normalized = (point_cloud_normalized - point_cloud_normalized_mean) / point_cloud_normalized_std

    # Convert the normalized point cloud back to a PyTorch tensor
    point_cloud_normalized = torch.from_numpy(point_cloud_normalized).to(point_cloud.device)
    return point_cloud_normalized


def frequency_analysis(point_cloud):
    # Compute the Fourier transform of the point cloud
    fourier_transform = torch.fft.fftn(point_cloud)

    # Compute the frequency magnitude and shift the zero frequency component to the center of the spectrum
    freq_magnitude = torch.abs(torch.fft.fftshift(fourier_transform))

    # Compute the frequency coefficients and shift the zero frequency component to the center of the spectrum
    #freq_coefficients = torch.fft.fftshift(fourier_transform)

    return freq_magnitude#, freq_coefficients

def farthest_point_sampling(points, num_centroids):
    """
    Performs farthest point sampling to select a given number of centroids
    from a set of points.

    Args:
        points (torch.Tensor): (N, D) tensor of point cloud coordinates
        num_centroids (int): number of centroids to select

    Returns:
        centroids (torch.Tensor): (num_centroids, D) tensor of centroid coordinates
    """

    N = points.shape[0]
    centroids = torch.zeros((num_centroids, points.shape[1]), device=points.device)
    distances = torch.full((N,), float('inf'), device=points.device)
    torch.manual_seed(0)
    farthest = torch.randint(0, N, (1,), device=points.device).item()
    
    for i in range(num_centroids):
        centroids[i] = points[farthest]
        dists = torch.sqrt(torch.sum((points - centroids[i])**2, dim=-1))
        distances = torch.min(distances, dists)
        farthest = torch.argmax(distances)

    return centroids

def knn_clustering(points, centroids, k=512):
    """
    Performs K nearest neighbor clustering to group the points around each centroid
    into patches.

    Args:
        points (torch.Tensor): (N, D) tensor of point cloud coordinates
        centroids (torch.Tensor): (num_centroids, D) tensor of centroid coordinates
        k (int): number of nearest neighbors to select for each centroid

    Returns:
        patches (list of torch.Tensor): list of length num_centroids containing tensors
                                        of size (k, D) representing the patches around
                                        each centroid
    """

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points.cpu().numpy())
    _, indices = nbrs.kneighbors(centroids.cpu().numpy())
    indices = torch.from_numpy(indices).to(points.device)
    patches = [points[idx] for idx in indices]

    return patches

def compute_patch_differences(points, centroids, k=512):
    """
    Computes the position and color differences between each sub-set of points and
    its corresponding centroid, resulting in the final subset of points to be fed as
    input to the model.

    Args:
        points (torch.Tensor): (N, D) tensor of point cloud coordinates
        centroids (torch.Tensor): (num_centroids, D) tensor of centroid coordinates
        k (int): number of nearest neighbors to select for each centroid

    Returns:
        patch_diffs (torch.Tensor): (num_centroids, k, D) tensor of position and color
                                     differences between each sub-set of points and its
                                     corresponding centroid
    """

    patches = knn_clustering(points, centroids, k=k)
    patch_diffs = torch.stack([patches[i] - centroids[i] for i in range(centroids.shape[0])])
    return patch_diffs


def get_processed_patches_rgb(point_cloud_tensor, rgb_data, patch_size, point_size, input_channel, input_size, add_freq=True):
    
    centroids = farthest_point_sampling(point_cloud_tensor, patch_size)
    patches = knn_clustering(point_cloud_tensor, centroids, k=point_size)
    patches_rgb = knn_clustering(rgb_data, centroids, k=point_size)
    
    if add_freq:
        for i in range(patch_size):
            freq_mag_shifted= frequency_analysis(patches[i])
            patches[i] = torch.cat([patches[i], patches_rgb[i], freq_mag_shifted], dim=1) 
            patches[i] = torch.reshape(patches[i], (input_channel, input_size, input_size))
    else:
        for i in range(patch_size):
            patches[i] = torch.reshape(patches_rgb[i], (input_channel, input_size, input_size))         
    return patches

