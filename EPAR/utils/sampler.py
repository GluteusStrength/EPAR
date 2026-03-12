import torch
import abc
import numpy as np
from typing import Union

class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage <= 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)

class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features): # Reduce the dimensionality of features
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features), torch.from_numpy(sample_indices)
    
    @staticmethod
    def _compute_combined_distance(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor, euclidean_weight: float = 1.0, cosine_weight: float = 1.0
    ) -> torch.Tensor:
        """
        Computes a combined distance (Euclidean * Cosine) between points.
        This function is memory-efficient for a large matrix_a and small matrix_b.
        """
        # Step 1: Compute Euclidean distance
        a_times_a = torch.sum(matrix_a * matrix_a, dim=1, keepdim=True)
        b_times_b = torch.sum(matrix_b * matrix_b, dim=1, keepdim=True).T
        a_times_b = matrix_a.mm(matrix_b.T)
        euclidean_dist = (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

        # Step 2: Compute Cosine distance (1 - similarity)
        # Normalize the vectors to get their direction
        a_norm = torch.nn.functional.normalize(matrix_a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(matrix_b, p=2, dim=1)
        # Calculate the dot product (cosine similarity)
        cosine_sim = a_norm.mm(b_norm.T)
        cosine_dist = torch.exp(1.0 - cosine_sim)
        
        # Step 3: Combine the distances by multiplication (Element-wise)
        combined_distance = euclidean_dist * cosine_dist
        
        return combined_distance

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes batchwise Euclidean distances using PyTorch.
        Calculates the pairwise distances (Euclidean) between two matrices
        """
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """
        Runs iterative greedy coreset selection.
        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        # distance_matrix = self._compute_combined_distance(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        # features_dim = features.shape[-1]
        # if features_dim == 1152:
        #     num_coreset_samples = 3136
        # else:
        #     num_coreset_samples = 784

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)

class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        num_coreset_samples: int = None,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        self.num_coreset_samples = num_coreset_samples
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        # approximate_distance_matrix = self._compute_combined_distance(
        #     features, features[start_points]
        # )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        if self.num_coreset_samples is None:
            num_coreset_samples = int(len(features) * self.percentage)
        else:
            num_coreset_samples = self.num_coreset_samples
        with torch.no_grad():
            for _ in range(num_coreset_samples):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]
                )
                # coreset_select_distance = self._compute_combined_distance(
                #     features, features[select_idx : select_idx + 1]
                # )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)

class MahalanobisGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        super().__init__(percentage, device, dimension_to_project_features_to)
        self.inv_cov_matrix = None

    def _compute_mahalanobis_distance(
        self, matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes Mahalanobis distances using the pre-computed inverse covariance matrix."""
        
        diff_matrix = matrix_a.unsqueeze(1) - matrix_b.unsqueeze(0)
        
        term_1 = torch.einsum('ijk,kl->ijl', diff_matrix, self.inv_cov_matrix)
        
        mahalanobis_sq_dist = torch.einsum('ijl,ijk->ij', term_1, diff_matrix)
        
        return mahalanobis_sq_dist.clamp(min=0).sqrt()
    
    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection with Mahalanobis distance."""
        
        features_centered = features - features.mean(dim=0)
        cov_matrix = torch.matmul(features_centered.T, features_centered) / (len(features) - 1)
        self.inv_cov_matrix = torch.inverse(cov_matrix)
        
        distance_matrix = self._compute_mahalanobis_distance(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)

# This is the corrected class to use for your large dataset.
# The original code for this class was fundamentally flawed as it tried to compute
# an N x N Mahalanobis distance matrix.
class ApproximateMahalanobisGreedyCoresetSampler(ApproximateGreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        super().__init__(
            percentage, device, number_of_starting_points, dimension_to_project_features_to
        )
        self.inv_cov_matrix = None

    def _compute_mahalanobis_distance(
        self, matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Mahalanobis distance between points in `matrix_a` and `matrix_b`
        using the pre-computed inverse covariance matrix.

        This implementation is memory-efficient for large `matrix_a` and small `matrix_b`.
        """
        if self.inv_cov_matrix is None:
            raise ValueError("Inverse covariance matrix not computed yet.")

        # Reshape matrix_b to [1, M, D] for broadcasting
        # Where M is the number of points in matrix_b (e.g., num_starting_points or 1)
        b_unsqueezed = matrix_b.unsqueeze(0)
        
        # Calculate Mahalanobis distance efficiently without a huge intermediate tensor
        # This is the key change to fix the memory error
        diff_matrix = matrix_a.unsqueeze(1) - b_unsqueezed
        
        # Use einsum to compute the distance.
        mahalanobis_sq_dist = torch.einsum('ijk,kl,ijl->ij', diff_matrix, self.inv_cov_matrix, diff_matrix)

        return mahalanobis_sq_dist.clamp(min=0).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection with Mahalanobis distance."""

        if self.inv_cov_matrix is None:
            features_centered = features - features.mean(dim=0)
            cov_matrix = torch.matmul(features_centered.T, features_centered) / (len(features) - 1)
            # Add a small regularization term to avoid singularity
            cov_matrix += 1e-6 * torch.eye(cov_matrix.shape[0], device=self.device)
            self.inv_cov_matrix = torch.inverse(cov_matrix)

        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_mahalanobis_distance(
            features, features[start_points]
        )
        
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in range(num_coreset_samples):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                
                coreset_select_distance = self._compute_mahalanobis_distance(
                    features, features[select_idx].unsqueeze(0)
                )

                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)