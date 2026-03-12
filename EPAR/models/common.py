"""
Source code from PatchCore common.py
"""

import copy
import os
import pickle
from typing import List
from typing import Union
from tqdm import tqdm
import cv2

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
# from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from scipy.spatial.distance import pdist

class FaissNNRGB(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4, mode="dist") -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None
        self.mode = mode

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.mode == "dist":
            if self.on_gpu:
                return faiss.GpuIndexFlatL2(
                    faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
                )
            return faiss.IndexFlatL2(dimension)
        if self.mode == "dir":
            if self.on_gpu:
                return faiss.GpuIndexFlatIP(
                    faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
                )
            return faiss.IndexFlatIP(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None: # Save the index to a file
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

class FaissNN3D(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 4, mode="dist") -> None:
        """FAISS Nearest neighbourhood search.

        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None
        self.mode = mode

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.mode == "dist":
            if self.on_gpu:
                return faiss.GpuIndexFlatL2(
                    faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
                )
            return faiss.IndexFlatL2(dimension)
        if self.mode == "dir":
            if self.on_gpu:
                return faiss.GpuIndexFlatIP(
                    faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
                )
            return faiss.IndexFlatIP(dimension)

    def fit(self, features: np.ndarray) -> None:
        """
        Adds features to the FAISS search index.

        Args:
            features: Array of size NxD.
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns distances and indices of nearest neighbour search.

        Args:
            query_features: Features to retrieve.
            index_features: [optional] Index features to search in.
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None: # Save the index to a file
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

class FaissNNMM(object):
    """
    FAISS Nearest neighbourhood search.

    Args:
        on_gpu: If set true, nearest neighbour searches are done on GPU.
        num_workers: Number of workers to use with FAISS for similarity search.
    
    Figure out via Mahalanobis distance
    """

    def __init__(self, on_gpu: bool = False, num_workers: int = 4):
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None
        self.whitening = None   # whitening transform (numpy array)

    def _create_index(self, dimension: int):
        if self.on_gpu:
            res = faiss.StandardGpuResources()
            return faiss.GpuIndexFlatL2(res, dimension, faiss.GpuIndexFlatConfig())
        else:
            return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray):
        """
        features: numpy array [N, D]
        - compute covariance Σ, whitening transform W = Σ^{-1/2}
        - transform all features, and add to Faiss index
        """
        if self.search_index:
            self.reset_index()

        N, D = features.shape
        # compute covariance
        features_centered = features - features.mean(axis=0, keepdims=True)
        cov = np.cov(features_centered, rowvar=False)

        # eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)
        # whitening matrix: U Λ^{-1/2} U^T
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-12)) @ eigvecs.T
        self.whitening = W.astype(np.float32)

        # transform features
        features_w = features @ self.whitening.T

        # build index
        self.search_index = self._create_index(D)
        self.search_index.add(features_w.astype(np.float32))

    def run(self, n_nearest_neighbours: int, query_features: np.ndarray, index_features: np.ndarray = None):
        """
        query_features: numpy array [M, D]
        """
        if self.whitening is None:
            raise ValueError("Whitening not computed. Call fit() first.")

        query_w = query_features @ self.whitening.T

        if index_features is None:
            # search in the fitted index
            return self.search_index.search(query_w.astype(np.float32), n_nearest_neighbours)
        else:
            # search against a separate set of features
            index_w = index_features @ self.whitening.T
            tmp_index = self._create_index(index_w.shape[-1])
            tmp_index.add(index_w.astype(np.float32))
            return tmp_index.search(query_w.astype(np.float32), n_nearest_neighbours)
    
    def _train(self, _index, _features):
        pass

    def save(self, filename: str) -> None: # Save the index to a file
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None

    def reset_index(self):
        if self.search_index is not None:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNNRGB(FaissNNRGB):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)

class ApproximateFaissNN3D(FaissNN3D):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)

class ApproximateFaissNNMM(FaissNNMM):
    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)


class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: list):
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxC
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # NxCxWxH -> NxCWH
        return features.reshape(len(features), -1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=224, gaussian_blur=None):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4
        # self.smoothing = 1
        self.gaussian_blur = gaussian_blur
    
    def refine_with_laplacian_pyramid(self, image, levels=4, gain_factor=1.5):
        gp = [image.astype(np.float32)]
        for i in range(levels - 1):
            gp.append(cv2.pyrDown(gp[i]))

        lp = [gp[levels - 1]]
        for i in range(levels - 1, 0, -1):
            expanded = cv2.pyrUp(gp[i], dstsize=(gp[i-1].shape[1], gp[i-1].shape[0]))
            laplacian_layer = gp[i-1] - expanded
            lp.insert(0, laplacian_layer)
            
        for i in range(levels - 1):
            lp[i] = lp[i] * gain_factor

        reconstructed_map = lp[levels - 1]
        for i in range(levels - 1, 0, -1):
            reconstructed_map = cv2.pyrUp(reconstructed_map, dstsize=(lp[i-1].shape[1], lp[i-1].shape[0]))
            reconstructed_map += lp[i-1]
            
        reconstructed_map = np.clip(reconstructed_map, 0, 1)

        return reconstructed_map

    def convert_to_segmentation(self, patch_scores):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
        
        # apply_gaussian_blur = []
        # if self.gaussian_blur is None:
        #     for patch_score in patch_scores:
        #         gaussian_filter = ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
        #         # laplacian_filter = self.refine_with_laplacian_pyramid(image=patch_score)
        #         apply_gaussian_blur.append(gaussian_filter)
        # else:
        #     apply_gaussian_blur = self.gaussian_blur(patch_scores)
        
        # return apply_gaussian_blur
        return patch_scores

class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


class NearestNeighbourScorer3D(object):
    def __init__(self, n_nearest_neighbours: int, nn_method_l2=None, nn_method_dir=None) -> None:
        """
        Nearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method_l2 = nn_method_l2
        self.nn_method_dir = nn_method_dir
        # L2 Distance Based
        self.imagelevel_nn_l2 = lambda query: self.nn_method_l2.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn_l2 = lambda query, index: self.nn_method_l2.run(1, query, index)
        if self.nn_method_dir is not None:
            # Direction Based
            self.imagelevel_nn_dir = lambda query: self.nn_method_dir.run(
                n_nearest_neighbours, query
            )
            self.pixelwise_nn_dir = lambda query, index: self.nn_method_dir.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.detection_features_norm = F.normalize(torch.from_numpy(self.detection_features), p=2, dim=1)
        self.nn_method_l2.fit(self.detection_features)
        if self.nn_method_dir is not None:
            self.nn_method_dir.fit(self.detection_features_norm.numpy())

    def coreset_update(self, query_features, max_ema_decay=0.99, min_ema_decay=0.95):
        # query_features: [n_samples, n_patches, dim]
        ns, _, _ = query_features.shape
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.numpy()
        for n in tqdm(range(ns)):
            query = query_features[n,:,:]
            query_distances, query_nns = self.nn_method.run(1, query)
            nearest_indices = query_nns[:, 0]
            # min-max normalization on query_distances
            # query_distances = (query_distances - query_distances.min()) / (query_distances.max() - query_distances.min())
            # # linear based ema-decay
            # ema_decay = (min_ema_decay-max_ema_decay) * query_distances + max_ema_decay
            ema_decay = 0.5
            features_to_update = self.detection_features[nearest_indices]
            # Exponential Moving Average based update
            updated_features = ema_decay * features_to_update + (1-ema_decay) * query
            # Replace features with updated one
            self.detection_features[nearest_indices] = updated_features

        return self.detection_features
    
    def coreset_update_every_iter(self, query_features, ema_decay=0.99):
        # query_features: [n_samples, n_patches, dim]
        ns, _, _ = query_features.shape
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.numpy()
        for n in tqdm(range(ns)):
            query = query_features[n,:,:]
            query_distances, query_nns = self.nn_method.run(1, query)
            nearest_indices = query_nns[:, 0]
            features_to_update = self.detection_features[nearest_indices]
            # Exponential Moving Average based update
            updated_features = ema_decay * features_to_update + (1-ema_decay) * query
            # Replace features with updated one
            self.detection_features[nearest_indices] = updated_features
        
        return self.detection_features
    
    def predict(self, query_features, max_val=None, foreground_mask=None):
        query_features = self.feature_merger.merge(query_features) # test case: [n_patches, feature_dim]
        query_features_norm = F.normalize(torch.from_numpy(query_features), p=2, dim=1).numpy()
        query_distances_l2, query_nns_l2 = self.imagelevel_nn_l2(query_features)  # [n_patches, n_nearest_neighbours]
        nearest_features = self.detection_features[query_nns_l2]  # [n_query, k, dim]
        if self.nn_method_dir is not None:
            query_distances_dir, query_nns_dir = self.imagelevel_nn_dir(query_features_norm)  # [n_patches, n_nearest_neighbours]
            anomaly_scores1 = query_distances_l2[:, 0]
            anomaly_scores2 = query_distances_dir[:, 0]
            anomaly_scores2 = 1 - anomaly_scores2
            anomaly_scores2 = torch.softmax(torch.from_numpy(anomaly_scores2), dim=0).numpy()
            # anomaly_scores = anomaly_scores1 * (1 - anomaly_scores2.numpy())
            # anomaly_scores1 = (anomaly_scores1 - anomaly_scores1.mean()) / anomaly_scores1.std()
            # anomaly_scores2 = (anomaly_scores2 - anomaly_scores2.mean()) / anomaly_scores2.std()
            anomaly_scores = anomaly_scores1 * anomaly_scores2
        else:
            anomaly_scores = query_distances_l2[:, 0]
        # Apply foreground mask
        if foreground_mask is not None:
            foreground_distance_score = anomaly_scores[foreground_mask.reshape(-1).numpy()]
        else:
            foreground_distance_score = None
        
        if max_val is not None: # Score Refinement (for Segmentation)
            for i in range(len(anomaly_scores)):
                # amplify the anomaly score
                anomaly_scores[i] = (anomaly_scores[i] * (anomaly_scores[i]/max_val))
        
        anomaly_score = np.max(anomaly_scores)

        return anomaly_score, anomaly_scores, foreground_distance_score, torch.mean(torch.from_numpy(nearest_features), dim=1)



    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )

class NearestNeighbourScorerRGB(object):
    def __init__(self, n_nearest_neighbours: int, nn_method_l2=None, nn_method_dir=None) -> None:
        """
        Nearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method_l2 = nn_method_l2
        self.nn_method_dir = nn_method_dir
        # L2 Distance Based
        self.imagelevel_nn_l2 = lambda query: self.nn_method_l2.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn_l2 = lambda query, index: self.nn_method_l2.run(1, query, index)
        if self.nn_method_dir is not None:
            # Direction Based
            self.imagelevel_nn_dir = lambda query: self.nn_method_dir.run(
                n_nearest_neighbours, query
            )
            self.pixelwise_nn_dir = lambda query, index: self.nn_method_dir.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.detection_features_norm = F.normalize(torch.from_numpy(self.detection_features), p=2, dim=1)
        self.nn_method_l2.fit(self.detection_features)
        if self.nn_method_dir is not None:
            self.nn_method_dir.fit(self.detection_features_norm.numpy())

    def coreset_update(self, query_features, max_ema_decay=0.99, min_ema_decay=0.95):
        # query_features: [n_samples, n_patches, dim]
        ns, _, _ = query_features.shape
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.numpy()
        for n in tqdm(range(ns)):
            query = query_features[n,:,:]
            query_distances, query_nns = self.nn_method.run(1, query)
            nearest_indices = query_nns[:, 0]
            # min-max normalization on query_distances
            # query_distances = (query_distances - query_distances.min()) / (query_distances.max() - query_distances.min())
            # # linear based ema-decay
            # ema_decay = (min_ema_decay-max_ema_decay) * query_distances + max_ema_decay
            ema_decay = 0.5
            features_to_update = self.detection_features[nearest_indices]
            # Exponential Moving Average based update
            updated_features = ema_decay * features_to_update + (1-ema_decay) * query
            # Replace features with updated one
            self.detection_features[nearest_indices] = updated_features

        return self.detection_features
    
    def coreset_update_every_iter(self, query_features, ema_decay=0.99):
        # query_features: [n_samples, n_patches, dim]
        ns, _, _ = query_features.shape
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.numpy()
        for n in tqdm(range(ns)):
            query = query_features[n,:,:]
            query_distances, query_nns = self.nn_method.run(1, query)
            nearest_indices = query_nns[:, 0]
            features_to_update = self.detection_features[nearest_indices]
            # Exponential Moving Average based update
            updated_features = ema_decay * features_to_update + (1-ema_decay) * query
            # Replace features with updated one
            self.detection_features[nearest_indices] = updated_features
        
        return self.detection_features
    
    def predict(self, query_features, max_val=None, foreground_mask=None):
        query_features = self.feature_merger.merge(query_features) # test case: [n_patches, feature_dim]
        query_features_norm = F.normalize(torch.from_numpy(query_features), p=2, dim=1).numpy()
        query_distances_l2, query_nns_l2 = self.imagelevel_nn_l2(query_features)  # [n_patches, n_nearest_neighbours]
        # Retrieve top-k nearest feature vectors
        nearest_features = self.detection_features[query_nns_l2]  # [n_query, k, dim]
        if self.nn_method_dir is not None:
            query_distances_dir, query_nns_dir = self.imagelevel_nn_dir(query_features_norm)  # [n_patches, n_nearest_neighbours]
            anomaly_scores1 = query_distances_l2[:, 0]
            anomaly_scores2 = query_distances_dir[:, 0]
            anomaly_scores2 = 1 - anomaly_scores2
            anomaly_scores2 = torch.softmax(torch.from_numpy(anomaly_scores2), dim=0).numpy()
            # anomaly_scores = anomaly_scores1 * (1 - anomaly_scores2.numpy())
            # anomaly_scores1 = (anomaly_scores1 - anomaly_scores1.mean()) / anomaly_scores1.std()
            # anomaly_scores2 = (anomaly_scores2 - anomaly_scores2.mean()) / anomaly_scores2.std()
            anomaly_scores = anomaly_scores1 * anomaly_scores2
        else:
            anomaly_scores = query_distances_l2[:, 0]
        # Apply foreground mask
        if foreground_mask is not None:
            foreground_distance_score = anomaly_scores[foreground_mask.reshape(-1).numpy()]
        else:
            foreground_distance_score = None
        
        if max_val is not None: # Score Refinement (for Segmentation)
            for i in range(len(anomaly_scores)):
                # amplify the anomaly score
                anomaly_scores[i] = (anomaly_scores[i] * (anomaly_scores[i]/max_val))
        
        anomaly_score = np.max(anomaly_scores)

        return anomaly_score, anomaly_scores, foreground_distance_score, torch.mean(torch.from_numpy(nearest_features), dim=1)


    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )


class NearestNeighbourScorerMM(object):
    def __init__(self, n_nearest_neighbours: int, nn_method=None) -> None:
        """
        Neearest-Neighbourhood Anomaly Scorer class.

        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)


    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

    def coreset_update(self, query_features, max_ema_decay=0.99, min_ema_decay=0.95):
        # query_features: [n_samples, n_patches, dim]
        ns, _, _ = query_features.shape
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.numpy()
        for n in tqdm(range(ns)):
            query = query_features[n,:,:]
            query_distances, query_nns = self.nn_method.run(1, query)
            nearest_indices = query_nns[:, 0]
            # min-max normalization on query_distances
            query_distances = (query_distances - query_distances.min()) / (query_distances.max() - query_distances.min())
            # linear based ema-decay
            ema_decay = (min_ema_decay-max_ema_decay) * query_distances + max_ema_decay
            # ema_decay = 0.5
            features_to_update = self.detection_features[nearest_indices]
            # Exponential Moving Average based update
            updated_features = ema_decay * features_to_update + (1-ema_decay) * query
            # Replace features with updated one
            self.detection_features[nearest_indices] = updated_features
        
        return self.detection_features

    def coreset_update_every_iter(self, query_features, ema_decay=0.99):
        # query_features: [n_samples, n_patches, dim]
        ns, _, _ = query_features.shape
        if isinstance(query_features, torch.Tensor):
            query_features = query_features.numpy()
        for n in tqdm(range(ns)):
            query = query_features[n,:,:]
            query_distances, query_nns = self.nn_method.run(1, query)
            nearest_indices = query_nns[:, 0]
            features_to_update = self.detection_features[nearest_indices]
            # Exponential Moving Average based update
            updated_features = ema_decay * features_to_update + (1-ema_decay) * query
            # Replace features with updated one
            self.detection_features[nearest_indices] = updated_features
        
        return self.detection_features
    def predict(self, query_features, max_val=None, foreground_mask=None):
        query_features = self.feature_merger.merge(query_features) # test case: [n_patches, feature_dim]
        query_distances, query_nns = self.imagelevel_nn(query_features)  # [n_patches, n_nearest_neighbours]
        anomaly_scores = query_distances[:, 0]
        max_scores_index = torch.argmax(torch.from_numpy(anomaly_scores))
        max_scores = torch.index_select(torch.from_numpy(anomaly_scores), 0, max_scores_index)
        max_scores = max_scores - torch.max(max_scores)
        weights = 1 - (torch.max(torch.exp(max_scores)) / torch.sum(torch.exp(max_scores)))
        # Apply foreground mask
        if foreground_mask is not None:
            foreground_distance_score = anomaly_scores[foreground_mask.reshape(-1).numpy()]
        else:
            foreground_distance_score = None
        
        if max_val is not None: # Score Refinement (for Segmentation)
            for i in range(len(anomaly_scores)):
                # amplify the anomaly score
                anomaly_scores[i] = (anomaly_scores[i] * (anomaly_scores[i]/max_val))
        anomaly_score = np.max(anomaly_scores) * weights.numpy()

        return anomaly_score, anomaly_scores, foreground_distance_score
    # def predict(self, query_features, max_val=None, foreground_mask=None):
    #     query_features = self.feature_merger.merge(query_features) # test case: [n_patches, feature_dim]
    #     query_distances, query_nns = self.imagelevel_nn(query_features)  # [n_patches, n_nearest_neighbours]
    #     # initial_scores = query_distances[:, 0]
    #     # # k-nearest neighbours representations
    #     # coreset_neighbors = self.detection_features[query_nns]
    #     # std_dev_per_feature = np.std(coreset_neighbors, axis=1)
    #     # calibration_factors = np.mean(std_dev_per_feature, axis=1) + 1e-6 # 분모 0 방지
    #     # calibrated_scores = initial_scores / calibration_factors
    #     # anomaly_scores = np.mean(query_distances, axis=1)
    #     # anomaly_scores = query_distances[:, 0] / calibrated_scores
    #     anomaly_scores = query_distances[:, 0]
    #     # Apply foreground mask
    #     if foreground_mask is not None:
    #         foreground_distance_score = anomaly_scores[foreground_mask.reshape(-1).numpy()]
    #     else:
    #         foreground_distance_score = None
    #     # Select foreground related features
    #     nn_features = self.detection_features[query_nns.reshape(-1), :].reshape(query_nns.shape[0], query_nns.shape[1], -1)
    #     nn_features_foreground = nn_features[foreground_mask.reshape(-1).numpy(),:,:]
    #     if foreground_distance_score is not None:
    #         max_index = np.argmax(foreground_distance_score)
    #         nearby_features = nn_features_foreground[max_index, :, :]
    #         anomaly_representation = query_features[max_index, :]
    #     else:
    #         max_index = np.argmax(query_distances[:, 0])
    #         nearby_features = anomaly_scores[max_index, :, :]
    #         anomaly_representation = query_features[max_index, :]
        
    #     if max_val is not None: # Score Refinement (for Segmentation)
    #         for i in range(len(anomaly_scores)):
    #             # amplify the anomaly score
    #             anomaly_scores[i] = (anomaly_scores[i] * (anomaly_scores[i]/max_val))
                
    #     return anomaly_scores, query_distances, foreground_distance_score, nearby_features, anomaly_representation
    

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )