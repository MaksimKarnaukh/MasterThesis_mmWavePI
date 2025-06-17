"""
File containing all data preprocessing classes.
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self, target_dim):
        """

        :param data: Input data [num_samples, time_steps, subcarriers].
        :param target_dim: Dimensionality after PCA reduction.
        """

        self.target_dim = target_dim

    def __call__(self, data):
        """
        Preprocess the data with PCA.

        :param data: Input data [num_samples, time_steps, subcarriers].
        :param target_dim: Dimensionality after PCA reduction.
        :return: Preprocessed data [num_samples, target_dim].
        """

        print(f"Original data shape: {data.shape}")
        flattened_data = data.reshape(data.shape[0], -1)
        print(f"Flattened data shape: {flattened_data.shape}")

        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(flattened_data)

        pca = PCA(n_components=self.target_dim)
        reduced_data = pca.fit_transform(standardized_data)

        print(f"Reduced data to shape: {reduced_data.shape}")
        return reduced_data
