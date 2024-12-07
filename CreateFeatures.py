import numpy as np

from mne_features.univariate import (
    compute_mean, compute_variance, compute_std, compute_ptp_amp, compute_skewness, 
    compute_kurtosis, compute_rms, compute_quantile, compute_hurst_exp, 
    compute_app_entropy, compute_decorr_time, compute_pow_freq_bands, 
    compute_hjorth_mobility_spect, compute_hjorth_complexity_spect, compute_hjorth_mobility, 
    compute_hjorth_complexity, compute_higuchi_fd, compute_katz_fd, compute_zero_crossings, 
    compute_line_length, compute_spect_slope, compute_spect_entropy, compute_energy_freq_bands, 
    compute_spect_edge_freq, compute_wavelet_coef_energy, compute_teager_kaiser_energy
)
import warnings

warnings.filterwarnings('ignore')

class CreateFeatures:
    """
    This class is responsible for extracting various features from EEG windows.
    It supports a range of features, and each can be customized based on the window data.
    """

    def __init__(self):
        # List of feature functions that will be used for extraction
        self.features_func = [
            compute_mean, compute_variance, compute_std, compute_ptp_amp, 
            compute_skewness, compute_kurtosis, compute_rms, compute_quantile, 
            compute_hurst_exp, compute_app_entropy, compute_decorr_time, compute_pow_freq_bands, 
            compute_hjorth_mobility_spect, compute_hjorth_complexity_spect, 
            compute_hjorth_mobility, compute_hjorth_complexity, compute_higuchi_fd, 
            compute_katz_fd, compute_zero_crossings, compute_line_length, 
            compute_spect_slope, compute_spect_entropy, compute_energy_freq_bands, 
            compute_spect_edge_freq, compute_wavelet_coef_energy, compute_teager_kaiser_energy
        ]

        # List of features that require the sampling frequency (sfreq)
        self.func_sfreq = [
            'compute_hjorth_complexity_spect', 'compute_decorr_time', 
            'compute_hjorth_mobility_spect', 'compute_spect_slope', 
            'compute_spect_entropy', 'compute_spect_edge_freq'
        ]

        # List of features that require spectral bands
        self.func_bands = ['compute_pow_freq_bands', 'compute_energy_freq_bands']

        # Definition of spectral bands (0.5-4, 4-8, 8-13, 13-30, >30 Hz)
        self.frequency_bands = np.array([0.5, 4., 8., 13., 30.])

        # Total number of features that will be extracted
        self.n_all_feat = 53

    def get_feature(self, window, feature_func):
        """
        Extracts features from a given window using the provided feature function.

        Args:
            window (numpy array): The window of EEG data (time x channels).
            feature_func (function): The function to compute the feature.

        Returns:
            The computed feature(s).
        """
        # Check if the feature requires the sampling frequency (sfreq)
        if feature_func.__name__ in self.func_sfreq:
            return feature_func(sfreq=128, data=window)

        # Special case for the 'compute_higuchi_fd' function, which requires kmax
        elif feature_func.__name__ == 'compute_higuchi_fd':
            return feature_func(data=window, kmax=2)

        # Check if the feature requires spectral bands
        elif feature_func.__name__ in self.func_bands:
            return feature_func(sfreq=128, data=window, freq_bands=self.frequency_bands)

        # Default case: compute the feature without additional parameters
        else:
            return list(feature_func(window))

    def fill_array_from_window(self, window):
        """
        Extracts features from a window and fills the pre-allocated array.

        Args:
            window (numpy array): Transposed EEG window (channels x time).
        """
        # Loop over each feature function to extract the feature
        for feature_func in self.features_func:
            # Extract the feature from the window
            feature_val = self.get_feature(window, feature_func)

            # Determine the number of features returned for each channel
            n_feat = int(len(feature_val) / self.n_channels)

            cont_feat_array = 0

            # Fill the array for each channel with the extracted features
            for chan in range(self.n_channels):
                val_chan = feature_val[cont_feat_array:cont_feat_array + n_feat]
                self.array_data[self.wind_cont][chan][self.cont_col_array:
                                                      self.cont_col_array + n_feat] = val_chan
                cont_feat_array += n_feat

            # Update the column index for the next feature
            self.cont_col_array += n_feat

    def change_dimensions(self):
        """
        Changes the dimensions of the array from window x channel x feature
        to channel x window x feature.

        Returns:
            The reshaped array.
        """
        array_data_reshaped = np.zeros([self.n_channels, self.n_windows, self.n_all_feat])
        
        for n_chan in range(self.array_data.shape[1]):
            for n_wind in range(self.array_data.shape[0]):
                array_data_reshaped[n_chan][n_wind] = self.array_data[n_wind][n_chan]

        return array_data_reshaped

    def get_features(self, batch_data):
        """
        Extracts features from a batch of EEG data (multiple windows).

        Args:
            batch_data (list or numpy array): Batch of EEG windows (n_windows x n_channels x n_time).

        Returns:
            numpy array: The extracted features (n_channels x n_windows x n_all_feat).
        """
        # Determine the number of windows and channels
        if isinstance(batch_data, list):
            self.n_windows = len(batch_data)
            self.n_channels = batch_data[0].shape[1]
        else:
            self.n_windows = batch_data.shape[0]
            self.n_channels = batch_data.shape[1]
        
        self.wind_cont = 0
        # Initialize the feature array (windows x channels x features)
        self.array_data = np.zeros([self.n_windows, self.n_channels, self.n_all_feat])
        self.wind_cont = 0

        # Loop over each window to extract features
        for window in batch_data:
            windowT = window.transpose()  # Transpose to channels x time
            self.cont_col_array = 0
            self.fill_array_from_window(windowT)
            self.wind_cont += 1

        # Reshape the array to channel x window x feature and return
        return self.change_dimensions()
