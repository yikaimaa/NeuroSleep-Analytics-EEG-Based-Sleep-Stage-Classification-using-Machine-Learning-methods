import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper

def extract_features_epoch(epoch, sfreq):
    """
    Extracts features for a single 30s epoch (1 channel).
    epoch: (n_samples,)
    sfreq: sampling frequency (100 Hz)
    """
    features = {}
    
    # 1. Time-Domain Features
    features['mean'] = np.mean(epoch)
    features['std'] = np.std(epoch)
    features['skew'] = skew(epoch)
    features['kurtosis'] = kurtosis(epoch)
    features['rms'] = np.sqrt(np.mean(epoch**2))
    features['zero_crossing_rate'] = np.sum(np.diff(np.sign(epoch)) != 0) / len(epoch)
    
    # 2. Hjorth Parameters
    dy = np.diff(epoch)
    ddy = np.diff(dy)
    var_y = np.var(epoch)
    var_dy = np.var(dy)
    var_ddy = np.var(ddy)
    
    # Activity (variance)
    features['hjorth_activity'] = var_y
    # Mobility (sqrt(var(dy)/var(y)))
    features['hjorth_mobility'] = np.sqrt(var_dy / var_y) if var_y != 0 else 0
    # Complexity (Mobility(dy)/Mobility(y))
    mobility_y = features['hjorth_mobility']
    mobility_dy = np.sqrt(var_ddy / var_dy) if var_dy != 0 else 0
    features['hjorth_complexity'] = mobility_dy / mobility_y if mobility_y != 0 else 0
    
    # 3. Frequency-Domain Features (PSD)
    # Using Welch's method for efficiency
    freqs, psd = welch(epoch, fs=sfreq, nperseg=int(sfreq * 2)) # 2s window
    
    bands = {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'sigma': (12.0, 16.0),
        'beta': (16.0, 30.0),
        'gamma': (30.0, 45.0)
    }
    
    total_power = np.trapz(psd, freqs)
    for band_name, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        band_power = np.trapz(psd[idx], freqs[idx])
        features[f'rel_power_{band_name}'] = band_power / total_power if total_power > 0 else 0
        features[f'abs_power_{band_name}'] = band_power
        
    return features

def extract_features_all(X, sfreq):
    """
    X: (n_epochs, n_channels, n_samples)
    """
    n_epochs, n_channels, _ = X.shape
    all_features = []
    
    for i in range(n_epochs):
        epoch_features = {}
        for ch in range(n_channels):
            ch_features = extract_features_epoch(X[i, ch], sfreq)
            # Add channel suffix to keys
            for k, v in ch_features.items():
                epoch_features[f"ch{ch}_{k}"] = v
        all_features.append(epoch_features)
        
    import pandas as pd
    return pd.DataFrame(all_features)
