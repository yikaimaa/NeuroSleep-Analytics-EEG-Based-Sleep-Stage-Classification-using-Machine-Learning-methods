import warnings
from pathlib import Path

import mne
import numpy as np

# Mapping for sleep stages (Sleep-EDF -> AASM)
# Sleep-EDF: W, 1, 2, 3, 4, R, ?
# AASM: W(0), N1(1), N2(2), N3(3), REM(4)
STAGE_MAPPING = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # Merge 3 and 4 into N3
    "Sleep stage R": 4,
}


def load_sleep_edf_subject(psg_file, hyp_file):
    """Loads a single subject's PSG and Hypnogram files and returns epoched data."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Channels contain different highpass filters.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Channels contain different lowpass filters.*",
            category=RuntimeWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Highpass cutoff frequency .* setting values to 0 and Nyquist.",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
    annot = mne.read_annotations(hyp_file)
    raw.set_annotations(annot, emit_warning=False)

    # Filter 30s epochs and align with hypnogram
    # The Sleep-EDF dataset follows 30s epochs
    events, event_id = mne.events_from_annotations(raw, chunk_duration=30.0, verbose=False)

    # Filter only the valid sleep stages (exclude 'Sleep stage ?' or unknown)
    valid_event_id = {k: v for k, v in event_id.items() if k in STAGE_MAPPING}

    # Pick EEG channels
    raw.pick(["EEG Fpz-Cz", "EEG Pz-Oz"])

    # Create epochs
    tmax = 30.0 - 1.0 / raw.info['sfreq']  # 30s epochs
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        event_id=valid_event_id,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    # Map raw labels to our 0-4 indices
    inv_event_id = {v: k for k, v in valid_event_id.items()}
    y = np.array([STAGE_MAPPING[inv_event_id[id_]] for id_ in epochs.events[:, 2]])

    X = epochs.get_data()  # (n_epochs, n_channels, n_times)

    return X, y, raw.info['sfreq']


def get_all_subjects(data_dir):
    """Finds all PSG/Hypnogram pairs in the directory."""
    data_path = Path(data_dir)
    psg_files = sorted(list(data_path.glob("*PSG.edf")))
    hyp_files = sorted(list(data_path.glob("*Hypnogram.edf")))

    subjects = []
    for psg in psg_files:
        # Match by subject prefix (e.g., SC4001)
        prefix = psg.name[:6]
        hyp = [h for h in hyp_files if h.name.startswith(prefix)]
        if hyp:
            subjects.append((psg, hyp[0]))
    return subjects


if __name__ == "__main__":
    # Test loading
    data_dir = "data/sleep-edf"
    subjects = get_all_subjects(data_dir)
    print(f"Found {len(subjects)} subjects.")

    if subjects:
        psg, hyp = subjects[0]
        X, y, sfreq = load_sleep_edf_subject(psg, hyp)
        print(f"Loaded subject {psg.name}: X={X.shape}, y={y.shape}, sfreq={sfreq}")
        print(f"Unique labels: {np.unique(y, return_counts=True)}")
