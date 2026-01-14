"""
OpenBCI Raw Wet VS Textile Analysis - Alishba Kaleem
Converted from Jupyter Notebook to Python script
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from pyxdf import load_xdf
import os
import json
import platform
import sys

# Set matplotlib backend
plt.ion()  # Interactive mode

# ==================== Configuration ====================
os_name = platform.system()

if os_name == "Windows":
    print("Running on Windows.")
    data_dir = r""
elif os_name == "Darwin":
    print("Running on macOS.")
    data_dir = r"/Users/yuan/Library/Application Support/Mountain Duck/Volumes.noindex/Access Pathways - BCIs.localized/myant-eeg-eog-design_prism/2024 - Yuan Dou/Data/0409_25_Yuan/"

# ==================== Helper Functions ====================

def get_fullcap_eeg_raw(file_name):
    """Load full-cap EEG data from XDF file."""
    file_path = data_dir + file_name
    data, header = load_xdf(file_path)

    # Get EEG stream
    eeg_streams = [stream for stream in data if stream["info"]["type"][0] == "EEG"]
    if not eeg_streams:
        raise RuntimeError(f"No EEG stream found in {file_name}")
    
    eeg_stream = eeg_streams[0]
    print(f"Using EEG stream: {eeg_stream['info']['name'][0]}")

    # Extract data
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    full_data = np.array(eeg_stream["time_series"]).T / 1e6  # shape: (n_channels, n_samples)

    # Get channel info and take first two
    ch_info = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    ch_names = [ch["label"][0] for ch in ch_info[:2]]
    ch_types = ["eeg", "eeg"]
    eeg_data = full_data[:2, :]  # Only first 2 channels

    # Create MNE Raw object
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    mne.set_log_level("WARNING")

    return raw


def mindset_get_eegraw(file_name):
    """Load Mindset EEG data with markers."""
    file_path = data_dir + file_name
    data, header = load_xdf(file_path)

    eeg_stream = None
    marker_onsets = []
    marker_descriptions = []

    for stream in data:
        stype = stream["info"]["type"][0]
        if stype == "EEG":
            eeg_stream = stream
        elif stype == "Marker":
            marker_onsets = stream["time_stamps"]
            marker_descriptions = [str(mark[0]) for mark in stream["time_series"]]

    if eeg_stream is None:
        raise ValueError("No EEG stream found in the file.")

    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    eeg_data = np.array(eeg_stream["time_series"]).T / 1e6  # Convert µV → V
    ch_labels = [ch["label"][0] for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]

    # Keep only Fp1 and Fz
    channel_indices = [ch_labels.index("Fp1"), ch_labels.index("Fz")]
    selected_data = eeg_data[channel_indices, :]
    ch_names = ["Raw Wet", "Textile"]
    ch_types = ["eeg", "eeg"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    eeg_raw = mne.io.RawArray(selected_data, info, verbose=False)
    mne.set_log_level("WARNING")

    # Attach annotations (markers) if present
    if len(marker_onsets) > 0 and len(marker_descriptions) > 0:
        eeg_start_time = eeg_stream["time_stamps"][0]
        adjusted_onsets = np.array(marker_onsets) - eeg_start_time
        annotations = mne.Annotations(
            onset=adjusted_onsets,
            duration=[0] * len(adjusted_onsets),
            description=marker_descriptions,
        )
        eeg_raw.set_annotations(annotations)

    return eeg_raw


def mindset_get_epochs(file_name, tmin, tmax, target_marker=None):
    """Extract epochs from Mindset EEG data based on markers."""
    raw = mindset_get_eegraw(file_name)

    file_path = os.path.join(data_dir, file_name)
    data, header = load_xdf(file_path)

    marker_streams = [stream for stream in data if stream["info"]["type"][0] == "Marker"]
    if marker_streams:
        marker_stream = marker_streams[0]
        markers = marker_stream["time_series"]
        marker_times = marker_stream["time_stamps"]

        onsets = []
        durations = []
        descriptions = []

        t0 = marker_times[0]
        for marker_str, ts in zip(markers, marker_times):
            try:
                marker_info = json.loads(marker_str[0])
            except Exception as e:
                print("Error parsing marker:", marker_str, e)
                continue
            marker_label = marker_info.get("status", "Unknown")
            
            if (target_marker is None) or (target_marker.lower() in marker_label.lower()):
                onset = ts - t0
                onsets.append(onset)
                durations.append(0)
                descriptions.append(marker_label)

        if len(onsets) > 0:
            annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
            raw.set_annotations(annotations)
            print("Markers extracted:")
            for onset, desc in zip(onsets, descriptions):
                print("At {:.3f} s: {}".format(onset, desc))
    else:
        print("No Marker stream found in file:", file_name)

    events, event_id = mne.events_from_annotations(raw)
    print(f"In file {file_name}, found event IDs: {event_id}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=(0, 0),
        preload=True,
    )
    return epochs


# ==================== Main Analysis Functions ====================

def analyze_blinks():
    """Analyze eye blinks data."""
    print("\n=== ANALYZING EYE BLINKS ===\n")
    
    blink_files = [
        r"eye_blinks_1-2_1.xdf",
        r"eye_blinks_1-2_2.xdf",
        r"eye_blinks_1-2_3.xdf",
    ]
    
    # Load and concatenate raw files
    raw_list = [get_fullcap_eeg_raw(fname) for fname in blink_files]
    concat_raw = mne.concatenate_raws(raw_list)
    
    # Apply filtering
    filtered_eeg = concat_raw.copy().filter(l_freq=1, h_freq=40, method="fir", verbose=False)
    filtered_eeg = filtered_eeg.notch_filter(freqs=[60], verbose=False)
    
    # Compute manual y-axis scale
    max_uV = np.max(np.abs(filtered_eeg.get_data() * 1e6))
    scale_limit_V = max_uV * 1e-6
    
    # Save filtered data
    filtered_eeg.save("filtered_blinks_raw.fif", overwrite=True)
    print("Saved: filtered_blinks_raw.fif")
    
    return filtered_eeg, scale_limit_V


def analyze_eyes_open_closed():
    """Analyze eyes open vs closed data."""
    print("\n=== ANALYZING EYES OPEN VS CLOSED ===\n")
    
    closed_files = [
        f"sourcedata/sub-P001_ses-S001_task-MI+Eyes Closed_run-{i:03d}.xdf"
        for i in range(1, 11)
    ]
    open_files = [
        f"sourcedata/sub-P001_ses-S001_task-MI+Neutral_run-{i:03d}.xdf"
        for i in range(1, 11)
    ]
    
    # Process first file from each condition
    closed_eeg_raw = mindset_get_eegraw(closed_files[0])
    closed_filtered_eeg = closed_eeg_raw.copy().filter(l_freq=1, h_freq=50, method="fir", verbose=False)
    closed_filtered_eeg = closed_filtered_eeg.notch_filter(freqs=[60], verbose=False)
    
    open_eeg_raw = mindset_get_eegraw(open_files[0])
    open_filtered_eeg = open_eeg_raw.copy().filter(l_freq=1, h_freq=50, method="fir", verbose=False)
    
    print("Eyes open/closed analysis complete")
    
    return closed_filtered_eeg, open_filtered_eeg


def main():
    """Main execution function."""
    print("Starting OpenBCI Analysis Pipeline...")
    print(f"Data directory: {data_dir}")
    
    # Run blink analysis
    try:
        filtered_eeg, scale_limit_V = analyze_blinks()
        print("✓ Blink analysis completed")
    except Exception as e:
        print(f"✗ Blink analysis failed: {e}")
    
    # Run eyes open/closed analysis
    try:
        closed_filtered, open_filtered = analyze_eyes_open_closed()
        print("✓ Eyes open/closed analysis completed")
    except Exception as e:
        print(f"✗ Eyes open/closed analysis failed: {e}")
    
    print("\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()