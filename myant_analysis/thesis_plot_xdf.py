"""
Daniel Yu
ECE499Y Thesis
2025-2026

Comprehensive XDF EEG Analysis Tool
Loads XDF EEG recordings and generates PSD, ICA, raw-signal, blink, and alpha analyses using MNE.

*Note*: This is script was designed specifically for data collected for Daniel Yu's ECE499Y thesis project, 
        so it may require adjustments to work with other datasets.

Usage:
    python plot_xdf.py /path/to/your/folder
    python plot_xdf.py /path/to/your/file.xdf --analysis blink
    python plot_xdf.py /path/to/your/file.xdf --analysis alpha
    python plot_xdf.py /path/to/your/folder --analysis blink
    python plot_xdf.py /path/to/your/folder --analysis alpha
    
    Or run interactively and enter path when prompted.
    
Options:
    --analysis blink    Perform blink epoch analysis (ERP, amplitude comparison, etc.)
    --analysis alpha    Analyze alpha band with eyes-open/eyes-closed subfolders
"""

import mne
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from pyxdf import load_xdf
from datetime import datetime

# Enable interactive plotting
plt.ion()


def get_channel_display_name(ch_name):
    """
    Returns a friendly display name for a channel.

    Inputs:
    - ch_name (str): Original channel name.

    Outputs:
    - str: Human-readable display name for the channel.
    """
    channel_mapping = {
        'Fp1': 'Textile Electrode (Fp1)',
        'Fp2': 'Clinical Standard (Fp2)',
        'CH4': 'CH4 (Hansmann)',
        'CH5': 'CH5 (Foltynsky)',
        'Ch4': 'CH4 (Hansmann)',
        'Ch5': 'CH5 (Foltynsky)'
    }
    return channel_mapping.get(ch_name, ch_name)


def setup_output_directory(xdf_file_path, analysis_type=None, subfolder_name=None):
    """
    Create and return output directories for analysis results.

    Inputs:
    - xdf_file_path (str):       Path to the XDF file.
    - analysis_type (str|None):  Analysis type, e.g. 'blink' or 'alpha'.
    - subfolder_name (str|None): Subfolder name for grouping (optional).

    Outputs:
    - tuple: (out_dir (str), main_dir (str)) paths for saving results.
    """
    # Get the XDF filename without extension
    xdf_basename = os.path.splitext(os.path.basename(xdf_file_path))[0]
    
    # Determine main directory based on analysis type
    if analysis_type == 'blink':
        main_dir = 'blink_results'
    elif analysis_type == 'alpha':
        main_dir = 'alpha_results'
    else:
        main_dir = 'xdf_results'
    
    if not os.path.isdir(main_dir):
        os.makedirs(main_dir)
    
    # For alpha analysis, create subfolder directly under alpha_results
    if analysis_type == 'alpha' and subfolder_name:
        sub_dir = os.path.join(main_dir, subfolder_name)
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
            print(f"Created subfolder directory: {sub_dir}")
        out_dir = os.path.join(sub_dir, xdf_basename)
    else:
        # Create subfolder for this specific file directly under main_dir
        out_dir = os.path.join(main_dir, xdf_basename)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")
    else:
        print(f"Using existing directory: {out_dir}")
    
    return out_dir, main_dir


def load_xdf_to_mne(xdf_file_path, cut_blink=False, cut_start_seconds=0):
    """
    Load an XDF file and return an MNE Raw object.

    Inputs:
    - xdf_file_path (str):      Path to the XDF file.
    - cut_blink (bool):         If True, cut 10s from start and end to remove blink artifacts.
    - cut_start_seconds (int):  Seconds to cut from start of recording.

    Outputs:
    - mne.io.RawArray: MNE Raw object containing EEG data and annotations.
    """
    print(f"\nLoading XDF file: {xdf_file_path}")

    # Load XDF file
    streams, header = load_xdf(xdf_file_path)

    # Find EEG stream
    eeg_stream = None
    marker_stream = None

    print(f"\nFound {len(streams)} streams:")
    for i, stream in enumerate(streams):
        stream_name = stream['info']['name'][0]
        stream_type = stream['info']['type'][0]
        print(f"  [{i}] {stream_name} (Type: {stream_type})")

        if stream_type == 'EEG' and eeg_stream is None:
            eeg_stream = stream
            print(f"      → Selected as EEG stream")
        elif stream_type == 'Marker':
            marker_stream = stream

    if eeg_stream is None:
        raise ValueError("No EEG stream found in XDF file")

    # Extract EEG data and metadata
    print(f"\nProcessing EEG stream: {eeg_stream['info']['name'][0]}")

    # Get sampling frequency
    if 'nominal_srate' in eeg_stream['info']:
        sfreq = float(eeg_stream['info']['nominal_srate'][0])
    elif 'effective_srate' in eeg_stream['info']:
        sfreq = float(eeg_stream['info']['effective_srate'])
    else:
        raise ValueError("Could not determine sampling rate")

    print(f"  Sampling rate: {sfreq} Hz")

    # Get channel names
    try:
        desc = eeg_stream['info'].get('desc')
        ch_info = None
        if isinstance(desc, list) and len(desc) > 0:
            first_desc = desc[0]
            if isinstance(first_desc, dict):
                channels = first_desc.get('channels')
                if isinstance(channels, list) and len(channels) > 0:
                    first_channels = channels[0]
                    if isinstance(first_channels, dict):
                        ch_info = first_channels.get('channel')

        if ch_info:
            if isinstance(ch_info, list):
                ch_names = [ch['label'][0] for ch in ch_info if isinstance(ch, dict) and 'label' in ch and ch['label']]
            else:
                ch_names = [f'Ch{i+1}' for i in range(eeg_stream['time_series'].shape[1])]
        else:
            raise ValueError("Missing channel metadata")
    except (KeyError, IndexError, TypeError, ValueError):
        # Fallback: generate channel names
        n_channels = eeg_stream['time_series'].shape[1]
        ch_names = [f'Ch{i+1}' for i in range(n_channels)]
        print(f"  Warning: Could not extract channel names, using default names")

    print(f"  Number of channels: {len(ch_names)}")
    print(f"  Channel names: {', '.join(ch_names)}")

    # Get EEG data (transpose to MNE format: channels × time)
    eeg_data = np.array(eeg_stream['time_series']).T

    # Convert to volts (common scaling)
    # Try to detect if data is in microvolts
    data_range = np.max(np.abs(eeg_data))
    if data_range > 1e-3:  # Likely in microvolts
        print(f"  Data appears to be in microvolts, converting to volts")
        eeg_data = eeg_data / 1e6

    print(f"  Data shape: {eeg_data.shape} (channels × samples)")
    print(f"  Duration: {eeg_data.shape[1] / sfreq:.2f} seconds")

    # Cut the first N seconds if requested
    if cut_start_seconds > 0:
        start_sample = int(sfreq * cut_start_seconds)
        if eeg_data.shape[1] <= start_sample:
            raise ValueError("File is too short to cut the requested start duration.")
        eeg_data = eeg_data[:, start_sample:]
        print(f"  Cut first {cut_start_seconds} seconds. New duration: {eeg_data.shape[1] / sfreq:.2f} seconds")

    # Cut the first and last 10 seconds if cut_blink is True
    if cut_blink:
        start_sample = int(sfreq * 10)  # 10 seconds in samples
        end_sample = eeg_data.shape[1] - start_sample
        if end_sample <= start_sample:
            raise ValueError("File is too short to cut 10 seconds from both ends.")
        eeg_data = eeg_data[:, start_sample:end_sample]
        print(f"  Cut first and last 10 seconds. New duration: {eeg_data.shape[1] / sfreq:.2f} seconds")

    # Create MNE Info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create Raw object
    raw = mne.io.RawArray(eeg_data, info, verbose=False)

    # Try to set montage for standard 10-20 electrode positions
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        print(f"  Applied standard 10-20 montage")
    except Exception as e:
        print(f"  Warning: Could not set montage: {e}")

    # Add markers as annotations if available
    if marker_stream is not None:
        print(f"\nProcessing markers...")
        marker_times = marker_stream['time_stamps']
        marker_descriptions = [str(m[0]) for m in marker_stream['time_series']]

        # Align marker times to EEG stream
        eeg_start_time = eeg_stream['time_stamps'][0]
        adjusted_onsets = marker_times - eeg_start_time

        annotations = mne.Annotations(
            onset=adjusted_onsets,
            duration=[0.0] * len(adjusted_onsets),
            description=marker_descriptions
        )
        raw.set_annotations(annotations)
        print(f"  Added {len(marker_descriptions)} markers")

    return raw


def print_info(raw):
    """
    Prints a concise summary of recording metadata and annotations.

    Inputs:
    - raw (mne.io.Raw): Loaded MNE Raw object.

    Outputs:
    - None: Prints information to stdout.
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print(f"\nChannels: {', '.join(raw.ch_names)}")
    
    if raw.annotations:
        print(f"\nAnnotations: {len(raw.annotations)} events")
        unique_descriptions = set(raw.annotations.description)
        print(f"Event types: {', '.join(unique_descriptions)}")
        for onset, description in zip(raw.annotations.onset, raw.annotations.description):
            print(f"Event: {description}, Timestamp: {onset:.2f} seconds")
    else:
        print("\nNo annotations/markers found")
    
    # Check for montage
    if raw.get_montage() is not None:
        montage = raw.get_montage()
        # Safely get montage name (handles missing 'kind' attribute)
        montage_name = getattr(montage, 'kind', 'Custom montage')
        print(f"\nMontage: {montage_name}")
        
        n_positioned = sum([1 for ch in raw.info['chs'] if ch['loc'][0] != 0])
        print(f"Positioned channels: {n_positioned}/{len(raw.ch_names)}")
        
        # Specifically check if Fp1 and Fp2 are positioned
        fp1_idx = raw.ch_names.index('Fp1') if 'Fp1' in raw.ch_names else None
        fp2_idx = raw.ch_names.index('Fp2') if 'Fp2' in raw.ch_names else None
        
        if fp1_idx is not None:
            fp1_pos = raw.info['chs'][fp1_idx]['loc'][:3]
            if fp1_pos[0] != 0:
                print(f"  Fp1 position: [{fp1_pos[0]:.3f}, {fp1_pos[1]:.3f}, {fp1_pos[2]:.3f}] ✓")
            else:
                print(f"  Fp1 position: Not set ✗")
        
        if fp2_idx is not None:
            fp2_pos = raw.info['chs'][fp2_idx]['loc'][:3]
            if fp2_pos[0] != 0:
                print(f"  Fp2 position: [{fp2_pos[0]:.3f}, {fp2_pos[1]:.3f}, {fp2_pos[2]:.3f}] ✓")
            else:
                print(f"  Fp2 position: Not set ✗")
    else:
        print("\nNo montage set (electrode positions unknown)")
    
    print("="*60 + "\n")


def run_ica_analysis(raw, out_dir, n_components=2):
    """
    Fits an ICA on the provided raw data and return the fitted ICA object.

    Inputs:
    - raw (mne.io.Raw):     Raw MNE object to analyze.
    - out_dir (str):        Output directory (used for context/logging).
    - n_components (int):   Number of ICA components to fit.

    Outputs:
    - mne.preprocessing.ICA: Fitted ICA object.
    """
    print(f"\nRunning ICA analysis with {n_components} components...")
    
    # Make a copy for ICA (will be filtered)
    raw_ica = raw.copy()
    raw_ica.filter(l_freq=1.0, h_freq=None, verbose=False)
    
    # Set up and fit ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=42,
        max_iter='auto',
        method='infomax',
        fit_params=dict(extended=True)
    )
    
    print("  Fitting ICA (this may take a moment)...")
    ica.fit(raw_ica, verbose=False)
    print(f"  ✓ ICA fitting complete. Found {ica.n_components_} components.")
    
    return ica


def create_comprehensive_summary(raw, ica, out_dir, fmin=1, fmax=50):
    """
    Generates and saves a comprehensive summary figure (raw, bands, variability, SNR).

    Inputs:
    - raw (mne.io.Raw):             Raw MNE object.
    - ica (mne.preprocessing.ICA):  Fitted ICA object (kept for compatibility).
    - out_dir (str):                Directory to save the figure.
    - fmin (float):                 Minimum PSD frequency.
    - fmax (float):                 Maximum PSD frequency.

    Outputs:
    - matplotlib.figure.Figure: The created figure object.
    """
    print("\nCreating comprehensive summary figure with additional analyses...")
    
    # Create figure with 3 rows
    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.45)
    
    # Get data
    data = raw.get_data()
    times = raw.times
    
    # Compute PSD for band analysis
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    colors = ['blue', 'orange']
    
    # ============================================================
    # 1. FULL RAW DATA (Row 1 - ALL 4 columns, shows entire recording)
    # ============================================================
    ax_raw = fig.add_subplot(gs[0, :])
    n_show = min(len(raw.ch_names), 2)
    
    colors_raw = ['blue', 'orange']
    for i in range(n_show):
        offset = i * np.std(data[i]) * 5
        color = colors_raw[i] if i < len(colors_raw) else f'C{i}'
        ax_raw.plot(times, data[i, :] * 1e6 + offset, 
                   label=raw.ch_names[i], alpha=0.7, color=color, linewidth=0.8)
    
    ax_raw.set_xlabel('Time (s)', fontsize=12)
    ax_raw.set_ylabel('Amplitude (µV)', fontsize=12)
    ax_raw.set_title(f'Complete Raw EEG Recording ({raw.times[-1]:.1f}s total)', fontsize=14, fontweight='bold')
    ax_raw.legend(fontsize=11)
    ax_raw.grid(True, alpha=0.3)
    ax_raw.set_xlim(0, raw.times[-1])
    
    # ============================================================
    # 2. FREQUENCY BAND POWER (Row 2, ALL 4 columns)
    # ============================================================
    ax_bands = fig.add_subplot(gs[1, :])
    
    # Define frequency bands
    bands = {
        'Delta\n(0.5-4 Hz)': (0.5, 4),
        'Theta\n(4-8 Hz)': (4, 8),
        'Alpha\n(8-13 Hz)': (8, 13),
        'Beta\n(13-30 Hz)': (13, 30),
        'Gamma\n(30-50 Hz)': (30, 50)
    }
    
    band_names = list(bands.keys())
    
    # Calculate power in each band for each channel
    band_power = {ch: [] for ch in raw.ch_names}
    
    for band_name, (fmin_band, fmax_band) in bands.items():
        for i, ch_name in enumerate(raw.ch_names):
            # Get frequencies within band
            freq_mask = (freqs >= fmin_band) & (freqs <= fmax_band)
            # Calculate mean power in band
            power = np.mean(psds[i, freq_mask])
            band_power[ch_name].append(power)
    
    # Plot grouped bar chart
    x = np.arange(len(band_names))
    width = 0.35
    
    for i, ch_name in enumerate(raw.ch_names):
        offset = (i - len(raw.ch_names)/2 + 0.5) * width
        color = colors[i] if i < len(colors) else f'C{i}'
        display_name = get_channel_display_name(ch_name)
        ax_bands.bar(x + offset, band_power[ch_name], width, 
                    label=display_name, alpha=0.8, color=color)
    
    ax_bands.set_xlabel('Frequency Band', fontsize=12)
    ax_bands.set_ylabel('Mean Power (V²/Hz)', fontsize=12)
    ax_bands.set_title('Power by Frequency Band (Delta, Theta, Alpha, Beta, Gamma)', fontsize=14, fontweight='bold')
    ax_bands.set_xticks(x)
    ax_bands.set_xticklabels(band_names, fontsize=10)
    ax_bands.legend(fontsize=11)
    ax_bands.grid(True, alpha=0.3, axis='y')
    ax_bands.set_yscale('log')
    
    # ============================================================
    # 3. CHANNEL-WISE SIGNAL VARIABILITY + SNR (Row 3, Left 3 columns)
    # ============================================================
    ax_var = fig.add_subplot(gs[2, :3])
    
    # Calculate various metrics per channel
    channel_means = np.mean(np.abs(data), axis=1) * 1e6
    channel_stds = np.std(data, axis=1) * 1e6
    channel_maxs = np.max(np.abs(data), axis=1) * 1e6
    
    x_pos = np.arange(len(raw.ch_names))
    width_bar = 0.25
    
    bars1 = ax_var.bar(x_pos - width_bar, channel_means, width_bar, 
                       label='Mean Amplitude', alpha=0.8, color='skyblue')
    bars2 = ax_var.bar(x_pos, channel_stds, width_bar, 
                       label='Std Deviation', alpha=0.8, color='lightcoral')
    bars3 = ax_var.bar(x_pos + width_bar, channel_maxs, width_bar, 
                       label='Max Amplitude', alpha=0.8, color='lightgreen')
    
    ax_var.set_xlabel('Channel', fontsize=12)
    ax_var.set_ylabel('Amplitude (µV)', fontsize=12)
    ax_var.set_title('Channel-wise Signal Variability', fontsize=14, fontweight='bold')
    ax_var.set_xticks(x_pos)
    ax_var.set_xticklabels(raw.ch_names)
    ax_var.legend(fontsize=11)
    ax_var.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # 4. SIGNAL-TO-NOISE RATIO (Row 3, Right 1 column)
    # ============================================================
    ax_snr = fig.add_subplot(gs[2, 3])
    
    # Calculate SNR for each channel
    # For blink analysis, we use general SNR: RMS signal / noise floor
    # Signal: RMS of the entire time series
    # Noise: Estimate from high-frequency content or baseline periods
    snr_values = []
    
    for i, ch_name in enumerate(raw.ch_names):
        # Method 1: RMS-based SNR
        # Signal RMS: Root mean square of the signal
        signal_rms = np.sqrt(np.mean(data[i, :]**2)) * 1e6  # in µV
        
        # Noise floor estimate: Use standard deviation of high-pass filtered signal
        # High-pass at 30 Hz to isolate noise
        noise_data = mne.filter.filter_data(
            data[i, :], raw.info['sfreq'], 
            l_freq=30, h_freq=None, 
            verbose=False
        )
        noise_rms = np.sqrt(np.mean(noise_data**2)) * 1e6  # in µV
        
        # Calculate SNR in dB
        if noise_rms > 0:
            snr_db = 20 * np.log10(signal_rms / noise_rms)  # 20*log10 for voltage ratio
        else:
            snr_db = 0
        
        snr_values.append(snr_db)
    
    # Plot SNR
    color_snr = [colors[i] if i < len(colors) else f'C{i}' for i in range(len(raw.ch_names))]
    bars_snr = ax_snr.bar(range(len(raw.ch_names)), snr_values, 
                           color=color_snr, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars_snr, snr_values)):
        height = bar.get_height()
        ax_snr.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f} dB',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax_snr.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax_snr.set_ylabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax_snr.set_title('Signal-to-Noise Ratio\n(RMS Signal/Noise Floor)', fontsize=13, fontweight='bold')
    ax_snr.set_xticks(range(len(raw.ch_names)))
    ax_snr.set_xticklabels(raw.ch_names, fontsize=11)
    ax_snr.grid(True, alpha=0.3, axis='y')
    ax_snr.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # Add average SNR text
    avg_snr = np.mean(snr_values)
    ax_snr.text(0.5, 0.95, f'Average SNR: {avg_snr:.1f} dB', 
               transform=ax_snr.transAxes, ha='center', va='top',
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Add SNR quality indicator
    quality = "Excellent" if avg_snr > 20 else "Good" if avg_snr > 15 else "Fair" if avg_snr > 10 else "Poor"
    ax_snr.text(0.5, 0.85, f'Quality: {quality}', 
               transform=ax_snr.transAxes, ha='center', va='top',
               fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='lightgreen' if avg_snr > 15 else 'lightyellow' if avg_snr > 10 else 'lightcoral', alpha=0.4))
    
    
    # ============================================================
    # OVERALL TITLE
    # ============================================================
    fig.suptitle('Comprehensive EEG Analysis: Raw Data + Frequency Bands + Variability', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'comprehensive_analysis_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comprehensive summary: {save_path}")
    
    return fig


def create_standalone_psd_summary(raw, out_dir, fmin=1, fmax=50, psd_ylim=None, highlight_alpha=False):
    """
    Creates and saves a PSD summary figure for selected channels.

    Inputs:
    - raw (mne.io.Raw):         Raw MNE object.
    - out_dir (str):            Directory to save the figure.
    - fmin (float):             Minimum PSD frequency.
    - fmax (float):             Maximum PSD frequency.
    - psd_ylim (tuple|None):    Optional y-axis limits.
    - highlight_alpha (bool):   If True, highlight 8-13 Hz region.

    Outputs:
    - matplotlib.figure.Figure: The created PSD figure.
    """
    print("\nCreating standalone PSD summary...")
    
    # Compute PSD
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    
    colors = ['blue', 'orange']

    # If there are more than two channels, focus PSD plots on CH4 and CH5.
    if len(raw.ch_names) > 2:
        selected_indices = [idx for idx in (3, 4) if idx < len(raw.ch_names)]
        if len(selected_indices) < 2:
            selected_indices = list(range(len(raw.ch_names)))
    else:
        selected_indices = list(range(len(raw.ch_names)))
    
    # ============================================================
    # 1. Main PSD Plot (Top Left)
    # ============================================================
    ax_psd = fig.add_subplot(gs[0, 0])
    
    for i in selected_indices:
        ch_name = raw.ch_names[i]
        color = colors[i] if i < len(colors) else f'C{i}'
        display_name = get_channel_display_name(ch_name)
        ax_psd.semilogy(freqs, psds[i], label=display_name, color=color, linewidth=2.5, alpha=0.8)
    
    ax_psd.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_psd.set_ylabel('Power Spectral Density (V²/Hz)', fontsize=12)
    ax_psd.set_title('Power Spectral Density', fontsize=13, fontweight='bold')
    ax_psd.grid(True, alpha=0.3)
    ax_psd.legend(fontsize=11, loc='upper right')
    ax_psd.set_xlim(fmin, fmax)
    if psd_ylim:
        ax_psd.set_ylim(psd_ylim)
    if highlight_alpha:
        ax_psd.axvspan(8, 13, color='purple', alpha=0.15, label='Alpha Band (8-13 Hz)')
        ax_psd.legend(fontsize=11, loc='upper right')
    
    # ============================================================
    # 2. Electrode Positions (Top Right)
    # ============================================================
    ax_topo = fig.add_subplot(gs[0, 1])
    if raw.get_montage() is not None:
        try:
            mne.viz.plot_sensors(raw.info, axes=ax_topo, show_names=True, show=False)
            ax_topo.set_title('Electrode Positions', fontsize=13, fontweight='bold')
        except Exception as e:
            ax_topo.text(0.5, 0.5, f'Electrode positions\nnot available', 
                        ha='center', va='center', transform=ax_topo.transAxes, fontsize=11)
            ax_topo.axis('off')
            ax_topo.set_title('Electrode Positions', fontsize=13, fontweight='bold')
    else:
        ax_topo.text(0.5, 0.5, 'No montage set\n(electrode positions unknown)', 
                    ha='center', va='center', transform=ax_topo.transAxes, fontsize=11)
        ax_topo.axis('off')
        ax_topo.set_title('Electrode Positions', fontsize=13, fontweight='bold')
    
    # ============================================================
    # 3. Frequency Band Power (Bottom Left)
    # ============================================================
    ax_bands = fig.add_subplot(gs[1, 0])
    
    # Define frequency bands
    bands = {
        'Delta\n(0.5-4 Hz)': (0.5, 4),
        'Theta\n(4-8 Hz)': (4, 8),
        'Alpha\n(8-13 Hz)': (8, 13),
        'Beta\n(13-30 Hz)': (13, 30),
        'Gamma\n(30-50 Hz)': (30, 50)
    }
    
    band_names = list(bands.keys())
    
    # Calculate power in each band
    band_power = {raw.ch_names[i]: [] for i in selected_indices}
    
    for band_name, (fmin_band, fmax_band) in bands.items():
        for i in selected_indices:
            ch_name = raw.ch_names[i]
            freq_mask = (freqs >= fmin_band) & (freqs <= fmax_band)
            power = np.mean(psds[i, freq_mask])
            band_power[ch_name].append(power)
    
    # Plot
    x = np.arange(len(band_names))
    width = 0.35
    
    n_selected = len(selected_indices)
    for plot_pos, i in enumerate(selected_indices):
        ch_name = raw.ch_names[i]
        offset = (plot_pos - n_selected/2 + 0.5) * width
        color = colors[i] if i < len(colors) else f'C{i}'
        display_name = get_channel_display_name(ch_name)
        ax_bands.bar(x + offset, band_power[ch_name], width, 
                    label=display_name, alpha=0.8, color=color)
    
    ax_bands.set_xlabel('Frequency Band', fontsize=12)
    ax_bands.set_ylabel('Mean Power (V²/Hz)', fontsize=12)
    ax_bands.set_title('Power by Frequency Band', fontsize=13, fontweight='bold')
    ax_bands.set_xticks(x)
    ax_bands.set_xticklabels(band_names, fontsize=10)
    ax_bands.legend(fontsize=11)
    ax_bands.grid(True, alpha=0.3, axis='y')
    ax_bands.set_yscale('log')
    
    # ============================================================
    # 4. PSD Statistics (Bottom Right)
    # ============================================================
    ax_stats = fig.add_subplot(gs[1, 1])
    
    # Calculate statistics for each channel
    stats_text = "PSD STATISTICS\n\n"
    
    for i in selected_indices:
        ch_name = raw.ch_names[i]
        # Peak frequency
        peak_freq = freqs[np.argmax(psds[i])]
        peak_power = np.max(psds[i])
        
        # Mean power
        mean_power = np.mean(psds[i])
        
        # Alpha band power
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        alpha_power = np.mean(psds[i, alpha_mask])
        
        stats_text += f"{ch_name}:\n"
        stats_text += f"  Peak: {peak_freq:.1f} Hz\n"
        stats_text += f"  Peak Power: {peak_power:.2e} V²/Hz\n"
        stats_text += f"  Mean Power: {mean_power:.2e} V²/Hz\n"
        stats_text += f"  Alpha Power: {alpha_power:.2e} V²/Hz\n\n"
    ax_stats.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                 verticalalignment='center', transform=ax_stats.transAxes)
    ax_stats.axis('off')
    ax_stats.set_title('Channel Statistics', fontsize=13, fontweight='bold')
    
    fig.suptitle('Power Spectral Density Summary', fontsize=15, fontweight='bold', y=0.98)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'psd_summary_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved standalone PSD summary: {save_path}")
    
    return fig


def create_standalone_ica_summary(raw, ica, out_dir):
    """
    Creates and saves plots summarizing ICA components and time courses.

    Inputs:
    - raw (mne.io.Raw):             Raw MNE object.
    - ica (mne.preprocessing.ICA):  Fitted ICA object.
    - out_dir (str):                Directory to save the figure.

    Outputs:
    - matplotlib.figure.Figure: The ICA summary figure.
    """
    print("\nCreating standalone ICA summary...")
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    
    times = raw.times
    sources = ica.get_sources(raw).get_data()
    n_components = min(ica.n_components_, 2)
    
    # ============================================================
    # 1. ICA Component Topomaps (Top Row)
    # ============================================================
    for idx in range(n_components):
        ax_comp = fig.add_subplot(gs[0, idx])
        
        try:
            ica.plot_components(picks=idx, axes=ax_comp, show=False, colorbar=True)
            ax_comp.set_title(f'ICA Component {idx}', fontsize=13, fontweight='bold')
        except Exception as e:
            ax_comp.text(0.5, 0.5, f'Component {idx}\n(no topomap)', 
                        ha='center', va='center', transform=ax_comp.transAxes, fontsize=11)
            ax_comp.axis('off')
            ax_comp.set_title(f'ICA Component {idx}', fontsize=13, fontweight='bold')
    
    # ============================================================
    # 2. ICA Time Courses (Bottom Left)
    # ============================================================
    ax_time = fig.add_subplot(gs[1, 0])
    
    duration_show = min(30.0, times[-1])
    time_mask = times <= duration_show
    
    colors = ['blue', 'orange', 'green', 'red']
    for idx in range(n_components):
        offset = idx * np.std(sources[idx]) * 3
        color = colors[idx] if idx < len(colors) else f'C{idx}'
        ax_time.plot(times[time_mask], sources[idx, time_mask] + offset,
                    label=f'IC{idx}', alpha=0.7, linewidth=1.5, color=color)
    
    ax_time.set_xlabel('Time (s)', fontsize=12)
    ax_time.set_ylabel('ICA Source Activity (a.u.)', fontsize=12)
    ax_time.set_title(f'ICA Time Course (first {duration_show:.0f}s)', fontsize=13, fontweight='bold')
    ax_time.legend(fontsize=11)
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlim(0, duration_show)
    
    # ============================================================
    # 3. ICA Statistics (Bottom Right)
    # ============================================================
    ax_stats = fig.add_subplot(gs[1, 1])
    
    stats_text = f"""ICA SUMMARY

Method: {ica.method}
Components: {ica.n_components_}
Samples Fitted: {ica.n_samples_}

COMPONENT STATISTICS:
"""
    
    for idx in range(n_components):
        # Calculate statistics
        ic_std = np.std(sources[idx])
        ic_max = np.max(np.abs(sources[idx]))
        ic_mean = np.mean(np.abs(sources[idx]))
        
        # Variance explained (approximate)
        total_var = np.sum([np.var(sources[i]) for i in range(ica.n_components_)])
        var_explained = (np.var(sources[idx]) / total_var) * 100
        
        stats_text += f"\nComponent {idx}:\n"
        stats_text += f"  Std Dev: {ic_std:.3f}\n"
        stats_text += f"  Max Abs: {ic_max:.3f}\n"
        stats_text += f"  Mean Abs: {ic_mean:.3f}\n"
        stats_text += f"  Var Explained: {var_explained:.1f}%\n"
    
    stats_text += """
INTERPRETATION:
• IC0 typically captures most variance
• Frontal components → eye artifacts
• High-frequency → muscle/noise
• Rhythmic patterns → neural activity
"""
    
    ax_stats.text(0.05, 0.5, stats_text, fontsize=9.5, family='monospace',
                 verticalalignment='center', transform=ax_stats.transAxes,
                 linespacing=1.4)
    ax_stats.axis('off')
    ax_stats.set_title('ICA Information', fontsize=13, fontweight='bold')
    
    fig.suptitle('Independent Component Analysis Summary', fontsize=15, fontweight='bold', y=0.98)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'ica_summary_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved standalone ICA summary: {save_path}")
    
    return fig


def create_standalone_raw_eeg(raw, out_dir, start_time=30, end_time=45, cut_start=0, cut_end=0):
    """
    Plots and saves a raw EEG segment for a specified time window.

    Inputs:
    - raw (mne.io.Raw):     Raw MNE object.
    - out_dir (str):        Directory to save the plot.
    - start_time (float):   Start time in seconds.
    - end_time (float):     End time in seconds.
    - cut_start (float):    Seconds to exclude from start.
    - cut_end (float):      Seconds to exclude from end.

    Outputs:
    - matplotlib.figure.Figure: The raw EEG plot figure.
    """
    print(f"\nCreating standalone raw EEG plot ({start_time}-{end_time}s window, excluding cut-off portions)...")

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Get data and times
    data, times = raw.get_data(return_times=True)

    # Adjust time window to exclude cut-off portions
    valid_start = max(start_time, cut_start)
    valid_end = min(end_time, times[-1] - cut_end)

    # Dynamically adjust to full raw EEG data if the window is invalid
    if valid_end <= valid_start:
        print("Invalid time window after applying cut-off portions. Using full raw EEG data.")
        valid_start = 0
        valid_end = times[-1]

    # Extract valid time window
    time_mask = (times >= valid_start) & (times <= valid_end)
    times_window = times[time_mask]
    data_window = data[:, time_mask]

    # Color mapping
    colors = {'Fp1': 'blue', 'Fp2': 'orange'}

    # Plot each channel with offset
    for i, ch_name in enumerate(raw.ch_names):
        color = colors.get(ch_name, 'gray')
        offset = i * np.std(data[i]) * 5  # Vertical offset
        display_name = get_channel_display_name(ch_name)
        ax.plot(times_window, data_window[i, :] * 1e6 + offset, 
                label=display_name, color=color, linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Amplitude (µV) + offset', fontsize=13, fontweight='bold')
    ax.set_title(f'Raw EEG Recording ({valid_start}-{valid_end}s window)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(valid_start, valid_end)

    # Add vertical lines every second for easier time reading
    for t in range(int(valid_start), int(valid_end) + 1):
        ax.axvline(x=t, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'raw_eeg_{valid_start}-{valid_end}s_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved standalone raw EEG: {save_path}")

    return fig


def create_alpha_wave_analysis(raw, out_dir, channel_indices=None):
    """
    Performs an alpha-band analysis and saves visual summaries.

    Inputs:
    - raw (mne.io.Raw):                 Raw MNE object.
    - out_dir (str):                    Directory to save plots.
    - channel_indices (iterable|None):  Optional channel indices to analyze.

    Outputs:
    - matplotlib.figure.Figure: Figure containing alpha analysis plots.
    """
    print("\n" + "="*60)
    print("ALPHA WAVE ANALYSIS (8-13 Hz)")
    print("="*60)
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    colors = {'Fp1': 'blue', 'Fp2': 'orange', 'Ch4': 'blue', 'Ch5': 'orange', 'CH4': 'blue', 'CH5': 'orange'}

    if channel_indices is None:
        selected_indices = list(range(len(raw.ch_names)))
    else:
        selected_indices = [idx for idx in channel_indices if 0 <= idx < len(raw.ch_names)]
        if not selected_indices:
            print("  Warning: No valid alpha channel indices provided; using all channels.")
            selected_indices = list(range(len(raw.ch_names)))
    
    # Extract alpha band using bandpass filter
    print("  Filtering for alpha band (8-13 Hz)...")
    raw_alpha = raw.copy().filter(l_freq=8.0, h_freq=13.0, verbose=False)
    alpha_data = raw_alpha.get_data()
    times = raw.times
    
    # Compute PSD
    spectrum = raw.compute_psd(fmin=1, fmax=50, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # ============================================================
    # 1. ALPHA POWER SPECTRUM (Top Left)
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    
    for i in selected_indices:
        ch_name = raw.ch_names[i]
        color = colors.get(ch_name, f'C{i}')
        display_name = get_channel_display_name(ch_name)
        ax1.semilogy(freqs[alpha_mask], psds[i, alpha_mask], 
                    linewidth=3, label=display_name, color=color)
        
        # Mark peak alpha frequency
        peak_idx = np.argmax(psds[i, alpha_mask])
        peak_freq = freqs[alpha_mask][peak_idx]
        peak_power = psds[i, alpha_mask][peak_idx]
        ax1.plot(peak_freq, peak_power, 'o', markersize=10, 
                color=color, markeredgecolor='black', markeredgewidth=2)
    
    ax1.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Power (V²/Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('Alpha Band Power Spectrum (8-13 Hz)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(8, 13)
    
    # ============================================================
    # 2. ALPHA POWER OVER TIME (Top Right)
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate moving window alpha power
    window_size = int(raw.info['sfreq'] * 2)  # 2-second windows
    step_size = int(raw.info['sfreq'] * 0.5)  # 0.5-second steps
    
    # Pre-calculate all alpha power values to find global min/max
    all_alpha_power_values = []
    alpha_power_data = []
    
    for i in selected_indices:
        ch_name = raw.ch_names[i]
        color = colors.get(ch_name, f'C{i}')
        display_name = get_channel_display_name(ch_name)
        
        alpha_power_time = []
        time_points = []
        
        for start in range(0, len(times) - window_size, step_size):
            end = start + window_size
            window_data = alpha_data[i, start:end]
            power = np.mean(window_data ** 2) * 1e12  # Convert to (µV)²
            alpha_power_time.append(power)
            time_points.append(times[start + window_size // 2])
            all_alpha_power_values.append(power)
        
        alpha_power_data.append((time_points, alpha_power_time, display_name, color))
    
    # Set fixed y-axis limits with some margin
    if all_alpha_power_values:
        alpha_power_min = min(all_alpha_power_values)
        alpha_power_max = max(all_alpha_power_values)
        alpha_power_range = alpha_power_max - alpha_power_min
        if alpha_power_range == 0:
            alpha_power_range = 1
        ylim_alpha_power = (alpha_power_min - 0.05 * alpha_power_range, 
                           alpha_power_max + 0.05 * alpha_power_range)
    else:
        ylim_alpha_power = (0, 1)
    
    # Plot with fixed y-axis
    for time_points, alpha_power_time, display_name, color in alpha_power_data:
        ax2.plot(time_points, alpha_power_time, linewidth=2, 
                label=display_name, color=color, alpha=0.8)
    
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Alpha Power (µV²)', fontsize=12, fontweight='bold')
    ax2.set_title('Alpha Power Over Time (2s windows)', fontsize=13, fontweight='bold')
    ax2.set_ylim(ylim_alpha_power)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ============================================================
    # 3. FILTERED ALPHA SIGNAL (Middle Left)
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Show 10-second window of filtered alpha
    duration_show = min(10.0, times[-1])
    time_mask = times <= duration_show
    
    for i in selected_indices:
        ch_name = raw.ch_names[i]
        color = colors.get(ch_name, f'C{i}')
        display_name = get_channel_display_name(ch_name)
        offset = i * 10  # Fixed offset of 10 µV per channel
        ax3.plot(times[time_mask], alpha_data[i, time_mask] * 1e6 + offset,
                linewidth=1.5, label=display_name, color=color, alpha=0.8)
    
    ax3.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Amplitude (µV) + offset', fontsize=12, fontweight='bold')
    ax3.set_title(f'Filtered Alpha Signal (first {duration_show:.0f}s)', 
                 fontsize=13, fontweight='bold')
    ax3.set_xlim(0, duration_show)
    ax3.set_ylim(-30, 30)  # Fixed y-axis
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # ============================================================
    # 4. ALPHA AMPLITUDE ENVELOPE (Middle Right)
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i in selected_indices:
        ch_name = raw.ch_names[i]
        color = colors.get(ch_name, f'C{i}')
        display_name = get_channel_display_name(ch_name)
        
        # Calculate amplitude envelope using Hilbert transform
        analytic_signal = signal.hilbert(alpha_data[i])
        envelope = np.abs(analytic_signal) * 1e6  # Convert to µV
        
        # Downsample for plotting (every 10th point)
        downsample = 10
        ax4.plot(times[::downsample], envelope[::downsample],
                linewidth=1.5, label=display_name, color=color, alpha=0.7)
    
    ax4.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Alpha Amplitude Envelope (µV)', fontsize=12, fontweight='bold')
    ax4.set_title('Alpha Wave Amplitude Modulation', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, 30)  # Fixed y-axis
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # ============================================================
    # 5. STATISTICS BOX (Bottom Row)
    # ============================================================
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate statistics
    stats_lines = ["╔" + "="*108 + "╗"]
    stats_lines.append("║" + " "*37 + "ALPHA WAVE STATISTICS" + " "*50 + "║")
    stats_lines.append("╠" + "="*108 + "╣")
    
    for i in selected_indices:
        ch_name = raw.ch_names[i]
        # Peak alpha frequency
        alpha_psd = psds[i, alpha_mask]
        peak_idx = np.argmax(alpha_psd)
        peak_freq = freqs[alpha_mask][peak_idx]
        
        # Mean alpha power
        mean_alpha_power = np.mean(alpha_psd)
        
        # Calculate envelope statistics
        analytic_signal = signal.hilbert(alpha_data[i])
        envelope = np.abs(analytic_signal) * 1e6
        mean_envelope = np.mean(envelope)
        std_envelope = np.std(envelope)
        
        # RMS of alpha signal
        alpha_rms = np.sqrt(np.mean(alpha_data[i]**2)) * 1e6
        
        stats_lines.append(f"║  {ch_name} Channel:")
        stats_lines.append(f"║    Peak Alpha Frequency:  {peak_freq:>6.2f} Hz")
        stats_lines.append(f"║    Mean Alpha Power:      {mean_alpha_power:>6.2e} V²/Hz")
        stats_lines.append(f"║    Alpha RMS:             {alpha_rms:>6.2f} µV")
        stats_lines.append(f"║    Envelope Mean:         {mean_envelope:>6.2f} µV")
        stats_lines.append(f"║    Envelope Std Dev:      {std_envelope:>6.2f} µV")
        
        if i != selected_indices[-1]:
            stats_lines.append("║")
    
    # Add correlation if two channels
    if len(selected_indices) == 2:
        correlation = np.corrcoef(alpha_data[selected_indices[0]], alpha_data[selected_indices[1]])[0, 1]
        stats_lines.append("║")
        stats_lines.append(f"║  Channel Correlation:     {correlation:>6.3f} ({'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'})")
    
    stats_lines.append("╚" + "="*108 + "╝")
    
    # Pad lines to same length
    max_len = max(len(line) for line in stats_lines)
    stats_lines = [line + " "*(max_len - len(line)) for line in stats_lines]
    
    stats_text = "\n".join(stats_lines)
    
    ax5.text(0.5, 0.5, stats_text, fontsize=9, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, 
                     edgecolor='black', linewidth=2))
    
    # ============================================================
    # OVERALL TITLE
    # ============================================================
    fig.suptitle('Alpha Wave Analysis: Power, Modulation & Time Course', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'alpha_wave_analysis_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved alpha wave analysis: {save_path}")
    print("="*60 + "\n")
    
    return fig


def apply_baseline(power, baseline=(0.0, 5.0), mode='logratio'):
    """
    Apply a baseline correction to a time-frequency power object.

    Inputs:
    - power (mne.time_frequency.EpochsTFR): Time-frequency power object.
    - baseline (tuple):                     Baseline time window (start, end) in seconds.
    - mode (str):                           Baseline correction mode for MNE.

    Outputs:
    - mne.time_frequency.EpochsTFR: Baseline-corrected power object.
    """
    power = power.copy()
    power.apply_baseline(baseline=baseline, mode=mode)
    return power


def create_alpha_time_frequency_power(raw, out_dir, channel_indices=(2, 3, 4, 5), fmin=0.5, fmax=20.0, n_freqs=50, baseline_seconds=2.0):
    """
    Computes Morlet time-frequency power maps and save heatmaps.

    Inputs:
    - raw (mne.io.Raw):             Raw MNE object.
    - out_dir (str):                Directory to save outputs.
    - channel_indices (iterable):   Channel indices to include.
    - fmin (float):                 Minimum frequency.
    - fmax (float):                 Maximum frequency.
    - n_freqs (int):                Number of frequencies to compute.
    - baseline_seconds (float):     Baseline window length in seconds.

    Outputs:
    - matplotlib.figure.Figure or None: Figure with time-frequency maps, or None on error.
    """
    print("\nCreating alpha time-frequency power heatmaps for selected channels...")

    sfreq = raw.info['sfreq']
    freqs = np.linspace(fmin, fmax, n_freqs)
    n_cycles = np.maximum(freqs / 2.0, 1.0)
    times = raw.times

    valid_channel_indices = [index for index in channel_indices if 0 <= index < len(raw.ch_names)]
    if not valid_channel_indices:
        print("  Warning: No valid channel indices found for time-frequency plot.")
        return None

    epoch_data = raw.get_data()[np.newaxis, :, :]
    epochs = mne.EpochsArray(epoch_data, info=raw.info.copy(), tmin=0.0, verbose=False)

    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        use_fft=True,
        return_itc=False,
        average=True,
        decim=1,
        n_jobs=1,
        verbose=False,
    )
    baseline_start = 1.0
    baseline_end = min(baseline_seconds, power.times[-1])
    if baseline_end <= baseline_start:
        baseline_start = 0.0
    power = apply_baseline(power, baseline=(baseline_start, baseline_end), mode='logratio')

    fig, axes = plt.subplots(len(valid_channel_indices), 1, figsize=(16, 7 * len(valid_channel_indices)), sharex=True)
    if len(valid_channel_indices) == 1:
        axes = [axes]

    for ax, channel_index in zip(axes, valid_channel_indices):
        channel_name = raw.ch_names[channel_index]
        power.plot(
            picks=[channel_index],
            fmin=fmin,
            fmax=fmax,
            axes=ax,
            show=False,
            title=f'Time-Frequency Power Map - {channel_name}',
        )
        ax.set_ylim(0, 20)
        ax.axhspan(8, 13, color='white', alpha=0.08, linewidth=0)
        ax.set_ylabel('Frequency (Hz)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', labelbottom=True)

    fig.suptitle('Alpha Time-Frequency Power', fontsize=15, fontweight='bold')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'alpha_time_frequency_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved alpha time-frequency power maps: {save_path}")

    return fig


def create_blink_epoch_analysis(raw, out_dir, channel_indices=None):
    """
    Detect blink events, create epochs, and produce summary plots.

    Inputs:
    - raw (mne.io.Raw):                 Raw MNE object.
    - out_dir (str):                    Directory to save figures.
    - channel_indices (iterable|None):  Optional indices to detect/compare blinks.

    Outputs:
    - tuple: (fig (matplotlib.figure.Figure)|None, epochs (mne.Epochs)|None)
    """
    print("\n" + "="*60)
    print("BLINK EPOCH ANALYSIS")
    print("="*60)
    print("\nDetecting blink events...")

    if channel_indices is None:
        if 'Fp1' in raw.ch_names and 'Fp2' in raw.ch_names:
            channel_indices = (raw.ch_names.index('Fp1'), raw.ch_names.index('Fp2'))
        else:
            channel_indices = (0, 1) if len(raw.ch_names) > 1 else (0,)

    blink_idx = channel_indices[0]
    compare_idx = channel_indices[1] if len(channel_indices) > 1 else channel_indices[0]
    blink_label = get_channel_display_name(f'CH{blink_idx + 1}')
    compare_label = get_channel_display_name(f'CH{compare_idx + 1}')

    # Detect EOG events (blinks) using the first selected channel as reference
    try:
        eog_events = mne.preprocessing.find_eog_events(
            raw, 
            ch_name=raw.ch_names[blink_idx],  # Use the selected channel for detection
            event_id=998,
            l_freq=1,
            h_freq=15,
            thresh=300e-6  # 75µV threshold (adjust if needed)
        )
    except Exception as e:
        print(f"  Error detecting blinks: {e}")
        print("  Trying alternative detection method...")
        
        # Alternative: use simple threshold detection
        data_blink = raw.get_data(picks=[blink_idx])[0]
        filtered = mne.filter.filter_data(data_blink, raw.info['sfreq'], l_freq=1, h_freq=15, verbose=False)
        threshold = np.std(filtered) * 3
        
        # Find peaks
        peaks, _ = signal.find_peaks(np.abs(filtered), height=threshold, distance=int(raw.info['sfreq'] * 0.5))
        
        if len(peaks) == 0:
            print("  ✗ No blinks detected. Skipping epoch analysis.")
            return None, None
        
        # Create events array
        eog_events = np.zeros((len(peaks), 3), dtype=int)
        eog_events[:, 0] = peaks
        eog_events[:, 2] = 998
    
    print(f"  ✓ Found {len(eog_events)} blink events")
    
    if len(eog_events) == 0:
        print("  ✗ No blinks detected. Skipping epoch analysis.")
        return None, None
    
    # Create epochs around blinks
    print("  Creating epochs around blink events...")
    epochs = mne.Epochs(
        raw,
        eog_events,
        event_id={'Blink': 998},
        tmin=-0.2,  # 200ms before blink
        tmax=0.7,   # 700ms after blink
        baseline=(None, 0),
        preload=True,
        reject=dict(eeg=600e-6),  # Reject epochs with >500µV amplitude
        verbose=False
    )
    
    print(f"  ✓ Created {len(epochs)} valid epochs (rejected {len(eog_events) - len(epochs)} bad epochs)")
    
    if len(epochs) == 0:
        print("  ✗ All epochs rejected. Try adjusting rejection threshold.")
        return None, None
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    colors = {blink_label: 'blue', compare_label: 'orange'}
    
    # Get epoch data
    data_fp1 = epochs.get_data(picks=[blink_idx])[:, 0, :] * 1e6  # Convert to µV
    data_fp2 = epochs.get_data(picks=[compare_idx])[:, 0, :] * 1e6
    times = epochs.times
    
    # ============================================================
    # 1. AVERAGE BLINK WAVEFORM (ERP)
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    evoked_fp1 = epochs.average(picks=[blink_idx])
    evoked_fp2 = epochs.average(picks=[compare_idx])
    
    ax1.plot(times, evoked_fp1.data[0] * 1e6, 'b-', linewidth=2.5, label=blink_label)
    ax1.plot(times, evoked_fp2.data[0] * 1e6, color='orange', linewidth=2.5, label=compare_label)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Blink onset')
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    ax1.fill_between(times, 0, evoked_fp1.data[0] * 1e6, alpha=0.2, color='blue')
    ax1.fill_between(times, 0, evoked_fp2.data[0] * 1e6, alpha=0.2, color='orange')
    
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Average Blink Response - ERP (n={len(epochs)} blinks)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ============================================================
    # 2. INDIVIDUAL EPOCHS OVERLAY (Fp1)
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    n_show = min(30, len(data_fp1))  # Show up to 30 individual blinks
    for i in range(n_show):
        ax2.plot(times, data_fp1[i], 'b-', alpha=0.15, linewidth=0.7)
    
    # Plot average on top
    ax2.plot(times, data_fp1.mean(axis=0), 'b-', linewidth=3, label='Average', zorder=10)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude (µV)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{blink_label}: Individual Blinks (showing {n_show}/{len(epochs)})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ============================================================
    # 3. PEAK AMPLITUDE DISTRIBUTION
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    peak_fp1 = np.max(np.abs(data_fp1), axis=1)
    peak_fp2 = np.max(np.abs(data_fp2), axis=1)
    
    ax3.hist(peak_fp1, bins=20, alpha=0.6, color='blue', label=blink_label, edgecolor='black', linewidth=1.2)
    ax3.hist(peak_fp2, bins=20, alpha=0.6, color='orange', label=compare_label, edgecolor='black', linewidth=1.2)
    
    # Add mean lines
    ax3.axvline(np.mean(peak_fp1), color='blue', linestyle='--', linewidth=2.5, 
               label=f'{blink_label} mean: {np.mean(peak_fp1):.1f}µV')
    ax3.axvline(np.mean(peak_fp2), color='orange', linestyle='--', linewidth=2.5, 
               label=f'{compare_label} mean: {np.mean(peak_fp2):.1f}µV')
    
    ax3.set_xlabel('Peak Amplitude (µV)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Blink Amplitude Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ============================================================
    # 4. AMPLITUDE SCATTER PLOT (Fp1 vs Fp2)
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Scatter plot
    ax4.scatter(peak_fp1, peak_fp2, alpha=0.6, s=50, edgecolors='black', linewidth=1)
    
    # Add diagonal line (perfect correlation)
    max_val = max(np.max(peak_fp1), np.max(peak_fp2))
    ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.5, label='Perfect correlation')
    
    # Calculate correlation
    correlation = np.corrcoef(peak_fp1, peak_fp2)[0, 1]
    
    ax4.set_xlabel(f'{blink_label} Peak Amplitude (µV)', fontsize=12, fontweight='bold')
    ax4.set_ylabel(f'{compare_label} Peak Amplitude (µV)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Blink Amplitude Correlation\n(r = {correlation:.3f})', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # ============================================================
    # 5. STATISTICS BOX
    # ============================================================
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate time to peak
    time_to_peak_fp1 = times[np.argmax(np.abs(evoked_fp1.data[0]))] * 1000
    time_to_peak_fp2 = times[np.argmax(np.abs(evoked_fp2.data[0]))] * 1000
    
    # Calculate consistency (coefficient of variation)
    cv_fp1 = (np.std(peak_fp1) / np.mean(peak_fp1)) * 100
    cv_fp2 = (np.std(peak_fp2) / np.mean(peak_fp2)) * 100
    
    stats_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                         BLINK EPOCH STATISTICS                                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  DETECTION SUMMARY                                                                                       ║
║    Total Blinks Detected:  {len(eog_events):<4}   |   Valid Epochs After Rejection:  {len(epochs):<4}   |   Rejection Rate: {(1 - len(epochs)/len(eog_events))*100:.1f}%    ║
║                                                                                                          ║
║  PEAK AMPLITUDE ANALYSIS                           │  TEMPORAL CHARACTERISTICS                          ║
║    Fp1 (Blue):                                     │    Time to Peak:                                   ║
║      Mean:     {np.mean(peak_fp1):>7.1f} µV                           │      Fp1:  {time_to_peak_fp1:>6.0f} ms                             ║
║      Std Dev:  {np.std(peak_fp1):>7.1f} µV                           │      Fp2:  {time_to_peak_fp2:>6.0f} ms                             ║
║      Max:      {np.max(peak_fp1):>7.1f} µV                           │                                                    ║
║      Min:      {np.min(peak_fp1):>7.1f} µV                           │    Baseline-to-Peak Duration:                      ║
║      CV:       {cv_fp1:>7.1f} %                            │      Fp1:  {(time_to_peak_fp1 + 200):.0f} ms (from -200ms)           ║
║                                                    │      Fp2:  {(time_to_peak_fp2 + 200):.0f} ms (from -200ms)           ║
║    Fp2 (Orange):                                   │                                                    ║
║      Mean:     {np.mean(peak_fp2):>7.1f} µV                           │  CONSISTENCY METRICS                               ║
║      Std Dev:  {np.std(peak_fp2):>7.1f} µV                           │    Coefficient of Variation (CV):                  ║
║      Max:      {np.max(peak_fp2):>7.1f} µV                           │      Fp1:  {cv_fp1:>5.1f}% ({"Low" if cv_fp1 < 30 else "Moderate" if cv_fp1 < 50 else "High"} variability)                ║
║      Min:      {np.min(peak_fp2):>7.1f} µV                           │      Fp2:  {cv_fp2:>5.1f}% ({"Low" if cv_fp2 < 30 else "Moderate" if cv_fp2 < 50 else "High"} variability)                ║
║                                                    │                                                    ║
║  CHANNEL COMPARISON                                │    Correlation: {correlation:>5.3f}                                ║
║    Amplitude Difference:  {np.mean(peak_fp1) - np.mean(peak_fp2):>6.1f} µV            │      {"Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"} correlation between channels    ║
║    Percent Difference:    {((np.mean(peak_fp1) - np.mean(peak_fp2))/np.mean(peak_fp2))*100:>6.1f} %              │                                                    ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax5.text(0.5, 0.5, stats_text, fontsize=9, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            transform=ax5.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, edgecolor='black', linewidth=2))
    
    # ============================================================
    # OVERALL TITLE
    # ============================================================
    fig.suptitle('Blink Epoch Analysis: Event-Related Potentials & Amplitude Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'blink_epoch_analysis_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved blink epoch analysis: {save_path}")
    print("="*60 + "\n")
    
    return fig, epochs


def calculate_snr_fp2_over_difference(raw):
    """
    Computes a custom SNR of Fp2 relative to (Fp2 - Fp1) difference across the recording.

    Inputs:
    - raw (mne.io.Raw): Raw MNE object containing Fp1 and Fp2.

    Outputs:
    - float|None: SNR in dB or None if channels missing.
    """
    if 'Fp1' in raw.ch_names and 'Fp2' in raw.ch_names:
        fp1_idx = raw.ch_names.index('Fp1')
        fp2_idx = raw.ch_names.index('Fp2')

        # Extract data for Fp1 and Fp2
        fp1_data = raw.get_data(picks=fp1_idx)[0]
        fp2_data = raw.get_data(picks=fp2_idx)[0]

        # Calculate the difference (Fp2 - Fp1)
        difference_data = fp2_data - fp1_data

        # Calculate RMS for signal (Fp2) and noise (difference)
        signal_rms = np.sqrt(np.mean(fp2_data**2))
        noise_rms = np.sqrt(np.mean(difference_data**2))

        # Compute SNR in dB
        if noise_rms > 0:
            return 20 * np.log10(signal_rms / noise_rms)
        else:
            return 0
    else:
        print("Fp1 or Fp2 not found in the data.")
        return None


def calculate_snr_fp1_over_difference(raw):
    """
    Computes a custom SNR of Fp1 relative to (Fp1 - Fp2) difference across the recording.

    Inputs:
    - raw (mne.io.Raw): Raw MNE object containing Fp1 and Fp2.

    Outputs:
    - float|None: SNR in dB or None if channels missing.
    """
    if 'Fp1' in raw.ch_names and 'Fp2' in raw.ch_names:
        fp1_idx = raw.ch_names.index('Fp1')
        fp2_idx = raw.ch_names.index('Fp2')

        # Extract data for Fp1 and Fp2
        fp1_data = raw.get_data(picks=fp1_idx)[0]
        fp2_data = raw.get_data(picks=fp2_idx)[0]

        # Calculate the difference (Fp1 - Fp2)
        difference_data = fp1_data - fp2_data

        # Calculate RMS for signal (Fp1) and noise (difference)
        signal_rms = np.sqrt(np.mean(fp1_data**2))
        noise_rms = np.sqrt(np.mean(difference_data**2))

        # Compute SNR in dB
        if noise_rms > 0:
            return 20 * np.log10(signal_rms / noise_rms)
        else:
            return 0
    else:
        print("Fp1 or Fp2 not found in the data.")
        return None


def calculate_percent_difference_fp1_fp2(raw):
    """
    Calculates the average percent difference between Fp1 and Fp2 over the recording.

    Inputs:
    - raw (mne.io.Raw): Raw MNE object containing Fp1 and Fp2.

    Outputs:
    - float|None: Percent difference value or None if channels missing.
    """
    if 'Fp1' in raw.ch_names and 'Fp2' in raw.ch_names:
        fp1_idx = raw.ch_names.index('Fp1')
        fp2_idx = raw.ch_names.index('Fp2')

        # Extract data for Fp1 and Fp2
        fp1_data = raw.get_data(picks=fp1_idx)[0]
        fp2_data = raw.get_data(picks=fp2_idx)[0]

        # Calculate percent difference as mean(|Fp2 - Fp1|) / mean(|Fp2|) * 100
        # This avoids issues with point-wise division by small values
        mean_diff = np.mean(np.abs(fp2_data - fp1_data))
        mean_fp2 = np.mean(np.abs(fp2_data))
        
        if mean_fp2 > 0:
            percent_diff = (mean_diff / mean_fp2) * 100
        else:
            percent_diff = 0
            print("Warning: Mean of |Fp2| is zero, returning 0% difference")
        
        return percent_diff
    else:
        print("Fp1 or Fp2 not found in the data.")
        return None


def calculate_alpha_band_power(raw, fmin=8, fmax=13):
    """
    Computes the mean alpha (8-13 Hz) band power across channels.

    Inputs:
    - raw (mne.io.Raw): Raw MNE object.
    - fmin (float):     Minimum frequency for alpha band.
    - fmax (float):     Maximum frequency for alpha band.

    Outputs:
    - float: Mean alpha band power (V^2/Hz).
    """
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    alpha_mask = (freqs >= fmin) & (freqs <= fmax)
    alpha_power = np.mean(psds[:, alpha_mask])
    return alpha_power


def calculate_alpha_band_power_by_channel(raw, fmin=8, fmax=13, channels=('Fp1', 'Fp2')):
    """
    Computes the mean alpha band power for specified channel names.

    Inputs:
    - raw (mne.io.Raw): Raw MNE object.
    - fmin (float):     Minimum frequency for alpha band.
    - fmax (float):     Maximum frequency for alpha band.
    - channels (tuple): Channel names to evaluate.

    Outputs:
    - dict: Mapping channel name -> mean alpha band power (V^2/Hz).
    """
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    alpha_mask = (freqs >= fmin) & (freqs <= fmax)

    channel_powers = {}
    for ch_name in channels:
        if ch_name in raw.ch_names:
            ch_idx = raw.ch_names.index(ch_name)
            channel_powers[ch_name] = float(np.mean(psds[ch_idx, alpha_mask]))
        else:
            print(f"Warning: Channel {ch_name} not found for alpha power calculation.")
    return channel_powers


def plot_alpha_attenuation_summary(alpha_percent_diffs, out_dir):
    """
    Plots the combined alpha attenuation differences (closed vs open) per swatch.

    Inputs:
    - alpha_percent_diffs (list):   List of (swatch_label, percent_diff) tuples.
    - out_dir (str):                Directory to save the plot.

    Outputs:
    - matplotlib.figure.Figure|None: The generated bar plot or None if no data.
    """
    if not alpha_percent_diffs:
        print("\nNo alpha attenuation differences to plot.")
        return None

    # Sort by swatch label
    alpha_percent_diffs.sort(key=lambda x: x[0])
    labels = [f"Swatch {label}" for label, _ in alpha_percent_diffs]
    values = [val for _, val in alpha_percent_diffs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(labels, values, color='purple', alpha=0.8)
    ax.set_ylabel('Alpha Attenuation Difference (%)', fontsize=12)
    ax.set_title('Alpha Attenuation: Eyes Closed vs Eyes Open', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'alpha_attenuation_summary_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved alpha attenuation summary: {save_path}")
    return fig


def plot_alpha_attenuation_summary_by_channel(fp1_diffs, fp2_diffs, out_dir):
    """
    Plots the alpha attenuation per-channel (Fp1 vs Fp2) for each swatch.

    Inputs:
    - fp1_diffs (list): (swatch_label, percent_diff) tuples for Fp1.
    - fp2_diffs (list): (swatch_label, percent_diff) tuples for Fp2.
    - out_dir (str):    Directory to save the plot.

    Outputs:
    - matplotlib.figure.Figure|None: The generated grouped bar plot or None if no data.
    """
    if not fp1_diffs and not fp2_diffs:
        print("\nNo per-channel alpha attenuation differences to plot.")
        return None

    fp1_map = {label: val for label, val in fp1_diffs}
    fp2_map = {label: val for label, val in fp2_diffs}
    swatch_labels = sorted(set(fp1_map.keys()) | set(fp2_map.keys()))

    if not swatch_labels:
        print("\nNo swatch labels found for per-channel alpha attenuation plot.")
        return None

    labels = [f"Swatch {label}" for label in swatch_labels]
    fp1_values = [fp1_map.get(label, np.nan) for label in swatch_labels]
    fp2_values = [fp2_map.get(label, np.nan) for label in swatch_labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, fp1_values, width, color='blue', alpha=0.8, label=get_channel_display_name('Fp1'))
    ax.bar(x + width / 2, fp2_values, width, color='orange', alpha=0.8, label=get_channel_display_name('Fp2'))
    ax.set_ylabel('Alpha Attenuation Difference (%)', fontsize=12)
    ax.set_title('Alpha Attenuation by Channel: Eyes Closed vs Eyes Open', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'alpha_attenuation_by_channel_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved alpha attenuation by-channel summary: {save_path}")
    return fig


def plot_all_snrs(snr_values, file_names, out_dir, analysis_type=None):
    """
    Creates and saves a bar plot of SNR values for multiple files.

    Inputs:
    - snr_values (list):        SNR values per file.
    - file_names (list):        Corresponding file names.
    - out_dir (str):            Directory to save the plot.
    - analysis_type (str|None): Optional analysis type for label formatting.

    Outputs:
    - matplotlib.figure.Figure: The saved SNR bar plot.
    """
    print("\nPlotting SNR values for all files in the order of file names...")

    # Sort values by file names
    sorted_indices = np.argsort(file_names)
    sorted_file_names = [file_names[i] for i in sorted_indices]
    sorted_snr_values = [snr_values[i] for i in sorted_indices]
    
    # Format labels for blink analysis
    if analysis_type == 'blink':
        display_names = []
        for name in sorted_file_names:
            # Extract swatch number from filename (e.g., 'swatch1.1_blink.xdf' -> 'Swatch 1.1')
            if 'swatch' in name.lower():
                # Remove extension and _blink suffix
                base_name = name.replace('.xdf', '').replace('_blink', '')
                # Extract the number part (e.g., '1.1' from 'swatch1.1')
                number_part = base_name.lower().replace('swatch', '')
                display_names.append(f'Swatch {number_part}')
            else:
                display_names.append(name)
    else:
        display_names = sorted_file_names

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot SNR values
    ax.bar(display_names, sorted_snr_values, color='green', alpha=0.8)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title('SNR for All Files', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Save plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'snr_all_files_sorted_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved SNR plot for all files: {save_path}")

    return fig


def plot_all_percent_differences(percent_diff_values, file_names, out_dir, analysis_type=None):
    """
    Creates and saves a bar plot of percent differences between channels for multiple files.

    Inputs:
    - percent_diff_values (list):   Percent difference values per file.
    - file_names (list):            Corresponding file names.
    - out_dir (str):                Directory to save the plot.
    - analysis_type (str|None):     Optional analysis type for label formatting.

    Outputs:
    - matplotlib.figure.Figure: The saved percent-difference bar plot.
    """
    print("\nPlotting percent difference values for all files in the order of file names...")

    # Sort values by file names
    sorted_indices = np.argsort(file_names)
    sorted_file_names = [file_names[i] for i in sorted_indices]
    sorted_percent_diff_values = [percent_diff_values[i] for i in sorted_indices]
    
    # Format labels for blink analysis
    if analysis_type == 'blink':
        display_names = []
        for name in sorted_file_names:
            # Extract swatch number from filename (e.g., 'swatch1.1_blink.xdf' -> 'Swatch 1.1')
            if 'swatch' in name.lower():
                # Remove extension and _blink suffix
                base_name = name.replace('.xdf', '').replace('_blink', '')
                # Extract the number part (e.g., '1.1' from 'swatch1.1')
                number_part = base_name.lower().replace('swatch', '')
                display_names.append(f'Swatch {number_part}')
            else:
                display_names.append(name)
    else:
        display_names = sorted_file_names

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot percent difference values
    ax.bar(display_names, sorted_percent_diff_values, color='orange', alpha=0.8)
    ax.set_ylabel('Percent Difference (%)', fontsize=12)
    ax.set_title('Percent Difference Between Fp1 & Fp2', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'percent_diff_all_files_sorted_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved percent difference plot for all files: {save_path}")

    return fig


def main():
    """
    Command-line entry point: find XDF files, run analyses, and save results.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='XDF Folder Viewer - Analyze EEG data from all XDF files in a folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog= """
            Examples:
            python plot_xdf.py /path/to/folder                    # Analyze all XDF files in folder
            python plot_xdf.py /path/to/folder --analysis blink   # Include blink epoch analysis
            python plot_xdf.py /path/to/overall_dir --analysis alpha  # Subfolders: 1.1_eyes_open, 1.1_eyes_closed, ...
            """,
    )
    parser.add_argument('folder', nargs='?', help='Path to a folder containing XDF files or a single .xdf file')
    parser.add_argument('--analysis', choices=['blink', 'alpha'], 
                       help='Type of additional analysis to perform')

    args = parser.parse_args()

    # Get folder path from args or prompt user
    if args.folder:
        folder_path = args.folder
    else:
        print("="*60)
        print("XDF Folder Viewer")
        print("="*60)
        folder_path = input("\nEnter path to folder containing XDF files: ").strip()
        folder_path = folder_path.strip('"').strip("'")

    # Check if path exists
    if not os.path.exists(folder_path):
        print(f"\nError: Path not found: {folder_path}")
        return

    # Build the work queue from a single file, a folder of files, or alpha subfolders.
    xdf_jobs = []
    input_is_file = os.path.isfile(folder_path) and folder_path.lower().endswith('.xdf')

    if input_is_file:
        xdf_jobs.append({
            'xdf_file': os.path.basename(folder_path),
            'xdf_file_path': folder_path,
            'subfolder_name': None
        })
    elif args.analysis == 'alpha':
        subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        if not subfolders:
            print(f"\nNo subfolders found in folder: {folder_path}")
            return
        for subfolder in sorted(subfolders):
            sub_path = os.path.join(folder_path, subfolder)
            sub_xdfs = [f for f in os.listdir(sub_path) if f.endswith('.xdf')]
            if not sub_xdfs:
                print(f"\nNo .xdf files found in subfolder: {sub_path}")
                continue
            for xdf_file in sorted(sub_xdfs):
                xdf_jobs.append({
                    'xdf_file': xdf_file,
                    'xdf_file_path': os.path.join(sub_path, xdf_file),
                    'subfolder_name': subfolder
                })
        if not xdf_jobs:
            print(f"\nNo .xdf files found in any subfolder of: {folder_path}")
            return
    else:
        xdf_files = [f for f in os.listdir(folder_path) if f.endswith('.xdf')]
        if not xdf_files:
            print(f"\nNo .xdf files found in folder: {folder_path}")
            return
        for xdf_file in sorted(xdf_files):
            xdf_jobs.append({
                'xdf_file': xdf_file,
                'xdf_file_path': os.path.join(folder_path, xdf_file),
                'subfolder_name': None
            })

    is_single_file_run = len(xdf_jobs) == 1 and input_is_file

    # Collect per-file metrics so we can generate combined summary plots later.
    all_snr_values = []  # Fp2 SNR
    all_snr_values_fp1 = []  # Fp1 SNR
    all_percent_diff_values = []
    file_names = []
    file_names_fp1 = []
    main_results_dir = None  # Store the main directory for saving overall plots
    alpha_group_powers = {}  # Track open/closed alpha power per swatch and channel.
    
    # Track global PSD bounds so every file uses the same PSD scale.
    global_psd_min = float('inf')
    global_psd_max = float('-inf')
    raw_objects = []  # Keep filtered raws for the plot generation pass.

    # First pass: load each file, compute shared metrics, and cache the filtered data.
    for job in xdf_jobs:
        xdf_file = job['xdf_file']
        xdf_file_path = job['xdf_file_path']
        subfolder_name = job['subfolder_name']
        print(f"\nProcessing file: {xdf_file_path}")

        try:
            # Setup output directory (now based on filename and analysis type)
            out_dir, main_dir = setup_output_directory(xdf_file_path, args.analysis, subfolder_name)
            if main_results_dir is None:
                main_results_dir = main_dir

            # Keep the full recording intact for single-file runs.
            cut_blink = args.analysis == 'blink' and not is_single_file_run
            cut_start_seconds = 5 if args.analysis == 'alpha' and not is_single_file_run else 0

            # Load data with optional cutting
            raw = load_xdf_to_mne(xdf_file_path, cut_blink=cut_blink, cut_start_seconds=cut_start_seconds)

            # Apply filter
            print(f"\nApplying bandpass filter (1-50 Hz)...")
            raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=50.0, verbose=False)
            
            # Calculate PSD for global range
            spectrum = raw_filtered.compute_psd(fmin=1, fmax=50, verbose=False)
            psds, freqs = spectrum.get_data(return_freqs=True)
            global_psd_min = min(global_psd_min, np.min(psds[psds > 0]))
            global_psd_max = max(global_psd_max, np.max(psds))
            
            # Store for second pass
            raw_objects.append({
                'raw_filtered': raw_filtered,
                'out_dir': out_dir,
                'xdf_file': xdf_file,
                'xdf_file_path': xdf_file_path
            })

            # For alpha analysis, compute alpha band power per subfolder
            if args.analysis == 'alpha' and subfolder_name:
                subfolder_lower = subfolder_name.lower()
                state = None
                swatch_key = None
                if '_eyes_open' in subfolder_lower:
                    state = 'open'
                    swatch_key = subfolder_lower.replace('_eyes_open', '')
                elif '_eyes_closed' in subfolder_lower:
                    state = 'closed'
                    swatch_key = subfolder_lower.replace('_eyes_closed', '')

                if state and swatch_key:
                    alpha_power_combined = calculate_alpha_band_power(raw_filtered)
                    alpha_power_by_channel = calculate_alpha_band_power_by_channel(raw_filtered)

                    if swatch_key not in alpha_group_powers:
                        alpha_group_powers[swatch_key] = {
                            'combined': {'open': [], 'closed': []},
                            'fp1': {'open': [], 'closed': []},
                            'fp2': {'open': [], 'closed': []}
                        }

                    alpha_group_powers[swatch_key]['combined'][state].append(alpha_power_combined)

                    if 'Fp1' in alpha_power_by_channel:
                        alpha_group_powers[swatch_key]['fp1'][state].append(alpha_power_by_channel['Fp1'])
                    if 'Fp2' in alpha_power_by_channel:
                        alpha_group_powers[swatch_key]['fp2'][state].append(alpha_power_by_channel['Fp2'])

        except Exception as e:
            print(f"\nError loading file {xdf_file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Second pass: generate plots after the shared PSD limits are known.
    if not raw_objects:
        print("\nNo files were processed successfully. Nothing to plot.")
        return

    print(f"\n\nCalculated global PSD range: {global_psd_min:.2e} to {global_psd_max:.2e}")
    psd_ylim = (global_psd_min * 0.5, global_psd_max * 2)  # Add some margin
    
    for obj in raw_objects:
        raw_filtered = obj['raw_filtered']
        out_dir = obj['out_dir']
        xdf_file = obj['xdf_file']
        
        print(f"\n{'='*60}")
        print(f"Processing: {xdf_file}")
        print(f"{'='*60}")
        
        try:
            # Print summary
            print_info(raw_filtered)

            # Create all analyses
            print("\n" + "="*60)
            print("GENERATING COMPREHENSIVE ANALYSIS")
            print("="*60)

            # 1. Run ICA analysis
            ica = run_ica_analysis(raw_filtered, out_dir, n_components=2)

            # 2. Create standalone PSD summary with consistent y-axis
            create_standalone_psd_summary(
                raw_filtered,
                out_dir,
                psd_ylim=psd_ylim,
                highlight_alpha=(args.analysis == 'alpha')
            )

            # 3. Create standalone ICA summary
            create_standalone_ica_summary(raw_filtered, ica, out_dir)

            # 4. Create standalone raw EEG plot (30-45s window)
            plot_cut_start = 10 if args.analysis == 'blink' and not is_single_file_run else 0
            plot_cut_end = 10 if args.analysis == 'blink' and not is_single_file_run else 0
            create_standalone_raw_eeg(raw_filtered, out_dir, start_time=30, end_time=45, cut_start=plot_cut_start, cut_end=plot_cut_end)

            # 5. Create comprehensive summary (Raw data + Frequency bands + Variability + SNR)
            create_comprehensive_summary(raw_filtered, ica, out_dir)

            # 6. Optional: Blink epoch analysis
            if args.analysis == 'blink':
                blink_channel_indices = (3, 4) if is_single_file_run else None
                create_blink_epoch_analysis(raw_filtered, out_dir, channel_indices=blink_channel_indices)
            
            # Optional: Alpha wave analysis
            if args.analysis == 'alpha':
                alpha_channel_indices = (3, 4) if is_single_file_run else None
                create_alpha_wave_analysis(raw_filtered, out_dir, channel_indices=alpha_channel_indices)

                if is_single_file_run:
                    create_alpha_time_frequency_power(raw_filtered, out_dir, channel_indices=(3, 4))

            # 7. Calculate SNR between Fp2 and the difference (Fp2 - Fp1)
            if args.analysis == 'blink':
                snr_fp2_diff = calculate_snr_fp2_over_difference(raw_filtered)
                if snr_fp2_diff is not None:
                    all_snr_values.append(snr_fp2_diff)
                    file_names.append(xdf_file)
                
                # Calculate SNR for Fp1 as well
                snr_fp1_diff = calculate_snr_fp1_over_difference(raw_filtered)
                if snr_fp1_diff is not None:
                    all_snr_values_fp1.append(snr_fp1_diff)
                    file_names_fp1.append(xdf_file)

                # 8. Calculate average percent difference between Fp1 and Fp2
                percent_diff = calculate_percent_difference_fp1_fp2(raw_filtered)
                if percent_diff is not None:
                    all_percent_diff_values.append(percent_diff)

            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            print(f"\nAll plots saved to: {os.path.abspath(out_dir)}")

        except Exception as e:
            print(f"\nError processing file {xdf_file_path}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save combined SNR summaries only after all per-file blink results are collected.
    if not is_single_file_run and args.analysis == 'blink' and all_snr_values and main_results_dir:
        plot_all_snrs(all_snr_values, file_names, main_results_dir, args.analysis)
    
    # Repeat the combined SNR summary for Fp1 so both channels are reported together.
    if not is_single_file_run and args.analysis == 'blink' and all_snr_values_fp1 and main_results_dir:
        # Plot SNR values for Fp1 (Textile Electrode)
        print("\nPlotting SNR values for Fp1 (Textile Electrode) across all files...")

        sorted_indices = np.argsort(file_names_fp1)
        sorted_file_names = [file_names_fp1[i] for i in sorted_indices]
        sorted_snr_values = [all_snr_values_fp1[i] for i in sorted_indices]

        # Format labels for blink analysis
        display_names = []
        for name in sorted_file_names:
            if 'swatch' in name.lower():
                parts = name.lower().replace('_blink.xdf', '').replace('swatch', '').strip()
                display_names.append(f'Swatch {parts}')
            else:
                display_names.append(name.replace('.xdf', ''))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(display_names, sorted_snr_values, color='blue', alpha=0.8)
        ax.set_ylabel('SNR (dB)', fontsize=12)
        ax.set_title('SNR for All Files - Fp1 (Textile Electrode)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(display_names)))
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(main_results_dir, f'snr_all_files_fp1_{timestamp}.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved Fp1 SNR summary: {save_path}")
    
    # Plot the channel difference summary only once all files have been processed.
    if not is_single_file_run and args.analysis == 'blink' and all_percent_diff_values and main_results_dir:
        plot_all_percent_differences(all_percent_diff_values, file_names, main_results_dir, args.analysis)

    # Alpha mode aggregates open/closed recordings by swatch before building summary plots.
    if not is_single_file_run and args.analysis == 'alpha' and main_results_dir:
        alpha_percent_diffs = []
        alpha_percent_diffs_fp1 = []
        alpha_percent_diffs_fp2 = []

        for swatch_key, group in alpha_group_powers.items():
            # Combined (all channels)
            combined = group.get('combined', {})
            if combined.get('open') and combined.get('closed'):
                mean_open = np.mean(combined['open'])
                mean_closed = np.mean(combined['closed'])
                if mean_closed > 0:
                    percent_diff = ((mean_closed - mean_open) / mean_closed) * 100
                    alpha_percent_diffs.append((swatch_key, percent_diff))

            # Fp1 only
            fp1 = group.get('fp1', {})
            if fp1.get('open') and fp1.get('closed'):
                mean_open = np.mean(fp1['open'])
                mean_closed = np.mean(fp1['closed'])
                if mean_closed > 0:
                    percent_diff = ((mean_closed - mean_open) / mean_closed) * 100
                    alpha_percent_diffs_fp1.append((swatch_key, percent_diff))

            # Fp2 only
            fp2 = group.get('fp2', {})
            if fp2.get('open') and fp2.get('closed'):
                mean_open = np.mean(fp2['open'])
                mean_closed = np.mean(fp2['closed'])
                if mean_closed > 0:
                    percent_diff = ((mean_closed - mean_open) / mean_closed) * 100
                    alpha_percent_diffs_fp2.append((swatch_key, percent_diff))

        # Combined plot (existing behavior)
        plot_alpha_attenuation_summary(alpha_percent_diffs, main_results_dir)

        # Per-channel split plot (Fp1 vs Fp2)
        plot_alpha_attenuation_summary_by_channel(
            alpha_percent_diffs_fp1,
            alpha_percent_diffs_fp2,
            main_results_dir
        )
    
    # Close all matplotlib figures and exit
    plt.close('all')

if __name__ == "__main__":
    main()
