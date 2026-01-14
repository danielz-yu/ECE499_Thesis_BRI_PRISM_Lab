"""
Simple XDF File Viewer
Reads and plots EEG data from XDF files using MNE.

Usage:
    python plot_xdf.py /path/to/your/folder
    python plot_xdf.py /path/to/your/folder --analysis blink
    
    Or run interactively and enter path when prompted.
    
Options:
    --analysis blink    Perform blink epoch analysis (ERP, amplitude comparison, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from pyxdf import load_xdf
import sys
import os
import argparse
from datetime import datetime
from scipy import signal

# Enable interactive plotting
plt.ion()


def setup_output_directory(xdf_file_path):
    """Create output directory for results based on XDF filename."""
    # Get the XDF filename without extension
    xdf_basename = os.path.splitext(os.path.basename(xdf_file_path))[0]
    
    # Create main artifact_results folder
    main_dir = 'artifact_results'
    if not os.path.isdir(main_dir):
        os.makedirs(main_dir)
    
    # Create subfolder for this specific file
    out_dir = os.path.join(main_dir, xdf_basename)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print(f"Created output directory: {out_dir}")
    else:
        print(f"Using existing directory: {out_dir}")
    
    return out_dir


def load_xdf_to_mne(xdf_file_path, cut_blink=False):
    """
    Load XDF file and convert to MNE Raw object.

    Parameters
    ----------
    xdf_file_path : str
        Path to the XDF file
    cut_blink : bool
        Whether to cut the first and last 10 seconds of the raw EEG data

    Returns
    -------
    raw : mne.io.RawArray
        MNE Raw object containing EEG data
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
        ch_info = eeg_stream['info']['desc'][0]['channels'][0]['channel']
        ch_names = [ch['label'][0] for ch in ch_info]
    except (KeyError, IndexError):
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
    """Print detailed information about the loaded data."""
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
    Run Independent Component Analysis (ICA) on the data.
    
    Parameters
    ----------
    raw : mne.io.RawArray
        MNE Raw object
    out_dir : str
        Output directory
    n_components : int
        Number of ICA components
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
    Create a comprehensive summary figure with additional analyses NOT in PSD/ICA summaries.
    This includes: Full raw data, frequency band power, and channel-wise variability.
    
    Parameters
    ----------
    raw : mne.io.RawArray
        MNE Raw object
    ica : mne.preprocessing.ICA
        Fitted ICA object (not used, kept for compatibility)
    out_dir : str
        Output directory
    fmin : float
        Minimum frequency for PSD (Hz)
    fmax : float
        Maximum frequency for PSD (Hz)
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
        ax_bands.bar(x + offset, band_power[ch_name], width, 
                    label=ch_name, alpha=0.8, color=color)
    
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


def create_standalone_psd_summary(raw, out_dir, fmin=1, fmax=50):
    """
    Create a standalone PSD summary figure.
    
    Parameters
    ----------
    raw : mne.io.RawArray
        MNE Raw object
    out_dir : str
        Output directory
    fmin : float
        Minimum frequency for PSD (Hz)
    fmax : float
        Maximum frequency for PSD (Hz)
    """
    print("\nCreating standalone PSD summary...")
    
    # Compute PSD
    spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
    
    colors = ['blue', 'orange']
    
    # ============================================================
    # 1. Main PSD Plot (Top Left)
    # ============================================================
    ax_psd = fig.add_subplot(gs[0, 0])
    
    for i, ch_name in enumerate(raw.ch_names):
        color = colors[i] if i < len(colors) else f'C{i}'
        ax_psd.semilogy(freqs, psds[i], label=ch_name, color=color, linewidth=2.5, alpha=0.8)
    
    ax_psd.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_psd.set_ylabel('Power Spectral Density (V²/Hz)', fontsize=12)
    ax_psd.set_title('Power Spectral Density', fontsize=13, fontweight='bold')
    ax_psd.grid(True, alpha=0.3)
    ax_psd.legend(fontsize=11, loc='upper right')
    ax_psd.set_xlim(fmin, fmax)
    
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
    band_power = {ch: [] for ch in raw.ch_names}
    
    for band_name, (fmin_band, fmax_band) in bands.items():
        for i, ch_name in enumerate(raw.ch_names):
            freq_mask = (freqs >= fmin_band) & (freqs <= fmax_band)
            power = np.mean(psds[i, freq_mask])
            band_power[ch_name].append(power)
    
    # Plot
    x = np.arange(len(band_names))
    width = 0.35
    
    for i, ch_name in enumerate(raw.ch_names):
        offset = (i - len(raw.ch_names)/2 + 0.5) * width
        color = colors[i] if i < len(colors) else f'C{i}'
        ax_bands.bar(x + offset, band_power[ch_name], width, 
                    label=ch_name, alpha=0.8, color=color)
    
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
    
    for i, ch_name in enumerate(raw.ch_names):
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
    Create a standalone ICA summary figure.
    
    Parameters
    ----------
    raw : mne.io.RawArray
        MNE Raw object
    ica : mne.preprocessing.ICA
        Fitted ICA object
    out_dir : str
        Output directory
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
    Create a standalone raw EEG recording plot for a specific time window, excluding cut-off portions.

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE Raw object
    out_dir : str
        Output directory
    start_time : float
        Start time in seconds (default: 30)
    end_time : float
        End time in seconds (default: 45)
    cut_start : float
        Time in seconds to exclude from the start of the recording (default: 0)
    cut_end : float
        Time in seconds to exclude from the end of the recording (default: 0)
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
        ax.plot(times_window, data_window[i, :] * 1e6 + offset, 
                label=ch_name, color=color, linewidth=1.2, alpha=0.8)

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


def create_blink_epoch_analysis(raw, out_dir):
    """
    Detect blinks and create epoch-based analysis.
    
    This provides:
    - Event-Related Potentials (ERPs): Average response to blinks
    - Peak amplitude analysis: How strong are individual blinks
    - Consistency analysis: Do all blinks look similar
    - Comparison of blink amplitudes between Fp1 and Fp2
    - Blink detection sensitivity for each electrode
    
    Parameters
    ----------
    raw : mne.io.RawArray
        MNE Raw object
    out_dir : str
        Output directory
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with epoch analysis
    epochs : mne.Epochs or None
        Epochs object if blinks detected, None otherwise
    """
    print("\n" + "="*60)
    print("BLINK EPOCH ANALYSIS")
    print("="*60)
    print("\nDetecting blink events...")
    
    # Detect EOG events (blinks) using Fp1 as reference
    try:
        eog_events = mne.preprocessing.find_eog_events(
            raw, 
            ch_name='Fp1',  # Use Fp1 for detection
            event_id=998,
            l_freq=1,
            h_freq=15,
            thresh=300e-6  # 75µV threshold (adjust if needed)
        )
    except Exception as e:
        print(f"  Error detecting blinks: {e}")
        print("  Trying alternative detection method...")
        
        # Alternative: use simple threshold detection
        data_fp1 = raw.get_data(picks='Fp1')[0]
        filtered = mne.filter.filter_data(data_fp1, raw.info['sfreq'], l_freq=1, h_freq=15, verbose=False)
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
    
    colors = {'Fp1': 'blue', 'Fp2': 'orange'}
    
    # Get epoch data
    data_fp1 = epochs.get_data(picks='Fp1')[:, 0, :] * 1e6  # Convert to µV
    data_fp2 = epochs.get_data(picks='Fp2')[:, 0, :] * 1e6
    times = epochs.times
    
    # ============================================================
    # 1. AVERAGE BLINK WAVEFORM (ERP)
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    evoked_fp1 = epochs.average(picks='Fp1')
    evoked_fp2 = epochs.average(picks='Fp2')
    
    ax1.plot(times, evoked_fp1.data[0] * 1e6, 'b-', linewidth=2.5, label='Fp1 (Blue)')
    ax1.plot(times, evoked_fp2.data[0] * 1e6, color='orange', linewidth=2.5, label='Fp2 (Orange)')
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
    ax2.set_title(f'Fp1: Individual Blinks (showing {n_show}/{len(epochs)})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ============================================================
    # 3. PEAK AMPLITUDE DISTRIBUTION
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    peak_fp1 = np.max(np.abs(data_fp1), axis=1)
    peak_fp2 = np.max(np.abs(data_fp2), axis=1)
    
    ax3.hist(peak_fp1, bins=20, alpha=0.6, color='blue', label='Fp1', edgecolor='black', linewidth=1.2)
    ax3.hist(peak_fp2, bins=20, alpha=0.6, color='orange', label='Fp2', edgecolor='black', linewidth=1.2)
    
    # Add mean lines
    ax3.axvline(np.mean(peak_fp1), color='blue', linestyle='--', linewidth=2.5, 
               label=f'Fp1 mean: {np.mean(peak_fp1):.1f}µV')
    ax3.axvline(np.mean(peak_fp2), color='orange', linestyle='--', linewidth=2.5, 
               label=f'Fp2 mean: {np.mean(peak_fp2):.1f}µV')
    
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
    
    ax4.set_xlabel('Fp1 Peak Amplitude (µV)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Fp2 Peak Amplitude (µV)', fontsize=12, fontweight='bold')
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
    Calculate the SNR between Fp2 and the difference (Fp2 - Fp1) averaged across the entire recording.

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE Raw object

    Returns
    -------
    float
        The calculated SNR in dB, or None if Fp1 or Fp2 is not found.
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


def calculate_percent_difference_fp1_fp2(raw):
    """
    Calculate the average percent difference between Fp1 and Fp2 across the entire recording.

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE Raw object

    Returns
    -------
    float
        The calculated average percent difference, or None if Fp1 or Fp2 is not found.
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


def plot_all_snrs(snr_values, file_names, out_dir):
    """
    Plot the SNR values for all analyzed XDF files in the order of file names.

    Parameters
    ----------
    snr_values : list of float
        List of SNR values for each file.
    file_names : list of str
        List of file names corresponding to the SNR values.
    out_dir : str
        Output directory to save the plot.
    """
    print("\nPlotting SNR values for all files in the order of file names...")

    # Sort values by file names
    sorted_indices = np.argsort(file_names)
    sorted_file_names = [file_names[i] for i in sorted_indices]
    sorted_snr_values = [snr_values[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot SNR values
    ax.bar(sorted_file_names, sorted_snr_values, color='green', alpha=0.8)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title('SNR for All Files', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(sorted_file_names)))
    ax.set_xticklabels(sorted_file_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'snr_all_files_sorted_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved SNR plot for all files: {save_path}")

    return fig


def plot_all_percent_differences(percent_diff_values, file_names, out_dir):
    """
    Plot the percent difference values for all analyzed XDF files in the order of file names.

    Parameters
    ----------
    percent_diff_values : list of float
        List of percent difference values for each file.
    file_names : list of str
        List of file names corresponding to the percent difference values.
    out_dir : str
        Output directory to save the plot.
    """
    print("\nPlotting percent difference values for all files in the order of file names...")

    # Sort values by file names
    sorted_indices = np.argsort(file_names)
    sorted_file_names = [file_names[i] for i in sorted_indices]
    sorted_percent_diff_values = [percent_diff_values[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot percent difference values
    ax.bar(sorted_file_names, sorted_percent_diff_values, color='orange', alpha=0.8)
    ax.set_ylabel('Percent Difference (%)', fontsize=12)
    ax.set_title('Percent Difference Between Fp1 & Fp2', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(sorted_file_names)))
    ax.set_xticklabels(sorted_file_names, rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(out_dir, f'percent_diff_all_files_sorted_{timestamp}.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved percent difference plot for all files: {save_path}")

    return fig


def main():
    """Main function with interactive mode."""
    # Parse code
    parser = argparse.ArgumentParser(
        description='XDF Folder Viewer - Analyze EEG data from all XDF files in a folder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_xdf.py /path/to/folder                    # Analyze all XDF files in folder
  python plot_xdf.py /path/to/folder --analysis blink   # Include blink epoch analysis
        """
    )
    parser.add_argument('folder', nargs='?', help='Path to folder containing XDF files')
    parser.add_argument('--analysis', choices=['blink'], 
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

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"\nError: Folder not found: {folder_path}")
        return

    # Get all .xdf files in the folder
    xdf_files = [f for f in os.listdir(folder_path) if f.endswith('.xdf')]
    if not xdf_files:
        print(f"\nNo .xdf files found in folder: {folder_path}")
        return

    # Initialize SNR values list for all files
    all_snr_values = []
    all_percent_diff_values = []
    file_names = []

    # Process each .xdf file
    for xdf_file in xdf_files:
        xdf_file_path = os.path.join(folder_path, xdf_file)
        print(f"\nProcessing file: {xdf_file_path}")

        try:
            # Setup output directory (now based on filename)
            out_dir = setup_output_directory(xdf_file_path)

            # Determine if blink analysis is requested
            cut_blink = args.analysis == 'blink'

            # Load data with optional cutting
            raw = load_xdf_to_mne(xdf_file_path, cut_blink=cut_blink)

            # Apply filter
            print(f"\nApplying bandpass filter (1-50 Hz)...")
            raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=50.0, verbose=False)

            # Print summary
            print_info(raw_filtered)

            # Create all analyses
            print("\n" + "="*60)
            print("GENERATING COMPREHENSIVE ANALYSIS")
            print("="*60)

            # 1. Run ICA analysis
            ica = run_ica_analysis(raw_filtered, out_dir, n_components=2)

            # 2. Create standalone PSD summary
            create_standalone_psd_summary(raw_filtered, out_dir)

            # 3. Create standalone ICA summary
            create_standalone_ica_summary(raw_filtered, ica, out_dir)

            # 4. Create standalone raw EEG plot (30-45s window)
            create_standalone_raw_eeg(raw_filtered, out_dir, start_time=30, end_time=45, cut_start=10, cut_end=10)

            # 5. Create comprehensive summary (Raw data + Frequency bands + Variability + SNR)
            create_comprehensive_summary(raw_filtered, ica, out_dir)

            # 6. Optional: Blink epoch analysis
            if args.analysis == 'blink':
                create_blink_epoch_analysis(raw_filtered, out_dir)

            # 7. Calculate SNR between Fp2 and the difference (Fp2 - Fp1)
            snr_fp2_diff = calculate_snr_fp2_over_difference(raw_filtered)
            if snr_fp2_diff is not None:
                all_snr_values.append(snr_fp2_diff)
                file_names.append(xdf_file)

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

    # After processing all files, plot all SNR values
    if all_snr_values:
        plot_all_snrs(all_snr_values, file_names, folder_path)
    
    # After processing all files, plot all percent difference values
    if all_percent_diff_values:
        plot_all_percent_differences(all_percent_diff_values, file_names, folder_path)

if __name__ == "__main__":
    main()
