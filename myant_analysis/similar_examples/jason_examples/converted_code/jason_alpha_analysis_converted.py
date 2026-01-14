"""
TEST CODE – CONVERTED AND USED FOR ENVIRONMENT SETUP ONLY
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import pickle
import csv
import json
from pyxdf import load_xdf
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp

# Set matplotlib backend for interactive plotting
plt.switch_backend('Qt5Agg')

def create_output_directory(out_dir='alpha_results'):
    """Create output directory if it doesn't exist"""
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return out_dir

def find_xdf_files(input_dir):
    """Find all XDF files containing 'eyes' in participant folders"""
    xdf_files = {}
    participant_folders = [f for f in os.listdir(input_dir) 
                          if f.startswith('Participant') and f[-1].isdigit()]
    
    for participant_folder in participant_folders:
        sourcedata_path = os.path.join(input_dir, participant_folder, 'sourcedata')
        if os.path.exists(sourcedata_path):
            xdf_files[participant_folder] = [
                f for f in os.listdir(sourcedata_path) 
                if 'eyes' in f and f.endswith('.xdf')
            ]
    
    return xdf_files

def load_and_process_streams(input_dir, participant, xdf_files, epoch_twin_size=5):
    """Load XDF streams and process EEG data"""
    print(f'Parsing XDF files for {participant}')
    
    streams = {}
    for xdf_file in xdf_files[participant]:
        print(f'File: {xdf_file}')
        
        # Load LSL streams from xdf
        found_streams, _ = load_xdf(os.path.join(input_dir, participant, 'sourcedata', xdf_file))
        for found_stream in found_streams:
            stream_name = found_stream['info']['name'][0]
            if stream_name not in streams:
                streams[stream_name] = []
            streams[stream_name].append(found_stream)
    
    return streams

def create_raw_data(streams, epoch_twin_size=5):
    """Create MNE Raw objects from streams"""
    eeg_raw_list, raw = {}, {}
    
    for stream in streams:
        # Ignore marker stream
        if stream == 'mindset_Marker_PRISM-DIY-MSI-4':
            continue

        eeg_raw_list[stream] = []
        
        # Load EEG data
        for i in range(len(streams[stream])):
            print(f'Processing {stream} stream')
            
            # Get EEG info
            if stream == 'BrainAmpSeries-Dev_1':
                sfreq = float(streams[stream][i]['info']['nominal_srate'][0])
                ch_names = [ch['label'][0] for ch in streams[stream][i]['info']['desc'][0]['channels'][0]['channel']]
                montage_file = f'{stream}_montage.elc'
            elif stream == 'InEarEEG':
                sfreq = float(streams[stream][i]['info']['effective_srate'])
                ch_names = ['A1', 'A2']
                montage_file = f'{stream}_montage.elc'
            
            eeg_info = mne.create_info(ch_names, sfreq, ch_types='eeg')
            
            # Load montage if file exists
            if os.path.exists(montage_file):
                montage = mne.channels.read_custom_montage(montage_file)
            else:
                montage = None

            # Apply scaling factor to EEG data
            if stream == 'InEarEEG':
                eeg_data = np.transpose(streams[stream][i]['time_series']) / 24 / 4.1887e9 * 4.5
            elif stream == 'BrainAmpSeries-Dev_1':
                eeg_data = np.transpose(streams[stream][i]['time_series']) / 1e6
            
            eeg_raw = mne.io.RawArray(eeg_data, eeg_info, verbose=False)
            if montage:
                eeg_raw.set_montage(montage)

            # Create marker annotations
            onset, duration, description = [], [], []
            markers = streams['mindset_Marker_PRISM-DIY-MSI-4'][i]['time_series']
            markers_ts = streams['mindset_Marker_PRISM-DIY-MSI-4'][i]['time_stamps']
            
            for m in range(len(markers)):
                marker = json.loads(markers[m][0])['status']
                if 'eyes' in marker:
                    # Extract overlapping 5s epochs with 1s stride from 15s trials
                    start = np.arange(0, 15-epoch_twin_size+1, epoch_twin_size) + markers_ts[m] - streams[stream][i]['time_stamps'][0]
                    for s in start:
                        onset.append(s)
                        duration.append(epoch_twin_size)
                        description.append(marker)
            
            annotations = mne.Annotations(onset, duration, description)
            eeg_raw = eeg_raw.set_annotations(annotations)
            
            if stream == 'InEarEEG':
                eeg_raw = eeg_raw.resample(250)

            eeg_raw_list[stream].append(eeg_raw)

        # Concatenate all epochs and filter
        raw[stream] = mne.concatenate_raws(eeg_raw_list[stream])
        raw[stream] = raw[stream].filter(l_freq=1.0, h_freq=40.0)

    return raw

def create_epochs(raw):
    """Create epochs from raw data"""
    epochs = {}
    for stream in raw:
        events, event_id = mne.events_from_annotations(raw[stream])
        epochs[stream] = mne.Epochs(
            raw[stream], events, event_id=event_id, preload=True,
            tmin=0., tmax=5, baseline=None
        )
    return epochs

def compute_psd(epochs, psd_method='multitaper'):
    """Compute Power Spectral Density"""
    psds = {}
    for headset in epochs:
        psds[headset] = epochs[headset].compute_psd(
            method=psd_method, fmin=1, fmax=40, normalization='full'
        )
    return psds

def plot_psd(psds, participant_num, out_dir, plot_average=False, dB=True):
    """Plot PSD for all conditions"""
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    title = f'Participant {participant_num} Alpha Wave Modulation Power Spectral Density'
    fig.suptitle(title)

    # Plot PSDs
    psds['BrainAmpSeries-Dev_1']['eyes open'].plot(axes=ax[0,0], average=plot_average, dB=dB)
    psds['BrainAmpSeries-Dev_1']['eyes closed'].plot(axes=ax[0,1], average=plot_average, dB=dB)
    psds['BrainAmpSeries-Dev_1']['eyes open'].pick(['T7','T8']).plot(axes=ax[1,0], average=plot_average, dB=dB)
    psds['BrainAmpSeries-Dev_1']['eyes closed'].pick(['T7','T8']).plot(axes=ax[1,1], average=plot_average, dB=dB)
    psds['InEarEEG']['eyes open'].plot(axes=ax[2,0], average=plot_average, dB=dB)
    psds['InEarEEG']['eyes closed'].plot(axes=ax[2,1], average=plot_average, dB=dB)

    # Set labels and titles
    ax[0,0].set_xlabel('')
    ax[0,1].set_xlabel('')
    ax[1,0].set_xlabel('')
    ax[1,1].set_xlabel('')
    ax[2,0].set_xlabel('Frequency (Hz)')
    ax[2,1].set_xlabel('Frequency (Hz)')

    ax[0,0].set_title('Eyes Open')
    ax[0,1].set_title('Eyes Closed')
    for i in range(1, 3):
        ax[i,0].set_title('')
        ax[i,1].set_title('')

    ax[0,0].set_ylabel('Scalp EEG\nPower (dB µV²/Hz)')
    ax[1,0].set_ylabel('Temporal EEG\nPower (dB µV²/Hz)')
    ax[2,0].set_ylabel('In-Ear EEG\nPower (dB µV²/Hz)')
    ax[0,1].set_ylabel('')
    ax[1,1].set_ylabel('')
    ax[2,1].set_ylabel('')

    # Synchronize y-axes
    scalp_ylim = [min(ax[0,0].get_ylim()[0], ax[0,1].get_ylim()[0]), 
                  max(ax[0,0].get_ylim()[1], ax[0,1].get_ylim()[1])]
    temporal_ylim = [min(ax[1,0].get_ylim()[0], ax[1,1].get_ylim()[0]), 
                     max(ax[1,0].get_ylim()[1], ax[1,1].get_ylim()[1])]
    inear_ylim = [min(ax[2,0].get_ylim()[0], ax[2,1].get_ylim()[0]), 
                  max(ax[2,0].get_ylim()[1], ax[2,1].get_ylim()[1])]

    ax[0,0].set_ylim(scalp_ylim)
    ax[0,1].set_ylim(scalp_ylim)
    ax[1,0].set_ylim(temporal_ylim)
    ax[1,1].set_ylim(temporal_ylim)
    ax[2,0].set_ylim(inear_ylim)
    ax[2,1].set_ylim(inear_ylim)

    # Customize colors for temporal electrodes
    for a in [ax[1,0], ax[1,1]]:
        for line in a.get_lines():
            if (line.get_color() == np.array([[0., 0., 1.]])).all():
                line.set_color(np.array([0., 0.549, 0.]))
            if (line.get_color() == np.array([1., 1., 0.])).all():
                line.set_color(np.array([1., 0.5529, 0.]))

    plt.tight_layout()
    plt.show()

    # Save plot
    plt.savefig(os.path.join(out_dir, f'{title}.png'))
    return fig

def compute_ram(power_closed, power_open):
    """Compute Relative Alpha Modulation (RAM)"""
    power_open = np.mean(np.mean(np.max(power_open, axis=-1), axis=-1))
    power_closed = np.mean(np.max(power_closed, axis=-1), axis=-1)
    return np.mean(power_closed / power_open)

def compute_alpha_modulation(psds, fmin=7, fmax=15):
    """Compute alpha modulation ratios"""
    scalings = mne.defaults._handle_default("scalings", None)['eeg']
    power = {'scalp': {}, 'temporal': {}, 'ear': {}}
    
    for task in ['eyes open', 'eyes closed']:
        # Get PSD in alpha frequency range
        power['scalp'][task] = psds['BrainAmpSeries-Dev_1'][task].get_data(fmin=fmin, fmax=fmax)
        power['temporal'][task] = psds['BrainAmpSeries-Dev_1'][task].get_data(picks=['T7','T8'], fmin=fmin, fmax=fmax)
        power['ear'][task] = psds['InEarEEG'][task].get_data(fmin=fmin, fmax=fmax)

        # Scale power to µV²
        power['scalp'][task] = power['scalp'][task] * scalings * scalings
        power['temporal'][task] = power['temporal'][task] * scalings * scalings
        power['ear'][task] = power['ear'][task] * scalings * scalings

    # Compute RAM
    ram_data = {
        'Scalp RAM': compute_ram(power['scalp']['eyes closed'], power['scalp']['eyes open']),
        'Temporal RAM': compute_ram(power['temporal']['eyes closed'], power['temporal']['eyes open']),
        'In-Ear RAM': compute_ram(power['ear']['eyes closed'], power['ear']['eyes open']),
    }
    
    return ram_data, power

def perform_statistical_test(power_closed, power_open):
    """Perform t-test to check if closed condition is significantly higher than open"""
    power_open = np.mean(np.mean(np.max(power_open, axis=-1), axis=-1))
    power_closed = np.mean(np.max(power_closed, axis=-1), axis=-1)
    power_closed = power_closed / power_open
    result = ttest_1samp(power_closed, 1, alternative='greater')
    return result.pvalue

def plot_alpha_modulation_by_participant(all_power, out_dir):
    """Plot alpha modulation ratios by participant"""
    df = []
    names = {'scalp': 'Scalp', 'temporal': 'Temporal', 'ear': 'In-Ear'}
    
    for participant in all_power:
        row = {'Participant': participant}
        for headset in ['scalp', 'temporal', 'ear']:
            power_closed = all_power[participant][headset]['eyes closed']
            power_open = all_power[participant][headset]['eyes open']
            power_open = np.mean(np.mean(np.max(power_open, axis=-1), axis=-1))
            power_closed = np.mean(np.max(power_closed, axis=-1), axis=-1)
            power_closed = power_closed / power_open
            row[names[headset]] = np.mean(power_closed)
            row[f'{names[headset]} Error'] = np.std(power_closed) / np.sqrt(power_closed.size)
        df.append(row)
    
    df = pd.DataFrame(df)

    plt.figure(figsize=(10, 6))
    ax = df[['Scalp', 'Temporal', 'In-Ear']].plot.bar(rot=0)
    plt.xlabel('Participant')
    plt.ylabel('Alpha Modulation Ratio')
    plt.xticks(range(len(df)), range(1, len(df) + 1))

    # Add error bars
    errors = np.ravel(df[['Scalp Error', 'Temporal Error', 'In-Ear Error']].values.T)
    for patch, moe in zip(ax.patches, errors):
        height = patch.get_height()
        min_y, max_y = height - moe, height + moe
        plt.vlines(patch.get_x() + patch.get_width()/2, min_y, max_y, color='k')

    plt.ylim([0, None])
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(out_dir, 'alpha_modulation_ratio_by_participant.png'))
    
    return df

def main():
    """Main analysis pipeline"""
    # Configuration
    input_dir = r'S:\laura-wheeler_in-ear-eeg-auditory-bci_0589_data_prism\Raw Data\Study Data Organized - in-ear EEG and scalp working'
    
    # For local testing, uncomment and modify this path:
    # input_dir = '/path/to/your/local/data'
    
    epoch_twin_size = 5
    
    # Create output directory
    out_dir = create_output_directory()
    
    # Find XDF files
    xdf_files = find_xdf_files(input_dir)
    print("Found XDF files:", xdf_files)
    
    if not xdf_files:
        print("No XDF files found. Please check the input directory path.")
        return
    
    # Process each participant
    all_epochs = {}
    all_psds = {}
    all_ram = []
    all_power = {}
    all_ttest_results = []
    
    for i, participant in enumerate(xdf_files):
        print(f"\n=== Processing {participant} ===")
        
        # Load and process streams
        streams = load_and_process_streams(input_dir, participant, xdf_files, epoch_twin_size)
        
        # Create raw data
        raw = create_raw_data(streams, epoch_twin_size)
        
        # Create epochs
        epochs = create_epochs(raw)
        all_epochs[participant] = epochs
        
        # Compute PSD
        psds = compute_psd(epochs)
        all_psds[participant] = psds
        
        # Plot PSD
        plot_psd(psds, i+1, out_dir)
        
        # Compute alpha modulation
        ram_data, power = compute_alpha_modulation(psds)
        all_power[participant] = power
        
        ram_entry = {'Participant': int(participant.replace('Participant ', ''))}
        ram_entry.update(ram_data)
        all_ram.append(ram_entry)
        
        # Statistical tests
        ttest_entry = {
            'Participant': participant.replace('Participant ', ''),
            'Scalp p-value': perform_statistical_test(power['scalp']['eyes closed'], power['scalp']['eyes open']),
            'Temporal p-value': perform_statistical_test(power['temporal']['eyes closed'], power['temporal']['eyes open']),
            'In-Ear p-value': perform_statistical_test(power['ear']['eyes closed'], power['ear']['eyes open'])
        }
        all_ttest_results.append(ttest_entry)
    
    # Compute average statistics
    ram_df = pd.DataFrame(all_ram)
    ram_df.loc[len(ram_df)] = ['average', ram_df['Scalp RAM'].mean(), 
                               ram_df['Temporal RAM'].mean(), ram_df['In-Ear RAM'].mean()]
    ram_df = ram_df.round(decimals=2)
    
    # Compute average t-test results
    avg_power = {'scalp': {}, 'temporal': {}, 'ear': {}}
    for participant in all_power:
        for headset in avg_power:
            for task in ['eyes closed', 'eyes open']:
                if task not in avg_power[headset]:
                    avg_power[headset][task] = []
                avg_power[headset][task].append(all_power[participant][headset][task])
    
    for headset in avg_power:
        for task in ['eyes closed', 'eyes open']:
            avg_power[headset][task] = np.concatenate(avg_power[headset][task], axis=0)
    
    avg_ttest = {
        'Participant': 'average',
        'Scalp p-value': perform_statistical_test(avg_power['scalp']['eyes closed'], avg_power['scalp']['eyes open']),
        'Temporal p-value': perform_statistical_test(avg_power['temporal']['eyes closed'], avg_power['temporal']['eyes open']),
        'In-Ear p-value': perform_statistical_test(avg_power['ear']['eyes closed'], avg_power['ear']['eyes open'])
    }
    all_ttest_results.append(avg_ttest)
    
    # Save results
    ram_fname = os.path.join(out_dir, 'alpha_modulation_ratio.csv')
    ram_df.to_csv(ram_fname, index=False)
    print(f'\nSaved RAM results to: {ram_fname}')
    
    ttest_df = pd.DataFrame(all_ttest_results)
    ttest_fname = os.path.join(out_dir, 'alpha_ttest_results.csv')
    ttest_df.to_csv(ttest_fname, index=False)
    print(f'Saved t-test results to: {ttest_fname}')
    
    # Plot summary
    participant_df = plot_alpha_modulation_by_participant(all_power, out_dir)
    
    # Print results
    print("\n=== Alpha Modulation Ratios ===")
    print(ram_df)
    print("\n=== Statistical Test Results ===")
    print(ttest_df)
    print("\n=== Analysis Complete ===")
    
    return ram_df, ttest_df, participant_df

if __name__ == "__main__":
    main()
