"""
TEST CODE – CONVERTED AND USED FOR ENVIRONMENT SETUP ONLY
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import mne
import os
from pyxdf import load_xdf
import pandas as pd

# Set matplotlib backend for interactive plotting (fallback to default if Qt5Agg not available)
try:
    plt.switch_backend('Qt5Agg')
except ImportError:
    # Use default backend if Qt5Agg is not available
    pass

def setup_output_directory():
    """Create output directory for results."""
    out_dir = 'artifact_results'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    return out_dir

def find_xdf_files(input_dir):
    """Find all XDF files containing 'blink' in the filename."""
    xdf_files = {}
    participant_folders = [f for f in os.listdir(input_dir) 
                          if f.startswith('Participant') and f[-1].isdigit()]
    
    for participant_folder in participant_folders:
        sourcedata_path = os.path.join(input_dir, participant_folder, 'sourcedata')
        if os.path.exists(sourcedata_path):
            xdf_files[participant_folder] = [
                f for f in os.listdir(sourcedata_path) 
                if 'blink' in f and f.endswith('.xdf')
            ]
    
    return xdf_files

def load_and_process_streams(input_dir, participant, xdf_files):
    """Load LSL streams from XDF files for a participant."""
    print(f'Parsing XDF files for {participant}')
    
    streams = {}
    for f in range(len(xdf_files[participant])):
        xdf_file = xdf_files[participant][f]
        print('file: ', xdf_file)
        
        # Load LSL streams from xdf
        found_streams, _ = load_xdf(os.path.join(input_dir, participant, 'sourcedata', xdf_file))
        for found_stream in found_streams:
            stream_name = found_stream['info']['name'][0]
            if stream_name not in streams:
                streams[stream_name] = []
            streams[stream_name].append(found_stream)
    
    return streams

def create_raw_data(streams):
    """Create MNE Raw data objects from streams."""
    eeg_raw_list, raw = {}, {}
    
    for stream in ['BrainAmpSeries-Dev_1', 'InEarEEG']:
        # Load EEG data
        for i in range(len(streams[stream])):
            print(f'Processing {stream} stream')
            
            # Get EEG info
            if stream == 'BrainAmpSeries-Dev_1':
                sfreq = float(streams[stream][i]['info']['nominal_srate'][0])
                ch_names = [ch['label'][0] for ch in streams[stream][i]['info']['desc'][0]['channels'][0]['channel']]
            elif stream == 'InEarEEG':
                sfreq = float(streams[stream][i]['info']['effective_srate'])
                ch_names = ['A1', 'A2']
                
            eeg_info = mne.create_info(ch_names, sfreq, ch_types='eeg')
            montage = mne.channels.read_custom_montage(f'{stream}_montage.elc')

            # Apply scaling factor to EEG data
            if stream == 'InEarEEG':
                eeg_data = np.transpose(streams[stream][i]['time_series']) / 24 / 4.1887e9 * 4.5
            elif stream == 'BrainAmpSeries-Dev_1':
                eeg_data = np.transpose(streams[stream][i]['time_series']) / 1e6
                
            eeg_raw = mne.io.RawArray(eeg_data, eeg_info, verbose=False)
            eeg_raw.set_montage(montage)

            # Process marker annotations
            if stream == 'BrainAmpSeries-Dev_1':
                # Detect onsets of blinks
                markers = streams['mindset_Marker_PRISM-DIY-MSI-4'][i]['time_series']
                markers_ts = streams['mindset_Marker_PRISM-DIY-MSI-4'][i]['time_stamps']
                eog_ch = eeg_raw.copy().pick('Fz').filter(l_freq=1, h_freq=10, verbose=False).get_data()
                thresh = (np.max(eog_ch) - np.min(eog_ch)) * 0.5
                eog_events = mne.preprocessing.find_eog_events(
                    eeg_raw, ch_name='Fz', 
                    tstart=markers_ts[-1] - streams[stream][i]['time_stamps'][0], 
                    thresh=thresh, filter_length='3s', verbose=False
                )
                
                onsets, durations, descriptions = [], [], []
                scalp_ear_offset = (streams['InEarEEG'][i]['time_stamps'][0] - 
                                  streams['BrainAmpSeries-Dev_1'][i]['time_stamps'][0])
                
                for event in eog_events:
                    onset = event[0] / eeg_raw.info["sfreq"]
                    
                    # Check timing constraints
                    if onset - 0.6 < (markers_ts[-1] - streams[stream][i]['time_stamps'][0]):
                        continue
                    elif onset + 0.6 > (streams[stream][i]['time_stamps'][-1] - streams[stream][i]['time_stamps'][0]):
                        continue
                    if onset - 0.6 - scalp_ear_offset < (markers_ts[-1] - streams['InEarEEG'][i]['time_stamps'][0]):
                        print('SKIPPED')
                        continue
                    elif onset + 0.6 - scalp_ear_offset > (streams['InEarEEG'][i]['time_stamps'][-1] - streams['InEarEEG'][i]['time_stamps'][0]):
                        print('SKIPPED')
                        continue
                        
                    onsets.append(onset)
                    durations.append(0.4)
                    descriptions.append('blink')
                    
                annotations = mne.Annotations(onsets, durations, descriptions)
                eeg_raw = eeg_raw.set_annotations(annotations)
                
            elif stream == 'InEarEEG':
                # Use annotations from scalp EEG
                annotations = eeg_raw_list['BrainAmpSeries-Dev_1'][i].annotations
                annotations.onset = annotations.onset - (streams['InEarEEG'][i]['time_stamps'][0] - 
                                                        streams['BrainAmpSeries-Dev_1'][i]['time_stamps'][0])
                eeg_raw = eeg_raw.set_annotations(annotations)
                eeg_raw = eeg_raw.resample(250)

            # Add raw to list
            if stream not in eeg_raw_list:
                eeg_raw_list[stream] = []
            eeg_raw_list[stream].append(eeg_raw)

        # Concatenate all epochs and filter
        raw[stream] = mne.concatenate_raws(eeg_raw_list[stream])
        raw[stream] = raw[stream].filter(l_freq=1.0, h_freq=40.0)
    
    return raw

def create_epochs(raw):
    """Create epochs from raw data."""
    epochs = {}
    
    for stream in raw:
        events, event_id = mne.events_from_annotations(raw[stream])
        epochs[stream] = mne.Epochs(
            raw[stream], events, event_id=event_id, preload=True,
            tmin=-1, tmax=1, baseline=(-0.6, -0.4)
        )
    
    return epochs

def plot_erp(epochs, out_dir):
    """Plot Event Related Potentials for all participants."""
    plt.close('all')
    fig, ax = plt.subplots(4, 2, figsize=(8, 8))
    title = 'Blink Artifact Event Related Potentials'
    fig.suptitle(title)

    participants = list(epochs.keys())
    for i in range(len(ax)):
        participant = participants[i]
        
        # Plot Scalp ERP
        evoked = epochs[participant]['BrainAmpSeries-Dev_1'].average().crop(tmin=-0.6, tmax=0.6)
        evoked.plot(axes=ax[i, 0], picks='all')
        ax[i, 0].set_title('')
        ax[i, 0].set_ylabel(f'P{i+1}\nµV')
        if i != len(ax) - 1:
            ax[i, 0].set_xlabel('')
            ax[i, 0].set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6], 
                               ['', '', '', '', '', '', ''])
        
        # Plot In-Ear ERP
        evoked = epochs[participant]['InEarEEG'].average().crop(tmin=-0.6, tmax=0.6)
        evoked.plot(axes=ax[i, 1])
        ax[i, 1].set_title('')
        ax[i, 1].set_ylabel('')
        if i != len(ax) - 1:
            ax[i, 1].set_xlabel('')
            ax[i, 1].set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6], 
                               ['', '', '', '', '', '', ''])

    ax[0, 0].set_title('Scalp EEG')
    ax[0, 1].set_title('In-Ear EEG')

    # Adjust ERP y-axes
    for j in range(2):
        ylim = np.array([ax[i, j].get_ylim() for i in range(len(ax))])
        ylim = [np.min(ylim[:, 0]), np.max(ylim[:, 1])]
        for i in range(len(ax)):
            ax[i, j].set_ylim(ylim)
            ax[i, j].vlines(0, ylim[0], ylim[1], linestyles='dashed', colors='gray')

    # Adjust spacing
    plt.subplots_adjust(left=0.11, right=0.98, top=0.91, bottom=0.07, 
                       wspace=0.125, hspace=0.21)

    fig.savefig(os.path.join(out_dir, f'{title}.png'))
    fig.show()

def compute_amplitudes(participant, headset, evoked, clean_interval=(-0.6, -0.45), 
                      artifact_interval=(-0.6, 0.6)):
    """Compute amplitude metrics for clean vs artifact periods."""
    evoked_clean = evoked.get_data()[
        :, np.logical_and(evoked.times >= clean_interval[0], 
                         evoked.times <= clean_interval[1])
    ]
    evoked_artifact = evoked.get_data()[
        :, np.logical_and(evoked.times >= artifact_interval[0], 
                         evoked.times <= artifact_interval[1])
    ]
    
    amp_clean = (np.max(evoked_clean, axis=1) - np.min(evoked_clean, axis=1)) * 1e6
    amp_artifact = (np.max(evoked_artifact, axis=1) - np.min(evoked_artifact, axis=1)) * 1e6
    percent_change = (amp_artifact - amp_clean) / amp_clean * 100
    max_increase_idx = np.argmax(percent_change)
    max_amp_idx = np.argmax(amp_artifact)
    
    return {
        'Participant': participant.replace('Participant ', ''),
        'Channels': headset,
        'Clean Amplitude': np.max(amp_clean),
        'Artifact Amplitude': np.max(amp_artifact),
        'Max Increase Channel': evoked.ch_names[max_increase_idx],
        'Max Percent Change': np.max(percent_change),
        'Max Amplitude Channel': evoked.ch_names[max_amp_idx],
        'Max Amp Percent Change': percent_change[max_amp_idx],
    }

def analyze_amplitudes(epochs):
    """Analyze amplitude changes due to blink artifacts."""
    clean_intervals = {
        'Participant 35': (-0.6, -0.35),
        'Participant 37': (-0.6, -0.42),
        'Participant 38': (-1, -0.6),
        'Participant 40': (-1, -0.72)
    }
    
    amplitudes = []
    for participant in clean_intervals:
        # Scalp
        evoked = epochs[participant]['BrainAmpSeries-Dev_1'].average().crop(tmin=-1, tmax=1)
        ampdict = compute_amplitudes(participant, 'Scalp', evoked, 
                                   clean_interval=clean_intervals[participant])
        amplitudes.append(ampdict)
        
        # In-Ear
        evoked = epochs[participant]['InEarEEG'].average().crop(tmin=-1, tmax=1)
        ampdict = compute_amplitudes(participant, 'In-Ear', evoked, 
                                   clean_interval=clean_intervals[participant])
        amplitudes.append(ampdict)

    amplitudes_df = pd.DataFrame(amplitudes)
    return amplitudes_df

def print_summary_statistics(amplitudes_df):
    """Print summary statistics for amplitude analysis."""
    scalp_data = amplitudes_df.loc[amplitudes_df['Channels'] == 'Scalp']
    inear_data = amplitudes_df.loc[amplitudes_df['Channels'] == 'In-Ear']
    
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Scalp EEG - Mean Max Amplitude Percent Change: {scalp_data['Max Amp Percent Change'].mean():.2f}%")
    print(f"In-Ear EEG - Mean Max Amplitude Percent Change: {inear_data['Max Amp Percent Change'].mean():.2f}%")
    
    print("\nScalp EEG Results:")
    print(scalp_data[['Participant', 'Max Amplitude Channel', 'Max Amp Percent Change']])
    
    print("\nIn-Ear EEG Results:")
    print(inear_data[['Participant', 'Max Amplitude Channel', 'Max Amp Percent Change']])

def main():
    """Main analysis function."""
    # Setup
    out_dir = setup_output_directory()
    
    # Input directory
    input_dir = os.getcwd()
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Warning: Input directory does not exist: {input_dir}")
        print("Please update the input_dir variable with the correct path.")
        return
    
    # Find XDF files
    xdf_files = find_xdf_files(input_dir)
    print("Found XDF files:")
    for participant, files in xdf_files.items():
        print(f"  {participant}: {files}")
    
    if not xdf_files:
        print("No XDF files found. Please check the input directory path.")
        return
    
    # Process each participant
    all_epochs = {}
    for participant in xdf_files:
        try:
            # Load streams
            streams = load_and_process_streams(input_dir, participant, xdf_files)
            
            # Create raw data
            raw = create_raw_data(streams)
            
            # Create epochs
            epochs = create_epochs(raw)
            all_epochs[participant] = epochs
            
        except Exception as e:
            print(f"Error processing {participant}: {str(e)}")
            continue
    
    if not all_epochs:
        print("No data was successfully processed.")
        return
    
    print(f"\nSuccessfully processed {len(all_epochs)} participants")
    
    # Plot ERPs
    plot_erp(all_epochs, out_dir)
    
    # Analyze amplitudes
    amplitudes_df = analyze_amplitudes(all_epochs)
    
    # Save results
    amplitudes_df.to_csv(os.path.join(out_dir, 'amplitude_analysis.csv'), index=False)
    
    # Print summary
    print_summary_statistics(amplitudes_df)
    
    print(f"\nResults saved in: {out_dir}")
    print("- ERP plot: Blink Artifact Event Related Potentials.png")
    print("- Amplitude analysis: amplitude_analysis.csv")

if __name__ == "__main__":
    main()
