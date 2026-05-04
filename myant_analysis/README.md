# Comprehensive XDF EEG Analysis Tool – For Thesis (Daniel Yu)

This repository contains `thesis_plot_xdf.py`, a thesis-focused EEG analysis script for `.xdf` recordings. It loads XDF EEG streams with MNE, cuts and filters the signals, and exports plots for:

- 📊 PSD and frequency-band summaries
- 🧠 ICA component summaries
- 📈 Raw EEG segment views
- 👁️ Blink epoch / ERP analysis (--analysis blink)
- 🌊 Alpha band and attenuation analysis (--analysis alpha)

Please note that this script was made specifically for the needs of my thesis.

## What The Script Does

For each XDF file, the script:

1. Loads EEG (and marker stream if available) from `.xdf`
2. Extracts/normalizes channel data into MNE `RawArray`
3. Determines standard 10-20 location
4. Applies 1-50 Hz filtering
5. Computes shared PSD scaling across all files in a run
6. Generates and saves analysis figures into results folders

For blink mode and alpha mode, it also computes grouped metrics across files and generates summary bar charts.

## Plots Generated

Per-file outputs (created for all runs):

- `psd_summary_<timestamp>.png`
	- PSD curves, electrode map (if location available), band-power bars, channel stats
- `ica_summary_<timestamp>.png`
	- ICA topomaps, ICA time courses, component statistics
- `raw_eeg_<window>_<timestamp>.png`
	- Raw EEG segment (default 30-45s, default is full recording if 30-45s does not exist)
- `comprehensive_analysis_<timestamp>.png`
	- Full recording view, frequency-band power, variability metrics, SNR bars

Additional per-file outputs by mode:

- Blink mode (`--analysis blink`):
	- `blink_epoch_analysis_<timestamp>.png`
		- Blink ERP, individual blink overlays, amplitude distributions, inter-channel comparison
- Alpha mode (`--analysis alpha`):
	- `alpha_wave_analysis_<timestamp>.png`
		- Alpha spectrum (8-13 Hz), alpha-over-time, filtered alpha, envelope, stats
	- Single-file alpha runs also create:
		- `alpha_time_frequency_<timestamp>.png`

Cross-file summary outputs:

- Blink mode (multi-file):
	- `snr_all_files_sorted_<timestamp>.png`
	- `snr_all_files_fp[#]_<timestamp>.png`
	- `percent_diff_all_files_sorted_<timestamp>.png`
- Alpha mode (multi-file):
	- `alpha_attenuation_summary_<timestamp>.png`
	- `alpha_attenuation_by_channel_<timestamp>.png`

## Output Folder Structure

- No analysis flag: outputs go to `xdf_results/<xdf_basename>/`
- `--analysis blink`: outputs go to `blink_results/<xdf_basename>/`
- `--analysis alpha`:
	- single-file run: `alpha_results/<xdf_basename>/`
	- subfolder batch run: `alpha_results/<subfolder_name>/<xdf_basename>/`

## How To Run

Run from repository root.

### 1) Single file

```bash
python thesis_plot_xdf.py /path/to/file.xdf
python thesis_plot_xdf.py /path/to/file.xdf --analysis blink
python thesis_plot_xdf.py /path/to/file.xdf --analysis alpha
```

### 2) Folder of XDF files

```bash
python thesis_plot_xdf.py /path/to/folder
python thesis_plot_xdf.py /path/to/folder --analysis blink
```

### 3) Alpha grouped-folder mode

```bash
python thesis_plot_xdf.py /path/to/alpha_parent --analysis alpha
```

In this mode, the parent folder must contain subfolders such as `1.1_eyes_open` and `1.1_eyes_closed`, each containing `.xdf` files.

### 4) Interactive mode

```bash
python thesis_plot_xdf.py
```

Then enter a path when prompted.

## Required XDF Input Layout

### A) Single-file mode

Any valid `.xdf` file path.

### B) Standard folder mode (no alpha grouping)

```text
your_folder/
	file1.xdf
	file2.xdf
	...
```

### C) Alpha grouped folder mode (`--analysis alpha`)

```text
alpha_parent/
	1.1_eyes_open/
		run1.xdf
		run2.xdf
	1.1_eyes_closed/
		run1.xdf
		run2.xdf
	2.1_eyes_open/
		...
	2.1_eyes_closed/
		...
```

Notes:

- Alpha grouping logic depends on subfolder names containing `_eyes_open` or `_eyes_closed`.
- Swatch/group labels are derived from the part before that suffix (example: `1.1` from `1.1_eyes_open`).
- If no EEG stream exists in an XDF file, that file is skipped with an error.

## Python And Library Versions

This workspace environment used:

- Python `3.12.11`
- `mne==1.10.1`
- `numpy==2.3.3`
- `scipy==1.16.2`
- `matplotlib==3.10.6`
- `pyxdf==1.17.0`

To install dependencies in a fresh environment:

```bash
pip install -r requirements.txt
```

## Other Notes

- The script attempts microvolt-to-volt conversion automatically when values look unscaled.
- For batch blink runs, it trims 10 seconds from start/end before analysis. This is only the case when the user input is a folder.
- For batch alpha runs, it trims 5 seconds from the start. This is only the case when the user input is a folder.
- Some channel-specific calculations expect channels like `Fp1` and `Fp2`; files without these labels may still plot, but some comparative metrics can be skipped.

## Troubleshooting

- `No .xdf files found`:
	- Check that files use lowercase `.xdf` extension and are in the expected folder level.
- `No EEG stream found in XDF file`:
	- Verify stream metadata/type in the recording.
- Empty or missing summary plots in grouped modes:
	- Ensure both open and closed sets exist for each swatch key.

## Contact

- UofT Email: [danielz.yu@mail.utoronto.ca](mailto:danielz.yu@mail.utoronto.ca)
- Personal Email: [danielu6776@gmail.com](mailto:danielu6776@gmail.com)
