# Thesis Notes | ECE499 UofT

**Myant Textile Electrode Designs:**
- 1.0 --> Raised Height (1x density)
    - 1.1: 1 mm
    - 1.2: 2 mm
    - 1.3: 3 mm
- 2.0 --> Silver Density (2mm height)
    - 2.1: least dense (1x)
    - 2.2: moderately dense (1.5x)
    - 2.3: most dense (2x)
- 3.0 --> Area Size (1x density, 2mm height)
    - 3.1: ~0.5 x 0.5 cm
    - 3.2: ~1.0 x 1.0 cm
    - 3.3: ~2.0 x 2.0 cm
- Notes:
    - All pad sizes are approximately 1.0 x 1.0 cm if not specified
    - Need more precise quantitative measurables


**Sept. 17th, 2025:**
- Generate a 3x3 plot for the blink and visual paradigm data
    - 3 rows
    - 3 columns 
    - each of the 9 swatches
- ECE499 proposal background research areas
    - textile electrodes
    - validation methods 
    - techniques/paradigms used to validate
    - active electrode design
    - instrumentation amplifiers

**Sept. 24th, 2025:**
- Continue working on the code and make it specific to our testing cases
- Can compute the ratio levels from the xdf data

**Oct. 8th, 2025:**
- Active electrode design should use an instrumentational amplifier (3 op-amp setup)
- Active shielding is definitely needed
- ADS1299 is the IC that OpenBCI uses
- Quasar/Wearable Sensing dry electrode diassembly

**Nov. 7th, 2025:**
- TODO for updating the plotting script
    - Find the SNR between Fp1 and Fp2
        - SNR --> Fp2 (Gold-Standard Electrode) / Difference between Fp1 and Fp2
    - Calculate the percent difference
        - % Difference = (| Gold-Standard - Textile | / Gold-Standard) * 100
    - Alpha band attenutation rates between Eyes Open and Eyes Closed
    - Add a 1-50 Hz filter to all the raw EEG data
    - Cut off first and last 10 seconds in the BLINK DATA
    - Use .annotations to mark off area to analyze in EYES OPEN and EYES CLOSED
        - Create a new user argument to specify this case when analyzing these recordings

**Nov. 27th, 2025:**
- 1 second cut off at start and end of annotations
- Epoch the annotations for eyes closed and eyes open
- Average the alpha band power across the ten samples and ratio them
- Add a colour filter between 8-12 Hz in the PSD graphs
- Plot all blink percent differences into a summary file

**Jan 14th, 2026:**
- Goals: 
    - Jan. 28th: Component selection complete, order missing components
    - Feb. 4th: Begin breadboard selection
    - Feb. 11th: Complete breadboard selection
    - Rest of Feb.: KiCAD schematic and function generator testing
    - Mar. 4th: Begin testing against clinical-grade EEG systems