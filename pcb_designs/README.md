# Printed Circuit Boards (PCB)

This folder contains the KiCad PCB designs for the active electrode prototypes, including custom KiCad component libraries, board schematics and layouts, and production files.

## Custom KiCad Libraries

The `custom_KiCad_components/` folder contains custom-designed symbols, footprints, and 3D models for components used in these PCB designs.

### Adding Custom Libraries to KiCad:

#### 1. Symbol Libraries
Open KiCad Symbol Editor → Preferences → Manage Symbol Libraries → Folder Icon (bottom left corner) → Select the symbols folder at:
```
/{USER_PATH}/custom_KiCad_components/ECE499_Thesis.kicad_sym
```

#### 2. Footprint Libraries
Open KiCad Footprint Editor → Preferences → Manage Footprint Libraries → Folder Icon (bottom left corner) → Select the .pretty folder at:
```
/{USER_PATH}/custom_KiCad_components/ECE499_Thesis.pretty/
```

#### 3. 3D Models
3D models are located in:
```
/{USER_PATH}/custom_KiCad_components/ECE499_Thesis.3dshapes/
```
3D models can be copied to your local computer's KiCad application folder at the `3dmodels` folder and synced to each component manually.

### Important Notes:
- In both the Symbol and Footprint Editors, ensure that custom libraries have **"Active"** and **"Visible"** checkmarks enabled.
  - These are set by default but may need verification if the libraries do not appear.
  - Refresh the editors after enabling to load the new custom components.

## Folder Structure

### `old_designs/`
Contains old, incorrect KiCad PCB designs kept for historical reference only. **Do not use these for production or future work.**

### `OpenEEG-J_Foltynsky/` and `OpenEEG-J_Hansmann/`
Two reference designs based on the [OpenEEG project](https://users.dcc.uchile.cl/~peortega/ae/):
- **J. Foltynsky Design**: OpenEEG variant by Jarek Foltynsky
- **J. Hansmann Design**: OpenEEG variant by Joerg Hansmann

Each folder contains:
- `.kicad_sch` 	— Schematic file
- `.kicad_pcb` 	— PCB layout file
- `.kicad_pro` 	— KiCad project file (main file) 
- `.step` 		— 3D STEP model of the board
- `-backups/` 	— Backup folder for version control
- `production/` — Generated production files (see explanation below)

#### Production Folder
The `production/` subfolder contains files generated using the **KiCad Fabrication Toolkit** (available natively in KiCad Content Manager):
- **Gerber Files** — A zipped file containing all PCB layers and drill holes for board fabrication
- **BOM (Bill of Materials)** — A CSV list of all active and passive components required for direct manufacturer PCB assembly
- **Positions CSV** — XY coordinates and rotation angles of all components required for direct manufacturer PCB assembly

## JLCPCB Ordering Details

These PCB designs are configured for ordering from JLCPCB. Use the following settings:

### Recommended Settings
- **URL**: https://jlcpcb.com/
- **Deburring**: No
  - Deburring is typically only available for very small boards
  - Use "No" for prototype boards (deburring adds time)
  - For final production boards, enable deburring to polish the edges

### PCB Assembly
- **Make sure to double-check the footprint for all components** before submitting.
- **Verify integrated circuit (IC) pin alignment** — Ensure all ICs are oriented with the correct first pin position matching the schematic.

### Upload Files
- Upload the `.zip` Gerber file directly from the `production/` folder
- Upload the BOM and positions CSV for assembly services if needed
- Review and confirm component placement on JLCPCB's interactive preview before finalizing