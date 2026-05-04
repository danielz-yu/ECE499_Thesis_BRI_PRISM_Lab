## Enclosure Design CAD

This folder contains the source and export files used for the active electrode enclosure.

- `3D_printing/` — print-preparation files:
	- `.3mf` file: Editable 3D slicer file. This can be modified when changing which parts to print or print settings before slicing.
	- `.gcode` file: Pre-sliced printer instructions prepared with my printer settings (PLA, Bambu Lab A1).
- `fusion360/` — Native Autodesk Fusion 360 part files:
	- `.f3d` files: Contains full editable single-part designs (retains design history and all features).
	- `.f3z` files: Fusion 360 assemblies (contain multiple parts and references). Assemblies require their referenced part files to open correctly.
- `renders/` — Rendered images showing what the assembled enclosure should look like.
- `step/` — STEP exports for CAD exchange. STEP is more interoperable but still loses parametric feature history compared to native files.
- `stl/` — Exported meshes for 3D printing or simulation. Note: STL is a triangulated mesh and loses feature/history/parametric data from the native CAD.


## Other Notes

- Use the native Fusion 360 `.f3d` files to make design changes or extract precise dimensions. 
- Use STEP only for exchange with other CAD packages or if the CAD software does not import native Fusion 360 files.
- Use STL for quick printing or prototyping, but be aware it is a mesh-only format.
- The `.3mf` in `3D_printing/` is the recommended starting point for preparing prints as it contains part placement and custom print settings.
- The included `.gcode` was sliced for a Bambu Lab A1 with PLA using my printer profile. If you use a different printer or material, re-slice from the `.3mf` (or from an STL exported from the native part) to ensure proper temperatures, speeds, and bed/filament settings.
