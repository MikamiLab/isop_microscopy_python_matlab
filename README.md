High-speed volumetric imaging of complex biological dynamics with image scanning light-sheet microscopy

# Repository Contents

## Velocity Analysis
- **velocity_timetrace_spin.ipynb**  
  - For *Spinning Confocal Microscope*.  
  - Calculate velocity profile from tracking stage log file.  
  - Used in: *Ext5ab*

- **velocity_timetrace_tardigrade.ipynb**  
  - For *ISOP Microscope*.  
  - Calculate velocity profile from tracking stage log file.  
  - Used in: *Ext6*

- **velocity_timetrace_chlamydomonas.ipynb**  
  - For *ISOP Microscope*.  
  - Calculate velocity profile.  
  - Used in: *Fig4bc*

- **velocity_timetrace_celegans.ipynb**  
  - For *ISOP Microscope*.  
  - Calculate velocity profile from tracking stage log file.  
  - Used in: *SFig2*

## Video Generation



## Stage Control
- **tracking_stage/main10.py**  
  - Main code to control tracking stage.

- **tracking_stage/app_module.py**  
  - Functions for `main10.py`.

---

## HDF5 Data Conversion
- **HDF5/Transform_binary_to_hdf5.ipynb**  
  - Read custom binary microscopy data and transform into HDF5 format (metadata + compression).  
  - Used in: *Fig2a, Fig3a,b,d, Fig4a,d, Ext3a, Video2–5*

- **HDF5/binary2hdf5.py**  
  - Functions to convert single binary microscopy files into HDF5 and read volumes at specific time points.  
  - Used in: *Fig2a, Fig3a,b,d, Fig4a,d, Ext3a, Video2–5*

- **HDF5/Transform_HDF5_to_tiff.ipynb**  
  - Convert HDF5 stacks into TIFF.  
  - Used in: *Fig2a, Fig3a,b,d, Fig4a,d, Ext3a, Video2–5*

- **HDF5/Transform_tiff_to_hdf5.ipynb**  
  - Convert TIFF stacks into HDF5.  
  - Used in: *Ext5*

- **HDF5/making_orthogonal_view_fluoimages_labelimages.ipynb**  
  - Create orthogonal view images combining fluorescence and label images.  
  - Used in: *Fig2a,b, Fig3a,b,d, Ext3a,b, Supp3a,b,d, Video2–5*

---

## Tracking & Signal Analysis
- **Chlamydomonas_3DU16/3DU16_v3.py**  
  - Tracking algorithms for Chlamydomonas data.  
  - Create trajectory images.  
  - Used in: *Fig4a, Ext7*

- **stardist_notebook/Extract_calcium_signal-v3-allbrightness.ipynb**  
  - Extract brightness values from ROIs.  
  - Used in: *Fig2c, Fig3c, Ext3c, Ext4b*

- **stardist_notebook/Calculate_label_movement.ipynb**  
  - Calculate 3D movement distance of ROIs.  
  - Used in: *Fig2g*

- **stardist_notebook/Calculate_cell_movement_tardigrade.ipynb**  
  - Calculate 3D distance between two ROIs.  
  - Used in: *Fig3c*

---

## Posture & Behavior Analysis (Spin Microscope)
- **Spin/test20240710_skeletonize2.m**  
  - Extract worm centerline (posture) from brightfield time series.  
  - Used in: *Fig2d, Supp3a–d*

- **Spin/test20240711_posture_combination.m**  
  - Convert posture into angle series, analyze and visualize dynamics.  
  - Used in: *Fig2d, Supp3a,b,d*

---

## Other Analyses
- **Other_Codes/Behavior5_waveform.ipynb**  
  - Visualize waveform patterns of Behavior5 and neural signals.  
  - Used in: *Fig2d, Supp3c,d*

- **Other_Codes/Motif#5_timeseries_head_bends_marker.ipynb**  
  - Plot neural motifs aligned with head-bend markers.  
  - Used in: *Supp3c,d*

- **Other_Codes/CorrelationMatrixSheet.ipynb**  
  - Visualize correlation matrices between neural activity and behavior.  
  - Used in: *Ext4a*

- **Other_Codes/Local_velocity_ErrorRate.ipynb**  
  - Visualize mean apparent speed vs error rate.  
  - Used in: *Fig2g*

- **Other_Codes/tardigrade_3D_distance_mCherry_GCaMP6s_Ratio.ipynb**  
  - Correlate fluorescence ratio signals with 3D cellular distances.  
  - Visualize time-series.  
  - Used in: *Fig3c*

- **Other_Codes/ColorBar_BehaviorType.ipynb**  
  - Plot behavioral time series with fixed color bar.  
  - Used in: *Fig2c, Ext3c, Ext4b*

- **Other_Codes/signal_plots_v2.ipynb**  
  - Create heatmaps of ratio signals.  
  - Used in: *Fig2c, Ext3c, Ext4b*

---

## Video Processing
- **Video/3Dimage_bleach_correct_v2.ipynb**  
  - Apply bleach correction to images.  
  - Used in: *Video2, Video3, Video4*

- **Video/Boundaries_Frames_ScaleBar_MP4.ipynb**  
  - Generate orthogonal-view worm videos with boundaries, scale bar, and padding.  
  - Used in: *Video2, Video3*

- **Video/Merge_FL_BL_newTimestamp_v2.ipynb**  
  - Merge fluorescence and brightfield videos, align timestamps, add scale bar.  
  - Used in: *Video2*

- **Video/Worm2-4_merge.ipynb**  
  - Integrate Worm2–4 videos with scale bars.  
  - Used in: *Video3*

- **Video/Video4_maker.ipynb**  
  - Integrate ISOP (worm1) and SPIN videos, add scale bars.  
  - Used in: *Video4*

- **Video/video5_new_layout_centered.ipynb**  
  - Used in: *Video5*.

- **Video/movie8.ipynb**  
  - Process and output annotated movie.  
  - Used in: *Video8*
