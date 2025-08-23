# Image-scanning light-sheet microscopy for high-speed volumetric imaging of complex biological dynamics

---

# Repository Contents

---

## Velocity Analysis
- **velocity_profile/velocity_timetrace_spin.ipynb**  
  - For *Spinning-Disk Confocal Microscope*.  
  - Calculate velocity profile from tracking stage log file.  
  - Used in: *Extended Data Fig. 5a and 5b*

- **velocity_profile/velocity_timetrace_tardigrade.ipynb**  
  - For *ISOP Microscope*.  
  - Calculate velocity profile from tracking stage log file.  
  - Used in: *Extended Data Fig. 6*

- **velocity_profile/velocity_timetrace_chlamydomonas.ipynb**  
  - For *ISOP Microscope*.  
  - Calculate velocity profile.  
  - Used in: *Fig. 4b and 4c*

- **velocity_profile/velocity_timetrace_celegans.ipynb**  
  - For *ISOP Microscope*.  
  - Calculate velocity profile from tracking stage log file.  
  - Used in: *Supplementary Figure 2*

---

## Stage Tracking Control
- **tracking_stage/main10.py**  
  - Main code to control automated stage tracking system.

- **tracking_stage/app_module.py**  
  - Functions for `main10.py`.

---

## HDF5 Data Conversion and Visualization
- **HDF5/Transform_binary_to_hdf5.ipynb**  
  - Read custom binary microscopy data and transform into HDF5 format (metadata + compression).  
  - Used in: *Fig. 2a, 3a, 3b, 3d, 4a, and 4d; Extended Data Fig. 3a; Supplementary Video 2-5*

- **HDF5/binary2hdf5.py**  
  - Functions to convert single binary microscopy files into HDF5 and read volumes at specific time points.  
  - Used in: *Fig. 2a, 3a, 3b, 3d, 4a, and 4d; Extended Data Fig. 3a; Supplementary Video 2-5*

- **HDF5/Transform_HDF5_to_tiff.ipynb**  
  - Convert HDF5 file into TIFF file.  
  - Used in: *Fig. 2a, 3a, 3b, 3d, 4a, and 4d; Extended Data Fig. 3a; Supplementary Video 2-5*

- **HDF5/Transform_tiff_to_hdf5.ipynb**  
  - Convert TIFF file into HDF5 file.  
  - Used in: *Extended Data Fig. 5*

- **HDF5/making_orthogonal_view_fluoimages_labelimages.ipynb**  
  - Create orthogonal view images combining fluorescence and label images.  
  - Used in: *Fig. 2a, 2b, 3a, 3b and 3d; Extended Data Fig. 3a and 3b; Supplementary Figure 3a, 3b, and 3d; Supplementary Video 2-5*

---

## Tracking & Signal Analysis
- **Chlamydomonas_3DU16/3DU16_v3.py**  
  - Tracking algorithms for Chlamydomonas data.  
  - Create trajectory images.  
  - Used in: *Fig. 4a; Extended Data Fig. 7*

- **stardist_notebook/Extract_calcium_signal-v3-allbrightness.ipynb**  
  - Extract brightness values from ROIs.  
  - Used in: *Fig. 2c and 3c; Extended Data Fig. 3c and 4b*

- **stardist_notebook/Calculate_label_movement.ipynb**  
  - Calculate 3D movement distance of ROIs.  
  - Used in: *Fig. 2g*

- **stardist_notebook/Calculate_cell_movement_tardigrade.ipynb**  
  - Calculate 3D distance between two ROIs.  
  - Used in: *Fig. 3c*

---

## Posture & Behavior Analysis
- **Spin/test20240710_skeletonize2.m**  
  - Extract worm centerline (posture) from brightfield images.  
  - Used in: *Fig. 2d; Supplementary Figure 3a, 3b, 3c, and 3d*

- **Spin/test20240711_posture_combination.m**  
  - Convert posture into angle series, analyze and visualize dynamics.  
  - Used in: *Fig. 2d; Supplementary Figure 3a, 3b, and 3d*


---

## Processing Supplementary Video
- **Video/3Dimage_bleach_correct_v2.ipynb**  
  - Apply bleach correction to images.  
  - Used in: *Supplementary Video 2-4*

- **Video/Boundaries_Frames_ScaleBar_MP4.ipynb**  
  - Generate orthogonal-view worm videos with boundaries, scale bar, and padding.  
  - Used in: *Supplementary Video 2 and 3*

- **Video/Merge_FL_BL_newTimestamp_v2.ipynb**  
  - Merge fluorescence and brightfield videos, align timestamps, and add scale bar.  
  - Used in: *Supplementary Video 2*

- **Video/Worm2-4_merge.ipynb**  
  - Integrate Worm2–4 videos with scale bars.  
  - Used in: *Supplementary Video 3*

- **Video/Video4_maker.ipynb**  
  - Integrate ISOP (worm1) and spinning-disk confocal microscopy videos, and add scale bars.  
  - Used in: *Supplementary Video 4*

- **Video/video5_new_layout_centered.ipynb**  
  - Merge fluorescence and brightfield videos, align timestamps, and add scale bar.
  - Used in: *Supplementary Video 5*

- **Video/movie8.ipynb**  
  - Process and output annotated movie.  
  - Used in: *Supplementary Video 8*

---

## Other Analyses
- **Other_Codes/Behavior5_waveform.ipynb**  
  - Visualize waveform patterns of Behavior5 and neural signals.  
  - Used in: *Fig. 2d; Supplementary Figure 3c and 3d*

- **Other_Codes/Motif#5_timeseries_head_bends_marker.ipynb**  
  - Plot neural motifs aligned with head-bend markers.  
  - Used in: *Supplementary Figure 3c and 3d*

- **Other_Codes/CorrelationMatrixSheet.ipynb**  
  - Visualize correlation matrices between neural activity and behavior.  
  - Used in: *Extended Data Fig. 4a*

- **Other_Codes/Local_velocity_ErrorRate.ipynb**  
  - Visualize mean apparent speed vs error rate.  
  - Used in: *Fig. 2g*

- **Other_Codes/tardigrade_3D_distance_mCherry_GCaMP6s_Ratio.ipynb**  
  - Correlate fluorescence ratio signals with 3D cellular distances.  
  - Visualize time-series.  
  - Used in: *Fig. 3c*

- **Other_Codes/ColorBar_BehaviorType.ipynb**  
  - Plot behavioral time series with fixed color bar.  
  - Used in: *Fig. 2c; Extended Data Fig. 3c and 4b*

- **Other_Codes/signal_plots_v2.ipynb**  
  - Create heatmaps of ratio signals.  
  - Used in: *Fig. 2c; Extended Data Fig. 3c and 4b*
 
- **Other_Codes/SGfilter-PhothobleachCorrection_v2.ipynb**  
  - Perform Savitzky–Golay filtering and double-exponential photobleaching correction on tdTomato and GCaMP6f fluorescence signals, while preserving original NaN gaps by temporarily interpolating them and then restoring the missing values after processing.  
  - Used in: *Fig. 2c; Extended Data Fig. 3c and 4b*
