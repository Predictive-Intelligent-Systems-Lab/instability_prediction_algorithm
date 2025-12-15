#############################################
#    Ocean Wave Extreme Events Prediction   #
#############################################

What it does: Analyzes ocean wave data to predict when extreme waves will occur

How it works: Takes wave measurements and uses machine learning to find warning patterns before dangerous waves

***********************
Files you need:
***********************
  RWs_H_g_2p2_tadv_1min_buoy_132.npz (wave data)
  recurrence_matrix_cnn.pth (trained AI model)

************************
How to run:
************************
1. python step_1_signal_loading_segmentation.py
2. python step_2_cao_theorm.py  
3. python step_3_average_mutual_information.py
4. python step_4_recurrence_matrix_generation.py
5. python step_5_convolutional_neural_network_prediction.py
6. python step_6_plotting.py

################################
What each script does:
################################

1. step_1_signal_loading_segmentation.py
========================
Takes raw wave data and breaks it into smaller pieces
Enhances the signal to make extreme events more visible
Creates: complete_signal/ and signal_segments/ folders

2. step_2_cao_theorm.py
============
Figures out how many dimensions needed for analysis
Takes: signal_segments/ files
Creates: signal_segments_embed/ files

3. step_3_average_mutual_information.py
===========================
Calculates timing parameters for the analysis
Takes: signal_segments_embed/ files  
Creates: signal_segments_embed_tau/ files

4. pystep_4_recurrence_matrix_generation.py
===============================
Creates visual patterns from the data for AI analysis
Takes: signal_segments_embed_tau/ files
Creates: signal_segments_recurrence/ files

5. step_5_convolutional_neural_network_prediction.py
==============================================================
Uses AI to classify patterns and predict extreme events
Takes: signal_segments_recurrence/ files
Creates: plot_data_signal/ folder with results

################################
Results you get:
################################

The final script creates analysis files in plot_data_signal/:
  signal_data.npz (processed wave signal)
  peak_data.npz (locations of extreme waves)
  precursor_data.npz (warning signals found)
  statistics.txt (summary of results)

################################
What you need installed:
################################

Python libraries: numpy, scipy, matplotlib, torch, scikit-learn, tqdm
