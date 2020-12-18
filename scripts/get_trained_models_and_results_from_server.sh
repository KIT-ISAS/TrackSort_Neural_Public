#!/bin/sh

sftp thumm@i81-gpu-server.iar.kit.edu <<EOF

## Get models
# pepper
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/MLP_separator_uncertainty.h5 /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/LSTM_64_16_separation_uncertainty.h5 /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CV_CV_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_CA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_CVBC_CV.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_CVBC_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_LV_CV.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CV_CV_CV.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_CVBC_CA.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_IA_CV.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_CA_CV.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_LV_CA.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_IA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_IA_CA.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_CA_CA.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CV_IA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CV_IA_CV.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CV_CVBC_CV.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CA_LV_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/KF_sep_CV_CVBC_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/me_gating_sep.h5 /home/jakob/code/TrackSort_Neural/models/pepper/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper/smape_gating_sep.pkl /home/jakob/code/TrackSort_Neural/models/pepper/
# cylinder
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/MLP_separator_uncertainty.h5 /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/LSTM_64_16_separation_uncertainty.h5 /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CV_CV_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_CA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_CVBC_CV.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_CVBC_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_LV_CV.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CV_CV_CV.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_CVBC_CA.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_IA_CV.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_CA_CV.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_LV_CA.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_IA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_IA_CA.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_CA_CA.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CV_IA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CV_IA_CV.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CV_CVBC_CV.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CA_LV_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/KF_sep_CV_CVBC_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
# spheres
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/MLP_separator_uncertainty.h5 /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/LSTM_64_16_separation_uncertainty.h5 /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CV_CV_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_CA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_CVBC_CV.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_CVBC_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_LV_CV.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CV_CV_CV.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_CVBC_CA.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_IA_CV.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_CA_CV.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_LV_CA.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_IA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_IA_CA.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_CA_CA.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CV_IA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CV_IA_CV.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CV_CVBC_CV.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CA_LV_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/spheres/KF_sep_CV_CVBC_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/spheres/
# wheat
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/MLP_separator_uncertainty.h5 /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/LSTM_64_16_separation_uncertainty.h5 /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CV_CV_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_CA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_CVBC_CV.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_CVBC_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_LV_CV.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CV_CV_CV.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_CVBC_CA.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_IA_CV.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_CA_CV.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_LV_CA.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_IA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_IA_CA.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_CA_CA.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CV_IA_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CV_IA_CV.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CV_CVBC_CV.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CA_LV_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/wheat/KF_sep_CV_CVBC_Ratio.pkl /home/jakob/code/TrackSort_Neural/models/wheat/
# pepper cylinder spheres wheat ME
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder_pepper_spheres_wheat/me_gating_sep.pkl /home/jakob/code/TrackSort_Neural/models/cylinder_pepper_spheres_wheat/


## Get Results
# pepper
get /home/thumm/masterthesis/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_se/* /home/jakob/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_se/
get /home/thumm/masterthesis/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_smape/* /home/jakob/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_smape/
get /home/thumm/masterthesis/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_me/* /home/jakob/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_me/
# pepper cylinder spheres wheat ME
get /home/thumm/masterthesis/code/TrackSort_Neural/results/cylinder_pepper_spheres_wheat/separation_prediction_gating_me/* /home/jakob/code/TrackSort_Neural/results/cylinder_pepper_spheres_wheat/separation_prediction_gating_me/
exit
EOF