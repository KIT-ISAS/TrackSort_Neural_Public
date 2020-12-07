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
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/me_gating_sep.h5 /home/jakob/code/TrackSort_Neural/models/cylinder/
get /home/thumm/masterthesis/code/TrackSort_Neural/models/cylinder/smape_gating_sep.pkl /home/jakob/code/TrackSort_Neural/models/cylinder/
# pepper cylinder ME
get /home/thumm/masterthesis/code/TrackSort_Neural/models/pepper_and_cylinder/me_gating_sep.pkl /home/jakob/code/TrackSort_Neural/models/pepper_and_cylinder/


## Get Results
# pepper
get /home/thumm/masterthesis/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_se/* /home/jakob/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_se/
get /home/thumm/masterthesis/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_smape/* /home/jakob/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_smape/
get /home/thumm/masterthesis/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_me/* /home/jakob/code/TrackSort_Neural/results/pepper_all/separation_prediction_gating_me/
# cylinder
get /home/thumm/masterthesis/code/TrackSort_Neural/results/cylinder_all/separation_prediction_gating_se/* /home/jakob/code/TrackSort_Neural/results/cylinder_all/separation_prediction_gating_se/
get /home/thumm/masterthesis/code/TrackSort_Neural/results/cylinder_all/separation_prediction_gating_smape/* /home/jakob/code/TrackSort_Neural/results/cylinder_all/separation_prediction_gating_smape/
get /home/thumm/masterthesis/code/TrackSort_Neural/results/cylinder_all/separation_prediction_gating_me/* /home/jakob/code/TrackSort_Neural/results/cylinder_all/separation_prediction_gating_me/
# pepper and cylinder
get /home/thumm/masterthesis/code/TrackSort_Neural/results/pepper_cylinder/separation_prediction_gating_me/* /home/jakob/code/TrackSort_Neural/results/pepper_and_cylinder/separation_prediction_gating_me/
exit
EOF
