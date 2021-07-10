# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 07:38:30 2021

@author: bb
"""

from tensorflow.keras.models import load_model
import pandas as pd
from os.path import join
from utility_functions import (load_variables,
                              predict_new_matches,
                              predict_nan_for_new_matches)
"""
Importing useful variables
"""
#models
logit_model = load_variables(join("models", "logit_model.pckl"))
rmf_classifier = load_variables(join("models", "rmf_classifier.pckl"))
ANN = load_model(join("models", "ANN.h5"))
#scaler
std_scaler = load_variables(join("objects", "std_scaler.pckl"))
#dataset
atp_data_new_df = load_variables(join("variables", 
                                      "datasets", 
                                      "atp_data_new_df.pckl"))
#labels and features
cat_features_list = load_variables(join("variables", 
                                        "datatypes", 
                                        "cat_features_list.pckl"))
new_cat_features_list = load_variables(join("variables", 
                                            "datatypes", 
                                            "new_cat_features_list.pckl"))
new_features_players_list = load_variables(join("variables", 
                                                "datatypes", 
                                                "new_features_players_list.pckl"))
new_num_features_list = load_variables(join("variables", 
                                            "datatypes", 
                                            "new_num_features_list.pckl"))
num_features_list = load_variables(join("variables", 
                                        "datatypes", 
                                        "num_features_list.pckl"))
temp_feature_type_df = load_variables(join("variables", 
                                           "datatypes", 
                                           "temp_feature_type_df.pckl"))
features_modes_to_drop = load_variables(join("variables", 
                                             "labels_and_features", 
                                             "features_modes_to_drop.pckl"))
features_to_remove = load_variables(join("variables", 
                                         "labels_and_features", 
                                         "features_to_remove.pckl"))
label_mapper = load_variables(join("variables", 
                                   "labels_and_features", 
                                   "label_mapper.pckl"))
X_train_enc_standardized_columns = load_variables(join("variables", 
                                                       "labels_and_features", 
                                                       "X_train_enc_standardized_columns.pckl"))
X_train_encoded_rf_columns = load_variables(join("variables", 
                                                 "labels_and_features", 
                                                 "X_train_encoded_rf_columns.pckl"))
#thresholds
ann_threshold = load_variables(join("variables", 
                                    "thresholds", 
                                    "ann_threshold.pckl"))
lr_threshold = load_variables(join("variables", 
                                   "thresholds", 
                                   "lr_threshold.pckl"))
rf_threshold = load_variables(join("variables", 
                                   "thresholds", 
                                   "rf_threshold.pckl"))

"""
Importing data to predict
"""
X_test_no_target_df=pd.read_csv("test_set_no_target_df.csv", sep=";")
X_test_no_target_df.drop(["score"], axis=1, inplace=True)
X_test_no_target_df.reset_index(inplace=True, drop=True)

"""
Estimating the duration of the match, the number of breakpoints avoided...
"""
new_data_new_matches_dict = predict_nan_for_new_matches(data_to_sample_on_df=atp_data_new_df,
                                                        data_to_pred_df=X_test_no_target_df, 
                                                        n_iterations_for_convergence=5, 
                                                        sample_size=0.01, 
                                                        sample_to_pred_quota=2, 
                                                        target_name="over_under_bet",
                                                        all_variables_type=temp_feature_type_df)

"""
Predicting with all 3 models
"""
model_dict = {"ANN":ANN, "rmf":rmf_classifier, "logit":logit_model}
thresholds_dict = {"logit":lr_threshold, "rmf":rf_threshold, "ANN":ann_threshold}
predictions_df, confidence_scores_df = predict_new_matches(new_data_new_matches_dict, 
                                                           model_dict, 
                                                           new_features_players_list, 
                                                           cat_features_list, 
                                                           num_features_list, 
                                                           new_cat_features_list, 
                                                           new_num_features_list, 
                                                           label_mapper,
                                                           thresholds_dict, 
                                                           std_scaler,
                                                           features_modes_to_drop,
                                                           X_train_enc_standardized_columns,
                                                           X_train_encoded_rf_columns,
                                                           features_to_remove)