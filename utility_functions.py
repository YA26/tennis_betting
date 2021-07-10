# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 13:39:30 2021

@author: bb
"""

"""

Utility functions

"""
import pickle
from MissingValuesHandler.missing_data_handler import RandomForestImputer
import pandas as pd
import numpy as np
import statsmodels.api as sms
from sklearn.model_selection import train_test_split

def load_variables(path):
    '''
    Load variables with pickle
    '''
    loaded_variable=None
    with open(path, "rb") as file:
        loaded_variable=pickle.load(file)
    return loaded_variable

def compute_new_numerical_features(data_df, new_features_players_list, num_features_list):
    """
    calculates the difference in age, height, rank and other numerical variables for both players
    """
    num_features_copy_list = num_features_list.copy()
    data_copy_df = data_df.copy()
    for line in new_features_players_list:
        first_feature = line[0]
        second_feature = line[1]
        third_feature = line[2]
        num_features_copy_list.append(first_feature)
        num_features_copy_list.remove(second_feature)
        num_features_copy_list.remove(third_feature)
        data_copy_df[first_feature] = abs(data_copy_df[second_feature] - data_copy_df[third_feature])
        data_copy_df.drop([second_feature, third_feature], inplace=True, axis=1)
    w_bp_pct = (data_copy_df["w_bpSaved"]/data_copy_df["w_bpFaced"])*100
    l_bp_pct = (data_copy_df["l_bpSaved"]/data_copy_df["l_bpFaced"])*100
    data_copy_df["bpSaved_pct_diff"] = abs(w_bp_pct-l_bp_pct)
    data_copy_df["bpSaved_pct_diff"] = data_copy_df["bpSaved_pct_diff"].fillna(0)
    data_copy_df["bpSaved_pct_diff"] = data_copy_df["bpSaved_pct_diff"].replace([np.inf, -np.inf], 0)
    num_features_copy_list.append("bpSaved_pct_diff")
    data_copy_df["bpSaved_pct_diff"] = data_copy_df["bpSaved_pct_diff"].round(1)
    for feature_name in ["w_bpSaved", "w_bpFaced", "l_bpSaved", "l_bpFaced"]:
        num_features_copy_list.remove(feature_name)
        data_copy_df.drop([feature_name], inplace=True, axis=1)
    return data_copy_df, num_features_copy_list

def map_columns(mapper_data_columns, mapped_data):
    """
    makes sure the structure of the dataset(the chosen features) is well kept for every given model
    """
    X_df = pd.DataFrame([])
    for column in mapper_data_columns:
        if column in mapped_data.columns:
            X_df[column] = mapped_data[column]
        else:
            X_df[column] = [0] * len(mapped_data)
    return X_df

def predict_nan_for_new_matches(data_to_sample_on_df, 
                                data_to_pred_df, 
                                n_iterations_for_convergence, 
                                sample_size, 
                                sample_to_pred_quota, 
                                target_name,
                                all_variables_type=pd.DataFrame([])):
    data_to_sample_on_copy_df = data_to_sample_on_df.copy()
    data_to_pred_copy_df = data_to_pred_df.copy()
    target_variable_df = data_to_sample_on_copy_df[target_name]
    new_data_list = []
    samples_to_pred_selected = 0
    while len(data_to_pred_copy_df)!=0:
        _, min_data_df = train_test_split(data_to_sample_on_copy_df, 
                                          test_size=sample_size, 
                                          random_state=42, 
                                          stratify=target_variable_df)
        min_data_to_pred_df = None
        print(f" Samples remaining to fill and predict: {len(data_to_pred_copy_df)}")
        if len(data_to_pred_copy_df) > sample_to_pred_quota:
            sample_to_pred_chosen_df = data_to_pred_copy_df[:sample_to_pred_quota]
            min_data_to_pred_df = min_data_df.append(sample_to_pred_chosen_df)
            indeces = list(sample_to_pred_chosen_df.index)
            data_to_pred_copy_df.drop(indeces, axis=0, inplace=True) 
            samples_to_pred_selected = sample_to_pred_quota
        else:
            samples_to_pred_selected = len(data_to_pred_copy_df) 
            min_data_to_pred_df = min_data_df.append(data_to_pred_copy_df)
            indeces = list(data_to_pred_copy_df.index)
            data_to_pred_copy_df.drop(indeces, axis=0, inplace=True)
        min_data_to_pred_df.reset_index(inplace=True, drop=True)
        random_forest_imputer = RandomForestImputer(data=min_data_to_pred_df,
                                                    target_variable_name=target_name,
                                                    n_iterations_for_convergence=n_iterations_for_convergence)
        #Setting the ensemble model parameters: it could be a random forest regressor or classifier
        random_forest_imputer.set_ensemble_model_parameters(n_estimators=40, additional_estimators=10)
        if not all_variables_type.empty:
            random_forest_imputer.set_features_type_predictions(all_variables_type.drop([target_name], axis=0))
            pred_target_variable_type = pd.DataFrame(all_variables_type.loc[target_name], columns=["Predictions"])
            random_forest_imputer.set_target_variable_type_prediction(pred_target_variable_type)
        #Launching training and getting our new dataset
        new_data_df = random_forest_imputer.train()
        new_data_list.append(new_data_df.tail(samples_to_pred_selected)) 
    new_data_to_pred_df = pd.concat(new_data_list, axis=0)
    new_data_to_pred_df.reset_index(inplace=True, drop=True)
    return new_data_to_pred_df

def predict_new_matches_logit(model, data, threshold, label_mapper):
    y_logit_pred_proba = model.predict(data) 
    y_logit_pred_bin = (y_logit_pred_proba > threshold).astype(int)
    y_logit_pred_bin = y_logit_pred_bin.to_numpy()
    y_logit_pred = [label_mapper[value] for value in y_logit_pred_bin]
    return y_logit_pred, y_logit_pred_bin, y_logit_pred_proba

def predict_new_matches_ANN(model, data, threshold, label_mapper):
    y_ann_pred_proba = model.predict(data)
    y_ann_pred_proba = np.array(y_ann_pred_proba)
    y_ann_pred_proba = y_ann_pred_proba.reshape(len(y_ann_pred_proba))
    y_ann_pred_bin = (y_ann_pred_proba > threshold).astype(int)
    y_ann_pred = [label_mapper[value] for value in y_ann_pred_bin]
    return y_ann_pred, y_ann_pred_bin, y_ann_pred_proba

def predict_new_matches_rf(model, data, threshold, label_mapper):
    y_rf_pred_proba = model.predict_proba(data)[:,0]
    y_rf_pred_bin = (y_rf_pred_proba > threshold).astype(int)
    y_rf_pred =[label_mapper[value] for value in y_rf_pred_bin]
    return y_rf_pred, y_rf_pred_bin, y_rf_pred_proba

def predict_new_matches(data_df, 
                        model_dict,
                        new_features_players_list, 
                        cat_features_list, 
                        num_features_list, 
                        new_cat_features_list, 
                        new_num_features_list, 
                        label_mapper_dict,
                        threshold_dict,
                        std_scaler,
                        features_modes_to_drop_list,
                        X_train_enc_standardized_columns_list,
                        X_train_encoded_rf_columns_list,
                        features_to_remove_list):
    X_new_df, _ = compute_new_numerical_features(data_df, new_features_players_list, num_features_list)
    X_new_df.drop(features_to_remove_list, axis=1, inplace=True)
    X_new_encoded_df = pd.get_dummies(X_new_df, columns=new_cat_features_list)
    features_rem_to_drop_set = set(X_new_encoded_df).intersection(set(features_modes_to_drop_list))
    features_rem_to_drop_list = list(features_rem_to_drop_set)
    X_new_enc_dropped_df = X_new_encoded_df.drop(features_rem_to_drop_list, axis=1)
    X_new_enc_dropped_df = sms.add_constant(X_new_enc_dropped_df) 
    X_new_enc_standardized_df = X_new_enc_dropped_df.copy(deep=True)
    X_new_enc_standardized_df[new_num_features_list] = std_scaler.transform(X_new_enc_standardized_df[new_num_features_list])
    X_logit_ann_df = map_columns(mapper_data_columns=X_train_enc_standardized_columns_list, mapped_data=X_new_enc_standardized_df)
    X_rf_df = map_columns(mapper_data_columns=X_train_encoded_rf_columns_list, mapped_data=X_new_encoded_df)   
    #Logit
    y_logit_pred, y_logit_pred_bin, y_logit_pred_proba = predict_new_matches_logit(model_dict["logit"], 
                                                                                   X_logit_ann_df, 
                                                                                   threshold_dict["logit"],
                                                                                   label_mapper_dict)
    #ANN
    y_ann_pred, y_ann_pred_bin, y_ann_pred_proba = predict_new_matches_ANN(model_dict["ANN"], 
                                                                           X_logit_ann_df, 
                                                                           threshold_dict["ANN"], 
                                                                           label_mapper_dict)
    #Random forest
    y_rf_pred, y_rf_pred_bin, y_rf_pred_proba = predict_new_matches_rf(model_dict["rmf"], 
                                                                       X_rf_df,
                                                                       threshold_dict["rmf"],
                                                                       label_mapper_dict)
    #Hard vote with stacking
    y_sum = y_rf_pred_bin + y_logit_pred_bin + y_ann_pred_bin
    y_final_bin = (y_sum >= 2).astype(int)
    y_final = [label_mapper_dict[value] for value in y_final_bin] 
    #We gather everything
    logit_thresh = round(float(threshold_dict['logit']), 3)
    ANN_thresh = round(float(threshold_dict['ANN']), 3)
    rmf_thresh = round(float(threshold_dict['rmf']), 3)
    y_df = pd.DataFrame({"final_predictions":y_final, "logit":y_logit_pred, "ANN":y_ann_pred, "random forest":y_rf_pred})
    y_proba_df = pd.DataFrame({f"logit-threshold: {logit_thresh}":y_logit_pred_proba,
                               f"ANN-threshold: {ANN_thresh}":y_ann_pred_proba,
                               f"random forest-threshold: {rmf_thresh}":y_rf_pred_proba})
    return y_df, y_proba_df