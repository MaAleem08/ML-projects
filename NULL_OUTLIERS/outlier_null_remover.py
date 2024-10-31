#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def preprocess_dataset(dataframe, cols, handle_missing_values=True, 
                      missing_value_cols=None, missing_value_method='mean', 
                      handle_outliers=True, outlier_cols=None, 
                      outlier_removal_method='iqr', 
                      numerical_cols=None, categorical_cols=None):
    
    # Check if missing_value_cols and outlier_cols are provided
    if missing_value_cols is None:
        missing_value_cols = []
    if outlier_cols is None:
        outlier_cols = []

    # Handling missing values
    if handle_missing_values:
        for col in missing_value_cols:
            if col in dataframe.columns:
                if col in numerical_cols:
                    if missing_value_method == 'mean':
                        dataframe[col].fillna(dataframe[col].mean(), inplace=True)
                    elif missing_value_method == 'median':
                        dataframe[col].fillna(dataframe[col].median(), inplace=True)
                elif col in categorical_cols:
                    # Use mode to fill missing values for categorical columns
                    mode_value = dataframe[col].mode()[0]
                    dataframe[col].fillna(mode_value, inplace=True)

    # Handling outliers
    if handle_outliers:
        for col in outlier_cols:
            if col in dataframe.columns:
                if outlier_removal_method == 'iqr':
                    q1 = dataframe[col].quantile(0.25)
                    q3 = dataframe[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    dataframe = dataframe[(dataframe[col] >= lower_bound) & (dataframe[col] <= upper_bound)]
                elif outlier_removal_method == 'z_score':
                    z_scores = (dataframe[col] - dataframe[col].mean()) / dataframe[col].std()
                    dataframe = dataframe[(z_scores.abs() <= 3)]  # Keep only rows within 3 standard deviations

    return dataframe

