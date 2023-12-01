import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r
import rpy2.robjects as ro
import rpy2.situation
import pathlib

# Enable automatic conversion from pandas to R
pandas2ri.activate()

# Import R packages
mice = importr('mice')
stringr = importr('stringr')

def data_imputation(imputation_type, train_file, val_file, test_file, numerical_vars, categorical_vars, output_path):
    """
    Perform specified type of data imputation on the given dataset.
    
    Args:
    imputation_type (str): The type of imputation - "remove", "simple", or "multiple".
    train_file (str): Path to the training data CSV file.
    val_file (str): Path to the validation data CSV file.
    test_file (str): Path to the testing data CSV file.
    numerical_vars (list): List of column headers indicating numerical variables.
    categorical_vars (list): List of column headers indicating categorical variables.
    output_path (str): Directory to save the imputation results.
    
    Raises:
    ValueError: If an unsupported imputation type is provided.
    """
    train_table = pd.read_csv(train_file)
    val_table = pd.read_csv(val_file)
    test_table = pd.read_csv(test_file)

    if imputation_type == 'remove':
        param_nan_counts = np.sum(pd.isna(train_table), axis=1)
        train_table_no_nan = train_table[param_nan_counts == 0]
        train_table_no_nan.to_csv(pathlib.join(output_path, 'train_no_nan.csv'), index=False)
        
        param_nan_counts = np.sum(pd.isna(val_table), axis=1)
        val_table_no_nan = val_table[param_nan_counts == 0]
        val_table_no_nan.to_csv(pathlib.join(output_path, 'val_no_nan.csv'), index=False)
        
        param_nan_counts = np.sum(pd.isna(test_table), axis=1)
        test_table_no_nan = test_table[param_nan_counts == 0]
        test_table_no_nan.to_csv(pathlib.join(output_path, 'test_no_nan.csv'), index=False)
    
    elif imputation_type == 'simple':
        numerical_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        numerical_imputer.fit(train_table[numerical_vars])
        train_table[numerical_vars] = numerical_imputer.transform(train_table[numerical_vars])
        val_table[numerical_vars] = numerical_imputer.transform(val_table[numerical_vars])
        test_table[numerical_vars] = numerical_imputer.transform(test_table[numerical_vars])
        
        categorical_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        categorical_imputer.fit(train_table[categorical_vars])
        train_table[categorical_vars] = categorical_imputer.transform(train_table[categorical_vars])
        val_table[categorical_vars] = categorical_imputer.transform(val_table[categorical_vars])
        test_table[categorical_vars] = categorical_imputer.transform(test_table[categorical_vars])
        
        train_table.to_csv(pathlib.join(output_path, 'train_simple_imputed.csv'), index=False)
        val_table.to_csv(pathlib.join(output_path, 'val_simple_imputed.csv'), index=False)
        test_table.to_csv(pathlib.join(output_path, 'test_simple_imputed.csv'), index=False)
    
    elif imputation_type == 'multiple':
    if imputation_type == 'multiple':
        # Convert Python lists to R vectors
        categorical_vars_r = ro.StrVector(categorical_vars)
        numerical_vars_r = ro.StrVector(numerical_vars)

        # Define the R function for multiple imputation using MICE
        r("""
        library(mice)
        library(stringr)

        function(train_file, val_file, test_file, numerical_vars, categorical_vars, output_path, lm_fit_summary_csv, lm_fit_pool_summary, save_model_name) {
            train <- read.csv(train_file, na.strings = c('NaN', '', 'nan'))
            val <- read.csv(val_file, na.strings = c('NaN', '', 'nan'))
            test <- read.csv(test_file, na.strings = c('NaN', '', 'nan'))

            # Add a column to each dataset to identify which set it belongs to
            train["set"] <- "train"
            val["set"] <- "val"
            test["set"] <- "test"

            # Merge the datasets together
            full_data <- rbind(train, val, test)

            for (col in numerical_vars) {
                full_data[[col]] <- as.numeric(as.character(full_data[[col]]))
            }

            for (col in categorical_vars) {
                full_data[[col]] <- as.factor(full_data[[col]])
            }

            # Perform the imputation
            imputed_data <- mice(full_data, m=5, maxit=50, method='pmm', seed=500)

            # Split the datasets apart again
            imputed_train <- imputed_data[imputed_data$set=="train", ]
            imputed_val <- imputed_data[imputed_data$set=="val", ]
            imputed_test <- imputed_data[imputed_data$set=="test", ]

            # Write the imputed datasets to csv
            write.csv(imputed_train, file.path(output_path, "imputed_train.csv"), row.names=FALSE)
            write.csv(imputed_val, file.path(output_path, "imputed_val.csv"), row.names=FALSE)
            write.csv(imputed_test, file.path(output_path, "imputed_test.csv"), row.names=FALSE)

            # Return the imputed data
            return(imputed_data)
        }""")(train_file, val_file, test_file, numerical_vars_r, categorical_vars_r, output_path, lm_fit_summary_csv, lm_fit_pool_summary, save_model_name)
    
    else:
        raise ValueError("Unsupported imputation_type. Choose from 'remove', 'simple', or 'multiple'.")