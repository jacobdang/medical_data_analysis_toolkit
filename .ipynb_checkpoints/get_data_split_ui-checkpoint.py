import gradio as gr
import pandas as pd
import numpy as np
import argparse
import sys
import os.path as pathlib
import os

def stratified_patient_split(orig_table, val_split_ratio=0.1, test_split_ratio=0.2, 
                             patient_id_column=None, random_seed=1):
    """
    Splits a dataset into training, validation, and test sets in a stratified manner. 
    Stratification is done based on patient IDs. All rows for a specific patient will end up in the same set.
    If val_split_ratio or test_split_ratio is None or 0.0, the corresponding dataset will not be returned.
    If patient_id_column is None, each row is assumed to be a separate unique patient and data is split randomly. Otherwise, we will split according to patient_id_column values and ensure that all rows for a specific patient_id_column value will end up in the same set.

    Parameters:
    orig_table (pd.DataFrame): DataFrame to split, with a column for patient IDs.
    val_split_ratio (float): Proportion of data for the validation set.
    test_split_ratio (float): Proportion of data for the test set. 
    patient_id_column (str): Column in orig_table with patient IDs. Default is None.
    random_seed (int): Seed for the random number generator.

    Returns:
    tuple: train_table, val_table, test_table (all pd.DataFrame). If val_split_ratio or test_split_ratio is None or 0.0, the corresponding dataset will not be returned.
    """

    np.random.seed(random_seed)
    
    if patient_id_column:
        # If patient_id_column is provided, split based on patient IDs
        patient_ids = orig_table[patient_id_column].unique()
        np.random.shuffle(patient_ids)

        num_val = round(len(patient_ids) * val_split_ratio)
        num_test = round(len(patient_ids) * test_split_ratio)

        val_ids = patient_ids[:num_val]
        test_ids = patient_ids[num_val:(num_val + num_test)]
        train_ids = patient_ids[(num_val + num_test):]

        train_table = orig_table[orig_table[patient_id_column].isin(train_ids)]
        val_table = orig_table[orig_table[patient_id_column].isin(val_ids)] if val_split_ratio else None
        test_table = orig_table[orig_table[patient_id_column].isin(test_ids)] if test_split_ratio else None
    else:
        # If patient_id_column is not provided, split randomly
        msk = np.random.rand(len(orig_table)) < (1 - val_split_ratio - test_split_ratio)
        train_table = orig_table[msk]
        temp_table = orig_table[~msk]

        msk = np.random.rand(len(temp_table)) < (val_split_ratio / (val_split_ratio + test_split_ratio))
        val_table = temp_table[msk] if val_split_ratio else None
        test_table = temp_table[~msk] if test_split_ratio else None

    return train_table, val_table, test_table

def interactive_main(file, val_split_ratio, test_split_ratio, patient_id_column, random_seed, output_dir):
    filename = file.name
    print("Loading dataset from " + filename)
    orig_table = pd.read_csv(file)

    # Call to stratified_patient_split 
    train_table, val_table, test_table = stratified_patient_split(orig_table, val_split_ratio, test_split_ratio, patient_id_column or None, random_seed)
    print("Splitting dataset completed")

    basename = os.path.join(output_dir, pathlib.Path(filename).stem)
    os.makedirs(output_dir, exist_ok=True)
    train_table_file = basename + "_train.csv"
    train_table.to_csv(train_table_file, index=False)
    if val_table is not None:
        val_table_file = basename + "_val.csv"
        val_table.to_csv(val_table_file, index=False)
    if test_table is not None:
        test_table_file = basename + "_test.csv"
        test_table.to_csv(test_table_file, index=False)
    print("Saving split datasets completed")

    return train_table_file, val_table_file, test_table_file

def main(file, val_split_ratio=0.1, test_split_ratio=0.2, patient_id_column="", random_seed=1, output_dir="output"):
    train_file, val_file, test_file = interactive_main(file, val_split_ratio, test_split_ratio, patient_id_column, random_seed, output_dir)
    return train_file, val_file, test_file

def download_files(train_file, val_file, test_file):
    return {
        "train_dataset": train_file,
        "val_dataset": val_file,
        "test_dataset": test_file
    }

input_components = [
    gr.File(label="Dataset"),
    gr.Number(label="Validation Split Ratio"),
    gr.Number(label="Test Split Ratio"),
    gr.Text(label="Patient ID Column"),
    gr.Number(label="Random Seed"),
    gr.Textbox(label="Output Folder")
]

#output_component = gr.Download(label="Download Split Datasets")

gr.Interface(fn=main, inputs=input_components, outputs=None, title="Dataset Splitter", analytics_enabled=False).launch(share=True)