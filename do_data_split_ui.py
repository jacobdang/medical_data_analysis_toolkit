import gradio as gr
import pandas as pd
import numpy as np
import os.path as pathlib


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

    np.random.seed(int(random_seed))
    
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
        val_table = orig_table[orig_table[patient_id_column].isin(val_ids)] if val_split_ratio > 0 else None
        test_table = orig_table[orig_table[patient_id_column].isin(test_ids)] if test_split_ratio > 0 else None
    else:
        # If patient_id_column is not provided, split randomly
        msk = np.random.rand(len(orig_table)) < (1 - val_split_ratio - test_split_ratio)
        train_table = orig_table[msk]
        temp_table = orig_table[~msk]

        msk = np.random.rand(len(temp_table)) < (val_split_ratio / (val_split_ratio + test_split_ratio))
        val_table = temp_table[msk] if val_split_ratio > 0 else None
        test_table = temp_table[~msk] if test_split_ratio > 0 else None

    return train_table, val_table, test_table


def main(file, val_split_ratio, test_split_ratio, patient_id_column, random_seed):
    filename = file.name
    print("Loading dataset from " + filename)
    orig_table = pd.read_csv(file)

    # Call to stratified_patient_split 
    train_table, val_table, test_table = stratified_patient_split(orig_table, val_split_ratio, test_split_ratio, patient_id_column or None, random_seed)
    print("Splitting dataset completed")

    basename = pathlib.basename(filename)
    train_table_file = basename + "_train.csv"
    train_table.to_csv(train_table_file, index=False)
    if val_table is not None:
        val_table_file = basename + "_val.csv"
        val_table.to_csv(val_table_file, index=False)
    else:
        val_table_file = None
    if test_table is not None:
        test_table_file = basename + "_test.csv"
        test_table.to_csv(test_table_file, index=False)
    else:
        test_table_file = None
    print("Saving split datasets completed")

    return train_table_file, val_table_file, test_table_file


input_components = [
    gr.File(label="Original Dataset"),
    gr.Number(label="Validation Split Ratio", value=0.2, minimum=0.0, maximum=1.0),
    gr.Number(label="Test Split Ratio", value=0.2, minimum=0.0, maximum=1.0),
    gr.Text(label="Patient ID Column"),
    gr.Number(label="Random Seed", value=0)
]

output_components = [
    gr.File(label="Train Dataset"),
    gr.File(label='Val Dataset'),
    gr.File(label='Test Dataset')
]

gr.Interface(fn=main, inputs=input_components, outputs=output_components, title="Dataset Splitter", analytics_enabled=False, allow_flagging='never').launch(share=True)