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

def interactive_main():
    """
    Main function to trigger the dataset splitting process. This function interacts with
    the user to get input for filename, split ratio, patient ID column, and random seed.
    """
    filename = input("Enter the filename of the dataset to be split: ").strip()
    val_split_ratio = float(input("Enter the validation split ratio (default is 0.1): ") or 0.1)
    test_split_ratio = float(input("Enter the test split ratio (default is 0.2): ") or 0.2)
    patient_id_column = input("Enter the name of the patient ID column (leave blank if not applicable): ").strip()
    random_seed = int(input("Enter the random seed for the split (default is 1): ") or 1)
    output_dir = input("Enter the output folder where the results are to be saved: ").strip()

    print("Loading dataset from " + filename)
    orig_table = pd.read_csv(filename)

    # Call to stratified_patient_split 
    train_table, val_table, test_table = stratified_patient_split(orig_table, val_split_ratio, test_split_ratio, patient_id_column or None, random_seed)
    print("Splitting dataset completed")
    basename = pathlib.join(output_dir, pathlib.splitext(pathlib.basename(filename))[0])
    os.makedirs(output_dir, exist_ok=True)
    train_table.to_csv(basename + "_train.csv", index=False)
    if val_table is not None:
        val_table.to_csv(basename + "_val.csv", index=False)
    if test_table is not None:
        test_table.to_csv(basename + "_test.csv", index=False)
    print("Saving split datasets completed")

def main():
    """
    Main function to trigger the dataset splitting process. It takes input arguments for 
    filename, split ratio, patient id column, and other parameters.
    """

    parser = argparse.ArgumentParser(description='Split dataset into train, validation, and test sets.')
    parser.add_argument('filename', type=str, help='File name of the dataset to be split')
    parser.add_argument('--val_split_ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--test_split_ratio', type=float, default=0.2, help='Test split ratio')
    parser.add_argument('--patient_id_column', type=str, help='Name of the patient ID column')
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed for the split')
    parser.add_argument('--output_dir', type=str, help='Output folder where the results are to be saved')

    args = parser.parse_args()

    print("Loading dataset " + args.filename)
    orig_table = pd.read_csv(args.filename)

    train_table, val_table, test_table = stratified_patient_split(orig_table, args.val_split_ratio, args.test_split_ratio, args.patient_id_column, args.random_seed)

    print("Splitting dataset completed")

    basename = pathlib.join(args.output_dir, pathlib.splitext(pathlib.basename(args.filename))[0])
    os.makedirs(args.output_dir, exist_ok=True)
    train_table.to_csv(basename + "train.csv", index=False)
    if val_table is not None:
        val_table.to_csv(basename + "val.csv", index=False)
    if test_table is not None:
        test_table.to_csv(basename + "test.csv", index=False)
    print("Saving split datasets completed")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        interactive_main()