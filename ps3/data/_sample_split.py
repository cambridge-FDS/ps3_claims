import pandas as pd
import hashlib
import numpy as np


# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.

def create_sample_split(df, id_column, training_frac=0.8):
    """
    Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    # Generate a hash for each key and convert it to an integer
    def hash_id(id_value):
        hash_obj = hashlib.sha256(str(id_value).encode())
        return int(hash_obj.hexdigest(), 16) %100

    hashes = df[id_column].apply(hash_id)
    train_threshold = int(training_frac * 100)
    df['sample'] = np.where(hashes < train_threshold, 'train', 'test')

    return df
