import pandas as pd
import hashlib
import numpy as np
from typing import Union


def hash_id(id_value: Union[str, int, float]) -> int:
        hash_obj = hashlib.sha256(str(id_value).encode())
        hash_value = int(hash_obj.hexdigest(), 16) % 100
        return hash_value


def create_sample_split(df: pd.DataFrame, id_column: Union[str, int, float], training_frac: float = 0.9) -> pd.DataFrame:
    """
    Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str or int or float
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.8

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    hashes = df[id_column].apply(hash_id)
    train_threshold = int(training_frac * 100)
    df['sample'] = np.where(hashes < train_threshold, 'train', 'test')
    return df
