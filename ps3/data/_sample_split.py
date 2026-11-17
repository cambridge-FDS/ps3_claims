import hashlib

import numpy as np
import pandas as pd


def _string_to_bucket(value):
    """Map a string ID to a deterministic integer in [0, 100)."""
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16) % 100


def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Each row is assigned to the training or test set depending on the ID's value
    modulo 100: IDs with ``id % 100 < training_frac * 100`` go to the training
    set, all others to the test set. The assignment is deterministic, so the
    same ID always ends up in the same sample, even if the data changes. String
    IDs are first converted to an integer with a hash.

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
        Copy of the data with an additional ``sample`` column containing
        "train" or "test", based on the IDs.
    """
    if not 0 <= training_frac <= 1:
        raise ValueError(f"training_frac must be between 0 and 1, got {training_frac}")

    ids = df[id_column]
    if pd.api.types.is_numeric_dtype(ids):
        bucket = ids.astype("int64") % 100
    else:
        bucket = ids.astype(str).map(_string_to_bucket)

    df = df.copy()
    df["sample"] = np.where(bucket < round(training_frac * 100), "train", "test")
    return df
