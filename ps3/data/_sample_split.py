import hashlib
import numpy as np
import pandas as pd

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """

def create_sample_split(
    df: pd.DataFrame,
    id_column: str,
    training_frac: float = 0.8,
) -> pd.DataFrame:
    """Return a copy with an ID-based train/test 'sample' column.

    training_frac is the approximate fraction of IDs used for training.
    """
    if not 0 < training_frac < 1:
        raise ValueError("training_frac must be between 0 and 1.")

    if df[id_column].isna().any():
        raise ValueError("The ID column must not contain missing values.")

    df = df.copy()

    def assign_sample(identifier: str) -> str:
        # Convert the ID to bytes for hashing.
        id_bytes = identifier.encode("utf-8")

        # Convert the hash to an integer.
        hash_value = int(hashlib.sha256(id_bytes).hexdigest(), 16)

        bucket = hash_value % 10000

        if bucket < training_frac * 10000:
            return "train"

        return "test"

    # Convert IDs to strings before passing them to assign_sample.
    df["sample"] = df[id_column].astype(str).apply(assign_sample)

    return df