import hashlib

import numpy as np

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

    # Define the split threshold 
    train_threshold = training_frac * 100

    # Create a stable hash bin (0-99) for each ID
    bins = df[id_column].apply(lambda x: 
        int(hashlib.sha256(str(x).encode()).hexdigest()[:8], 16) % 100
    )

    # Assign 'train' or 'test'
    df['sample'] = np.where(bins < train_threshold, 'train', 'test')
    return df
