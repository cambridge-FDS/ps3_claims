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
    # Create a copy to avoid modifying the original dataframe
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Validate that id_column exists in the dataframe
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in dataframe. "
                        f"Available columns: {df.columns.tolist()}")
    
    # Validate training_frac is between 0 and 1
    if not 0 < training_frac < 1:
        raise ValueError(f"training_frac must be between 0 and 1, got {training_frac}")
    
    # Get the ID column values
    id_values = df[id_column]
    
    # Calculate threshold (0-99 scale)
    # For training_frac=0.8: threshold=80
    # Buckets 0-79 (80 buckets) -> train
    # Buckets 80-99 (20 buckets) -> test
    threshold = int(training_frac * 100)
    
    # Determine bucket assignment based on ID type
    # Check if the ID column contains numeric values
    try:
        # Try to treat as numeric - this works for integer IDs like IDpol
        # Use modulo 100 to get bucket number (0-99)
        bucket = (id_values % 100).astype(int)
    except (TypeError, AttributeError):
        # If modulo fails, ID is likely string type
        # Use hashlib to convert string to integer representation
        def hash_to_bucket(x):
            """Convert string ID to bucket using MD5 hash."""
            # Create MD5 hash of the string
            hash_hex = hashlib.md5(str(x).encode()).hexdigest()
            # Convert hex to integer and take modulo 100
            return int(hash_hex, 16) % 100
        
        bucket = id_values.apply(hash_to_bucket)
    
    # Assign to 'train' or 'test' based on bucket value
    df['sample'] = np.where(bucket < threshold, 'train', 'test')
    
    return df
