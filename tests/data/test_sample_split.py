import pytest
import pandas as pd
import numpy as np
from ps3.data import create_sample_split

@pytest.mark.parametrize(
    "df, id_column, training_frac, expected_train_frac_range",
    [
        # Test case 1: Small dataset
        (pd.DataFrame({"id": [1, 2, 3, 4, 5]}), "id", 0.8, [0.2, 1.0]),
        # Test case 2: Larger dataset
        (pd.DataFrame({"id": list(range(1, 101))}), "id", 0.9, [0.83, 0.97]),
        # Test case 3: Edge case, training fraction 1.0
        (pd.DataFrame({"id": list(range(1, 21))}), "id", 1.0, [1.0, 1.0]),
        # Test case 4: Edge case, training fraction 0.0
        (pd.DataFrame({"id": list(range(1, 21))}), "id", 0.0, [0.0, 0.0]),
        # Test case 5: Dataset with non-integer IDs
        (pd.DataFrame({"id": ["a", "b", "c", "d", "e"]}), "id", 0.6, [0.2, 1.0]),
        (pd.DataFrame({"id": np.linspace(23, 83, 256)}), "id", 0.3, [0.25, 0.35]),
    ],
)
def test_create_sample_split(df, id_column, training_frac, expected_train_frac_range):
    # Apply the function
    result = create_sample_split(df, id_column, training_frac)

    # Ensure 'sample' column is created
    assert "sample" in result.columns, "'sample' column not found in output DataFrame."

    # Check that all rows are classified as 'train' or 'test'
    assert np.all(result["sample"].isin(["train", "test"])), "Unexpected values in 'sample' column."

    # Validate the training fraction
    train_frac_actual = (result["sample"] == "train").mean()
    assert expected_train_frac_range[0] <= train_frac_actual <= expected_train_frac_range[1], (
        f"Expected train fraction to lie in range {expected_train_frac_range}, "
        f"but got {train_frac_actual}."
    )
