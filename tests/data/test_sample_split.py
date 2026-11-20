import pandas as pd
import pytest

from ps3.data import create_sample_column, create_sample_split


def test_assigns_only_train_or_test_and_preserves_rows():
    df = pd.DataFrame({"policy_id": range(100), "x": range(100)})
    result = create_sample_column(df, "policy_id", training_frac=0.8)

    assert len(result) == len(df)
    assert result["sample"].isin(["train", "test"]).all()
    pd.testing.assert_frame_equal(result.drop(columns="sample"), df)


def test_assignments_are_stable_across_calls_and_row_order():
    df = pd.DataFrame({"policy_id": range(100), "x": range(100)})
    first = create_sample_column(df, ["policy_id"]).set_index("policy_id")["sample"]
    second = create_sample_column(df.sample(frac=1, random_state=7), ["policy_id"])
    second = second.set_index("policy_id")["sample"]

    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())


def test_new_records_do_not_change_existing_assignments():
    original = pd.DataFrame({"policy_id": [10, 20, 30]})
    extended = pd.concat([original, pd.DataFrame({"policy_id": [40, 50, 60]})])
    before = create_sample_column(original, "policy_id").set_index("policy_id")["sample"]
    after = create_sample_column(extended, "policy_id").set_index("policy_id")["sample"]

    pd.testing.assert_series_equal(before, after.loc[before.index])


def test_multiple_key_columns_keep_groups_together():
    df = pd.DataFrame(
        {"policy_id": [1, 1, 1, 2], "period": [1, 1, 2, 1], "value": [1, 2, 3, 4]}
    )
    result = create_sample_column(df, ["policy_id", "period"])
    assert result.groupby(["policy_id", "period"])["sample"].nunique().max() == 1
    # The two rows with the same composite key must have the same assignment.
    assert result.loc[0, "sample"] == result.loc[1, "sample"]


@pytest.mark.parametrize("fraction", [0, 1])
def test_fraction_boundaries(fraction):
    result = create_sample_column(pd.DataFrame({"id": range(10)}), "id", fraction)
    expected = "train" if fraction == 1 else "test"
    assert set(result["sample"]) == {expected}


def test_fraction_is_approximately_respected_for_many_ids():
    df = pd.DataFrame({"id": range(10_000)})
    result = create_sample_column(df, "id", training_frac=0.8)
    assert abs((result["sample"] == "train").mean() - 0.8) < 0.03


def test_legacy_entry_point_remains_compatible():
    df = pd.DataFrame({"id": [1, 2, 3]})
    pd.testing.assert_frame_equal(
        create_sample_split(df, "id"),
        create_sample_column(df, "id"),
    )


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        create_sample_column(pd.DataFrame({"id": [1]}), "id", fraction)


def test_rejects_missing_or_empty_keys():
    df = pd.DataFrame({"id": [1]})
    with pytest.raises(ValueError):
        create_sample_column(df, [])
    with pytest.raises(KeyError):
        create_sample_column(df, "unknown")



def test_integer_ids_keep_assignments_when_appended_rows_change_dtype():
    original = pd.DataFrame({"id": range(1, 101)})

    # Appending a fractional ID causes pandas to infer a float column.
    extended = pd.concat(
        [original, pd.DataFrame({"id": [101.5]})],
        ignore_index=True,
    )

    before = create_sample_column(original, "id")
    after = create_sample_column(extended, "id")

    # Compare labels by row position: the original IDs are still in rows 0-99.
    assert before["sample"].tolist() == after.iloc[: len(original)]["sample"].tolist()


def test_integral_float_and_integer_ids_have_same_assignment():
    integer_ids = pd.DataFrame({"id": pd.Series([1, 2], dtype="object")})
    mixed_ids = pd.DataFrame({"id": pd.Series([1.0, 2.0], dtype="object")})

    integer_result = create_sample_column(integer_ids, "id")
    mixed_result = create_sample_column(mixed_ids, "id")

    assert integer_result["sample"].tolist() == mixed_result["sample"].tolist()


def test_reordering_integer_and_integral_float_ids_keeps_assignments():
    first = pd.DataFrame(
        {"id": pd.Series([1, 1.0], dtype="object")}
    )
    reversed_rows = first.iloc[::-1].reset_index(drop=True)

    first_result = create_sample_column(first, "id")
    reversed_result = create_sample_column(reversed_rows, "id")

    assert first_result["sample"].nunique() == 1
    assert reversed_result["sample"].nunique() == 1
    assert first_result["sample"].iloc[0] == reversed_result["sample"].iloc[0]


def test_rejects_missing_and_unsupported_id_values():
    with pytest.raises(ValueError, match="missing"):
        create_sample_column(pd.DataFrame({"id": [1, None]}), "id")

    with pytest.raises(TypeError, match="Unsupported ID value type"):
        create_sample_column(
            pd.DataFrame({"id": [object()]}), "id"
        )
