import pandas as pd
from utils.data_cleaning import clean_dataset


def test_gentle_mode_preserves_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2, 2, 4],
            "name": ["a", "b", "b", None],
            "value": [10, None, 30, 5000],
            "target": [0, 1, 1, 0],
        }
    )

    cleaned, report = clean_dataset(
        df,
        target_col="target",
        preserve_structure=True,
        cleaning_policy={"mode": "gentle"},
        return_report=True,
    )

    assert list(cleaned.columns) == list(df.columns), "Gentle mode should preserve columns"
    assert report["policy"]["mode"] == "gentle"


if __name__ == "__main__":
    test_gentle_mode_preserves_columns()
    print("gentle mode test passed")
