import pytest
import pandas as pd
from process_cleaning_data import data_cleaning_normalization

# Global variables
TEXT_COLUMNS_NAMES = ["CM"]
LABEL_COLUMNS_NAMES = ["subrelation_presence"]


@pytest.fixture
def sample_data():
    # Create an example dumb dataframe
    df = pd.read_csv("./data/parsed_data.csv")
    return df.head(10)


def test_data_cleaning_normalization(sample_data):
    # Test cleaning process
    sample_data.fillna("not_applicable", inplace=True)
    cleaned_df = data_cleaning_normalization(
        sample_data, TEXT_COLUMNS_NAMES, LABEL_COLUMNS_NAMES
    )

    # Verify that function outputs it expected
    assert cleaned_df is not None, "The cleaned DataFrame should not be None"

    # Verify that labels have been transformed as expected
    expected_labels = {
        "oui": 0,
        "non": 1,
        "not_applicable": 2,
        "oui+": 3,
        "autre": 4,
        "NA": 5,
    }

    transformed_labels = sample_data["subrelation_presence"].map(expected_labels)
    assert all(
        cleaned_df["subrelation_presence_id"] == transformed_labels
    ), "Label transformation did not work as expected"
