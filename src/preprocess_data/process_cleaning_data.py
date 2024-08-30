import pandas as pd
import argparse


TEXT_COLUMNS_NAMES = ["sent_context", "CM", "relation", "subrelation"]
LABEL_COLUMNS_NAMES = ["relation_presence", "subrelation_presence"]


def order_data(df):
    new_order = [
        "cm_id",
        "example_id",
        "relation",
        "subrelation",
        "field",
        "context",
        "sent_context",
        "CM",
        "relation_presence",
        "subrelation_presence",
        "POS_Tags",
        "NER_Tags",
        "Lemmas",
        "Dependencies",
    ]
    df = df[new_order]
    return df


def typos_norm(df, col):
    df[col] = df[col].str.strip()  # blank deleted
    df[col] = df[col].str.lower()  # lower data
    df[col] = df[col].replace(
        {" de el ": " del ", " De el ": " Del ", " a el ": " al ", " A el ": " Al "},
        regex=True,
    )
    df.fillna("not_applicable", inplace=True)
    return df


def create_label2id(df, column_name):
    """
    This function creates a label2id dictionary for a sentence classification task.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column_name (str): The name of the column containing the labels.

    Returns:
    dict: A dictionary mapping each unique label to a unique ID.
    """
    # Expected labels
    expected_labels = ["oui", "non", "not_applicable", "oui+", "autre", "NA"]

    # Get labels present in the data
    unique_labels = df[column_name].unique()

    # Verify that each present label in the expected_labels
    for label in unique_labels:
        if label not in expected_labels:
            raise ValueError(f"New label found: {label}")

    # Create el label2id
    label2id = {label: idx for idx, label in enumerate(expected_labels)}

    return label2id


def transform_labels(df, column_name, label2id):
    """
    This function transforms the labels in a specified column of a dataframe
    using a provided label2id dictionary.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column_name (str): The name of the column containing the labels to be transformed.
    label2id (dict): A dictionary mapping labels to their respective IDs.

    Returns:
    pd.DataFrame: The dataframe with the transformed column.
    """
    # Verify if column name exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")

    # Verify if all the labels aer present in label2id
    unique_labels = df[column_name].unique()
    for label in unique_labels:
        if label not in label2id:
            raise ValueError(f"Label unexpected found: {label}")

    # Transform the column values using the label2id
    df[f"{column_name}_id"] = df[column_name].map(label2id)

    return df


def data_cleaning_normalization(df, text_columns_name, label_columns_name):
    """
    This function performs data cleaning and normalization on the given dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    text_columns_name (list): A list of column names containing text data to be cleaned and normalized.
    label_columns_name (list): A list of column names containing labels to be transformed using label2id.

    Returns:
    pd.DataFrame: The cleaned and normalized dataframe.
    """
    # Ensure that the input dataframe is not modified in place
    c_df = df.copy()

    # Order the data (assuming 'order_data' is a defined function)
    c_df = order_data(c_df)

    # Process text columns: clean typos and normalize
    for col in text_columns_name:
        if col in c_df.columns:
            c_df = typos_norm(c_df, col)
        else:
            print(f"Warning: Column '{col}' not found in the dataframe.")

    # Process label columns: create label2id and transform labels
    for col2 in label_columns_name:
        if col2 in c_df.columns:
            label2id = create_label2id(c_df, f"{col2}")
            c_df = transform_labels(c_df, f"{col2}", label2id)
        else:
            print(f"Warning: Column '{col2}' not found in the dataframe.")

    return c_df


if __name__ == "__main__":
    # args parser config
    xparser = argparse.ArgumentParser(description="Cleaning data from a csv file.")
    xparser.add_argument("data", help="Path to the file to preprocess")
    args = xparser.parse_args()
    # function use
    df = pd.read_csv(args.data)
    cleaned_data = data_cleaning_normalization(
        df, TEXT_COLUMNS_NAMES, LABEL_COLUMNS_NAMES
    )
    cleaned_data.to_csv("data/cleaned_data.csv", index=False)
    print("data saved as parsed_data")
