import pytest
import os
import shutil
import pandas as pd
from module_name.data_preprocesing import DataPreprocessor


@pytest.fixture
def data_preprocessor():
    return DataPreprocessor("intergration_test")


def test_adjust_column_names(data_preprocessor):
    df = pd.DataFrame({"A.B": [1, 2], "C(D)": [3, 4]})
    expected_columns = ["A_B", "C_D_"]
    adjusted_df, column_dict = data_preprocessor.adjust_column_names(df)
    assert list(adjusted_df.columns) == expected_columns
    assert column_dict == {"A.B": "A_B", "C(D)": "C_D_"}


def test_downcast_int_columns(data_preprocessor):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, dtype="int64")
    expected_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, dtype="int8")
    downcasted_df = data_preprocessor.downcast_int_columns(df)
    assert downcasted_df.equals(expected_df)


def test_downcast_float_columns(data_preprocessor):
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, dtype="float64")
    expected_df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, dtype="float32")
    downcasted_df = data_preprocessor.downcast_float_columns(df)
    assert downcasted_df.equals(expected_df)


def test_convert_object_to_category(data_preprocessor):
    df = pd.DataFrame(
        {
            "A": ["a", "b", "a", "a", "a"],
            "B": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"],
        }
    )
    expected_df = pd.DataFrame(
        {
            "A": pd.Series(["a", "b", "a", "a", "a"], dtype="category"),
            "B": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"]
            ),
        }
    )
    converted_df = data_preprocessor.convert_object_to_category(df)
    print(converted_df.dtypes)
    assert converted_df.equals(expected_df)


def test_convert_float_to_int(data_preprocessor):
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, dtype="float32")
    expected_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, dtype="int8")
    converted_df = data_preprocessor.convert_float_to_int(df)
    print(converted_df.dtypes)
    assert converted_df.equals(expected_df)


def test_optimize_dtypes(data_preprocessor):
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [3.2, 4.0, 5.0, 6.0, 7.0],
            "C": ["a", "b", "a", "a", "a"],
        }
    )
    optimized_df = data_preprocessor.optimize_dtypes(df, verbose=False)

    print(optimized_df.dtypes)
    assert optimized_df.dtypes["A"] == "int8"
    assert optimized_df.dtypes["B"] == "float32"
    assert optimized_df.dtypes["C"] == "category"


def test_store_df(data_preprocessor):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    filename = "test_df.pkl"
    data_preprocessor.store_df(df, filename)
    stored_df = pd.read_pickle(f"{data_preprocessor.df_path}/{filename}")
    assert df.equals(stored_df)
    # remove directory data_preprocessor.df_path
    shutil.rmtree(data_preprocessor.df_path)
