import os
import sys
import tempfile
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data_extraction import load_file, check_columns

columns = ["content", "score", "reviewId", "userName", "userImage", "thumbsUpCount",
                        "reviewCreatedVersion", "at", "replyContent", "repliedAt", "sortOrder", "appId"]

def make_valid_df():
    data = {col: ["ok"] for col in columns}
    return pd.DataFrame(data)

def temp_csv_file(df):
    data, path = tempfile.mkstemp(suffix='.csv')
    os.close(data)
    df.to_csv(path, index=False)
    return path

def test_load_file_valid():
    df = make_valid_df()
    path = temp_csv_file(df)
    loaded_df = load_file(path)

    assert loaded_df.equals(df)
    assert loaded_df is not None
    assert isinstance(loaded_df, pd.DataFrame)
    assert not loaded_df.empty
    assert list(loaded_df.columns) == list(df.columns)
    assert loaded_df.iloc[0]["content"] == "ok"
    assert "replyContent" in loaded_df.columns
    
def test_wrong_file_extension():
    data, path = tempfile.mkstemp(suffix='.txt')
    os.close(data)
    try:
        load_file(path)
        assert False, "Expected ValueError for non-csv file"
    except ValueError as e:
        assert "csv" in str(e).lower()
        assert "must" in str(e).lower()
    
def test_missing_file():
    path = "non_existent_file.csv"
    loaded_df = load_file(path)
    
    assert loaded_df is None
    assert loaded_df == None
    
def test_check_columns_valid():
    df = make_valid_df()
    assert check_columns(df) == True
    assert isinstance(check_columns(df), bool)
    assert len(df.columns) == len(columns)
    for col in df.columns:
        assert col in df.columns
        
def test_check_columns_missing():
    df = make_valid_df()
    df = df.drop(columns=["score"])
    assert check_columns(df) == False
    assert "score" not in df.columns
    assert len(df.columns) == len(columns) - 1
    
def test_check_columns_more_missing_columns():
    df = make_valid_df()
    df = df.drop(columns=["score", "userName", "appId"])
    assert check_columns(df) == False
    assert "score" not in df.columns
    assert "userName" not in df.columns
    assert "appId" not in df.columns
    assert len(df.columns) == len(columns) - 3
    
def test_end_to_end():
    df = make_valid_df()
    path = temp_csv_file(df)
    loaded_df = load_file(path)
    assert loaded_df is not None
    assert check_columns(loaded_df) == True
    
    loaded_df = loaded_df.drop(columns=["at"])
    assert check_columns(loaded_df) == False

def run_all_tests():
    test_load_file_valid()
    test_wrong_file_extension()
    test_missing_file()
    test_check_columns_valid()
    test_check_columns_missing()
    test_check_columns_more_missing_columns()
    test_end_to_end()
    print("All tests passed")
    
if __name__ == "__main__":
    print(run_all_tests())