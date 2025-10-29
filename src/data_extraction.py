import pandas as pd


# Load data from a CSV file
def load_file(path):
    try:
        # Check if the file has the correct extension (.csv)
        if not path.lower().endswith(".csv"):
            raise ValueError("The file must be a csv")

        data = pd.read_csv(path)
        return data

    except FileNotFoundError:
        print(f"Error: The file at {path} was not found")
        return None


# Check for required columns in the DataFrame
def check_columns(df):
    required_columns = [
        "content",
        "score",
        "reviewId",
        "userName",
        "userImage",
        "thumbsUpCount",
        "reviewCreatedVersion",
        "at",
        "replyContent",
        "repliedAt",
        "sortOrder",
        "appId",
    ]

    for col in required_columns:
        if col not in df.columns:
            print(
                f"Error: Missing required column '{col}' "
                "in the DataFrame"
            )
            return False

    print("All required columns are present")
    return True


if __name__ == "__main__":  # pragma: no cover
    path = "data/dataset.csv"
    df = load_file(path)
    print(check_columns(df))
