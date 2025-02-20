import os
import glob
import tarfile
import argparse
import pandas as pd
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", help="Path to extract csvs")

    print(f'The value of env variable is: {os.environ["ENV"]}')

    args = parser.parse_args()

    url = "https://storage.googleapis.com/ml-pipeline-playground/iris-csv-files.tar.gz"
    # Download the tar file and extract the csv files
    with urllib.request.urlopen(url) as res:
        tarfile.open(fileobj=res, mode="r|gz").extractall(".")

    csv_files = glob.glob("*.csv")

    names = ["X1", "X2", "X3", "X4", "y"]

    df = pd.concat(
        [pd.read_csv(csv_file, header=None, names=names) for csv_file in csv_files],
        ignore_index=True,
    )

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {args.output_path}")
    # Save the pandas df into a csv file
    df.to_csv(f"{args.output_path}/iris.csv", index=False)


if __name__ == "__main__":
    main()
