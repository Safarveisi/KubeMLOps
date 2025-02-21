import os
import argparse

import numpy
import pandas
import xgboost


def _make_parent_dirs_and_return_path(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path


def xgboost_predict(
    data_path: str,
    model_path: str,
    predictions_path: str,
    label_column: int = 4,
) -> None:
    """Make predictions using a trained XGBoost model.

    Parameters
    ==========
    data_path: str
        Path for the feature data in CSV format.
    model_path: str
        Path for the trained model in binary XGBoost format.
    predictions_path: str
        Output path for the predictions.
    label_column: int
        Column containing the label data.
    """

    df = pandas.read_csv(data_path, header=0)

    df = df.drop(columns=[df.columns[label_column]])

    testing_data = xgboost.DMatrix(
        data=df,
    )

    model = xgboost.Booster(model_file=model_path)

    predictions = model.predict(testing_data)

    numpy.savetxt(predictions_path, predictions)

if __name__ == "__main__":

    _parser = argparse.ArgumentParser(
        prog="Xgboost predict",
        description="Make predictions using a trained XGBoost model.",
    )
    _parser.add_argument(
        "--data", dest="data_path", type=str, required=True, default=argparse.SUPPRESS
    )
    _parser.add_argument(
        "--model", dest="model_path", type=str, required=True, default=argparse.SUPPRESS
    )
    _parser.add_argument(
        "--label-column",
        dest="label_column",
        type=int,
        required=False,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--predictions",
        dest="predictions_path",
        type=_make_parent_dirs_and_return_path,
        required=True,
        default=argparse.SUPPRESS,
    )
    _parsed_args = vars(_parser.parse_args())

    _outputs = xgboost_predict(**_parsed_args)
