import os
import argparse
from typing import Optional

import pandas as pd
import xgboost


def _make_parent_dirs_and_return_path(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path


def xgboost_train(
    training_data_path: str,
    model_path: str,
    model_config_path: str,
    label_column: int = 4,
    num_iterations: int = 10,
    num_class: int = 3,
    objective: str = "multi:softmax",
    booster: str = "gbtree",
    learning_rate: float = 0.3,
    min_split_loss: int = 0,
    max_depth: int = 6,
) -> None:
    """Train an XGBoost model.

    Parameters
    ==========
    training_data_path: str
        Path for the training data in CSV format.
    model_path: str
        Output path for the trained model in binary XGBoost format.
    model_config_path: str
        Output path for the internal parameter configuration of Booster as a JSON string.
    label_column: int
        Column containing the label data.
    num_iterations: int
        Number of boosting iterations.
    """

    df = pd.read_csv(training_data_path, header=0)

    training_data = xgboost.DMatrix(
        data=df.drop(columns=[df.columns[label_column]]),
        label=df[df.columns[label_column]],
    )

    booster_params = {}
    booster_params.setdefault("num_class", num_class)
    booster_params.setdefault("objective", objective)
    booster_params.setdefault("booster", booster)
    booster_params.setdefault("learning_rate", learning_rate)
    booster_params.setdefault("min_split_loss", min_split_loss)
    booster_params.setdefault("max_depth", max_depth)

    model = xgboost.train(
        params=booster_params,
        dtrain=training_data,
        num_boost_round=num_iterations,
    )

    # Saving the model in binary format
    model.save_model(model_path)

    model_config_str = model.save_config()
    with open(model_config_path, "w") as model_config_file:
        model_config_file.write(model_config_str)


if __name__ == "__main__":

    _parser = argparse.ArgumentParser(
        prog="Xgboost train", description="Train an XGBoost model"
    )
    _parser.add_argument(
        "--training-data",
        dest="training_data_path",
        type=str,
        required=True,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--label-column",
        dest="label_column",
        type=int,
        required=False,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--num-class",
        dest="num_class",
        type=int,
        required=False,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--num-iterations",
        dest="num_iterations",
        type=int,
        required=False,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--objective",
        dest="objective",
        type=str,
        required=False,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--booster", dest="booster", type=str, required=False, default=argparse.SUPPRESS
    )
    _parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        required=False,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--min-split-loss",
        dest="min_split_loss",
        type=float,
        required=False,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        required=False,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--model",
        dest="model_path",
        type=_make_parent_dirs_and_return_path,
        required=True,
        default=argparse.SUPPRESS,
    )
    _parser.add_argument(
        "--model-config",
        dest="model_config_path",
        type=_make_parent_dirs_and_return_path,
        required=True,
        default=argparse.SUPPRESS,
    )
    _parsed_args = vars(_parser.parse_args())

    _outputs = xgboost_train(**_parsed_args)
