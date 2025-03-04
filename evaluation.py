""" Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. """
""" SPDX-License-Identifier: MIT-0 """

import json
import os
import pathlib
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(level=logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting Evaluation ...")
    logger.info("Reading Test Predictions")
    y_pred_path = "/opt/ml/processing/input/predictions/x_test.csv.out"
    y_pred = pd.read_csv(y_pred_path, header=None).squeeze()  # Assuming one column
    logger.info("Reading Test Labels")
    y_true_path = "/opt/ml/processing/input/true_labels/y_test.csv"
    y_true = pd.read_csv(y_true_path, header=None).squeeze()  # Assuming one column
    mse = mean_squared_error(y_true, y_pred)
    logger.info(f"Mean Squared Error: {mse}")
    report_dict = {
        "regression_metrics": {
            "mean_squared_error": {
                "value": mse,
                "standard_deviation": "NaN",
            },
        },
    }
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    evaluation_path = os.path.join(output_dir, "evaluation_metrics.json")
    logger.info("Saving Evaluation Report")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))