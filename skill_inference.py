""" Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. """
""" SPDX-License-Identifier: MIT-0 """

import argparse
import sagemaker

from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-name", type=str)
    args, _ = parser.parse_known_args()

    print(F"Using SageMaker Endpoint: {args.endpoint_name}")
    predictor = Predictor(
        endpoint_name=args.endpoint_name,
        sagemaker_session=sagemaker.Session(),
        serializer=CSVSerializer()
    )

    print("Sending inference request with test payload ...")
    response = predictor.predict(
        "10597,20312,602,205,1916,56266,76578,9725,8.0"
    ).decode("utf-8")
    print(f"SageMaker returned the following response: {response}")
    print(type(response))
