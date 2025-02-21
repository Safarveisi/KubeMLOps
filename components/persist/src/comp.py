import os
import io
import boto3
import pandas as pd
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        help="Path to the csv file",
        type=str,
        required=True,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--key",
        help="Prefix in S3 in which the csv is persisted",
        type=str,
        required=True,
        default=argparse.SUPPRESS,
    )

    print(f"The bucket name is {os.environ['bucket']}")

    args = parser.parse_args()

    df = pd.read_csv(args.input_path, header=0)

    s3 = boto3.resource(
        service_name="s3",
        region_name="eu-central-1",
        aws_access_key_id=os.environ["access_key"],
        aws_secret_access_key=os.environ["secret_key"],
        endpoint_url="https://" + os.environ["endpoint_url"],
    )

    # Write the df into the S3 bucket
    with io.StringIO() as csv_buffer:
        df.to_csv(csv_buffer, header=True, index=False)
        csv_buffer.seek(0)
        s3.Object(os.environ["bucket"], args.key).put(Body=csv_buffer.getvalue())

    print("The shape of the written df is: {}".format(df.shape))


if __name__ == "__main__":
    main()
