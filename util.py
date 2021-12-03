import os
import zipfile


def get_data():
    # download dataset
    os.system(
        "kaggle competitions download -c house-prices-advanced-regression-techniques"
    )

    # extract files
    with zipfile.ZipFile(
        "house-prices-advanced-regression-techniques.zip", "r"
    ) as zip_ref:
        zip_ref.extractall("data")

    # remove zip file
    os.remove("house-prices-advanced-regression-techniques.zip")
