import os
import tarfile

import requests


def download():
    this_file_path = os.path.split(__file__)[0]
    data_path = os.path.join(this_file_path, "data")
    tar_path = os.path.join(this_file_path, "data.tar")

    if os.path.isdir(os.path.join(data_path, "dev")) and os.path.isdir(
        os.path.join(data_path, "val")
    ):
        return

    if not os.path.exists(tar_path):
        url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
        print(f"Downloading {url}")
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            f.write(r.content)
        print(f"Saved to {tar_path}")

    tar = tarfile.open(tar_path)
    tar.extractall(this_file_path)
    tar.close()
    print(f"Saved to {data_path}")


if __name__ == "__main__":
    download()
