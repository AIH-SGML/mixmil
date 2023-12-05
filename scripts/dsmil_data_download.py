"""
Script adapted from
https://github.com/binli123/dsmil-wsi/blob/master/download.py

Script and data provided by the DSMIL authors:

@inproceedings{li2021dual,
  title={Dual-stream multiple instance learning network for whole slide image classification with self-supervised contrastive learning},
  author={Li, Bin and Li, Yin and Eliceiri, Kevin W},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14318--14328},
  year={2021}
}
"""

import argparse
import os
import urllib.request
import zipfile

from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def unzip_data(zip_path, data_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_path)


DATASET_URLS = {
    "camelyon16": "https://uwmadison.box.com/shared/static/l9ou15iwup73ivdjq0bc61wcg5ae8dwe.zip",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="camelyon16", help="Dataset to be downloaded: camelyon16")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip file")
    args = parser.parse_args()

    assert args.dataset in DATASET_URLS, f"Dataset {args.dataset} not found"

    print(f"downloading dataset: {args.dataset}")
    unzip_dir = f"data/{args.dataset}"
    zip_file_path = f"data/{args.dataset}-dataset.zip"
    os.makedirs(unzip_dir, exist_ok=True)
    download_url(DATASET_URLS[args.dataset], zip_file_path)
    unzip_data(zip_file_path, unzip_dir)

    if not args.keep_zip:
        os.remove(f"{args.dataset}-dataset.zip")

    print("done!")


if __name__ == "__main__":
    main()
