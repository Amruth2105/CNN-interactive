
import urllib.request
import gzip
import numpy as np
import os

def load_mnist():
    """
    Downloads and parses MNIST dataset.
    Returns: (train_images, train_labels), (test_images, test_labels)
    """
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz"
    }

    def download_and_parse(filename, label=False):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filename)
        
        with gzip.open(filename, 'rb') as f:
            if label:
                # Magic number (4 bytes) + Item count (4 bytes)
                _ = f.read(8)
                data = np.frombuffer(f.read(), dtype=np.uint8)
            else:
                # Magic (4) + Count (4) + Rows (4) + Cols (4)
                _ = f.read(16)
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        return data

    print("Loading fashion-mnist/mnist data...")
    train_images = download_and_parse(files["train_img"])
    train_labels = download_and_parse(files["train_lbl"], label=True)
    test_images = download_and_parse(files["test_img"])
    test_labels = download_and_parse(files["test_lbl"], label=True)

    return (train_images, train_labels), (test_images, test_labels)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"Loaded: Train {x_train.shape}, Test {x_test.shape}")
