# CNN Interactive

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cnn-interactive-mguujakeyghanzjj4ptz8j.streamlit.app/)

An educational, interactive Convolutional Neural Network (CNN) built entirely from scratch using **NumPy** and visualized with **Streamlit**.

## Features

-   **Pure NumPy Implementation**: No TensorFlow or PyTorch. Learn how CNNs work under the hood.
-   **Interactive Visualization**:
    -   Draw digits on a canvas and see predictions in real-time.
    -   Visualize internal **Feature Maps** and **Filters**.
-   **Real-time Training**: Watch the loss and accuracy evolve as you train the model in the browser.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Amruth2105/CNN-interactive.git
    cd CNN-interactive
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## Structure

-   `cnn_lib.py`: Core CNN library (Conv3x3, MaxPool2, Softmax, ReLU).
-   `app.py`: Streamlit application interface.
-   `mnist_loader.py`: Custom MNIST data loader.
