"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

data_loader.py
- Load and preprocess MNIST data (http://yann.lecun.com/exdb/mnist/)
"""

# Necessary packages
import numpy as np
import pandas as pd
from keras.datasets import mnist
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(label_no):
    """MNIST data loading.

    Args:
      - label_data_rate: ratio of labeled data

    Returns:
      - x_label, y_label: labeled dataset
      - x_unlab: unlabeled dataset
      - x_test, y_test: test dataset
    """
    # # Import mnist data
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # # One hot encoding for the labels
    # y_train = np.asarray(pd.get_dummies(y_train))
    # y_test = np.asarray(pd.get_dummies(y_test))

    # # Normalize features
    # x_train = x_train / 255.0
    # x_test = x_test / 255.0

    # # Treat MNIST data as tabular data with 784 features
    # # Shape
    # no, dim_x, dim_y = np.shape(x_train)
    # test_no, _, _ = np.shape(x_test)

    # x_train = np.reshape(x_train, [no, dim_x * dim_y])
    # x_test = np.reshape(x_test, [test_no, dim_x * dim_y])

    # # Divide labeled and unlabeled data
    # idx = np.random.permutation(len(y_train))

    # # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    # label_idx = idx[:int(len(idx)*label_data_rate)]
    # unlab_idx = idx[int(len(idx)*label_data_rate):]

    # # Unlabeled data
    # x_unlab = x_train[unlab_idx, :]

    # # Labeled data
    # x_label = x_train[label_idx, :]
    # y_label = y_train[label_idx, :]

    df = pd.read_csv("./Levine_32dim.fcs.csv")

    # remove the event, time and individual columns
    df = df.drop(columns=['Event', 'Time', 'individual',
                          'file_number', 'event_number'])
    y_pd = df['label']
    df = df.drop(columns=['label'])

    # last column contains class labels
    X = df.values
    y = y_pd.values

    # get the X_labeled and X_unlabeled
    X_labeled = X[:104184]
    x_unlab = X[104185:]

    # get the y_labeled
    Y_labeled = y[:104184]

    # one hot encode Y_labeled

    x_train, x_test, y_train, y_test = train_test_split(X_labeled,
                                                        Y_labeled, test_size=0.3,
                                                        random_state=42)

    # get the number of samples in x_train
    num_total_samples = x_train.shape[0]

    # generate random indices for the samples to pick
    random_indices = np.random.choice(
        num_total_samples, size=label_no, replace=False)

    y_train = np.asarray(pd.get_dummies(y_train))
    y_test = np.asarray(pd.get_dummies(y_test))

    # select the corresponding samples from x_train and y_train
    selected_x_train = x_train[random_indices]
    selected_y_train = y_train[random_indices]

    return selected_x_train, selected_y_train, x_unlab, x_test, y_test
