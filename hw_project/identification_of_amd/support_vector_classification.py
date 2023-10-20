"""Classification of healthy and unhealthy retinal scans using SVC."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def classify_images(images_normal, images_amd, images_mh, kernel="linear"):
    """Classify retinal images directly using support vector classifier.

    Parameters
    ----------
    images_normal : list of numpy arrays
        Retinal images labeled as normal.
    images_amd : list of numpy arrays
        Retinal images labeled as positive for AMD.
    images_mh : list of numpy arrays
        Retinal images labeled as positive for MH.
    kernel : string
        Type of kernel used to create SVC object.

    Returns
    -------
    X_test : list of numpy arrays
        Test data for support vector classifier.
    y_test : list of numpy arrays
        Test labels for ground truth comparison.
    y_pred : numpy array
        Targets predicted by support vector classifier.
    accuracy : float
        Accuracy score predicted targets against given labels.
    """
    # Load the data into a combined list with appropriate labels
    data = images_normal + images_amd + images_mh
    labels = [0] * len(images_normal)
    labels += [1] * len(images_amd)
    labels += [2] * len(images_mh)

    # Resize the images to 64-by-64
    img_size = 64
    for i in range(len(data)):
        data[i] = cv2.resize(data[i], (img_size, img_size))

    # Convert data and label lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=3
    )

    # Create a Support Vector Classifier object with probability estimates enabled
    svc = SVC(kernel=kernel, probability=True)

    # Train the model using 80% of the data
    svc.fit(X_train.reshape((len(X_train), -1)), y_train)

    # Predict target labels on the remaining test data
    y_pred = svc.predict(X_test.reshape((len(X_test), -1)))

    # Calculate the accuracy score of the model
    accuracy = accuracy_score(y_test, y_pred)

    return X_test, y_test, y_pred, accuracy


def classify_points(x_y_pairs_normal, x_y_pairs_amd, x_y_pairs_mh, kernel="linear"):
    """Classify IS/OS shape characteristics using support vector classifier.

    Parameters
    ----------
    x_y_pairs_normal : list of tuples
        diff_x and diff_y metrics for normal, healthy retinas.
    x_y_pairs_amd : list of tuples
        diff_x and diff_y metrics for retinas with AMD.
    x_y_pairs_mh : list of tuples
        diff_x and diff_y metrics for retinas with macular hole.
    kernel : string
        Type of kernel used to create SVC object.

    Returns
    -------
    X_test : list of numpy arrays
        Test data for support vector classifier.
    y_test : list of numpy arrays
        Test labels for ground truth comparison.
    y_pred : numpy array
        Targets predicted by support vector classifier.
    accuracy : float
        Accuracy score predicted targets against given labels.
    """
    # Load the data into a combined list with appropriate labels
    data = x_y_pairs_normal + x_y_pairs_amd + x_y_pairs_mh
    labels = [0] * len(x_y_pairs_normal)
    labels += [1] * len(x_y_pairs_amd)
    labels += [2] * len(x_y_pairs_mh)

    # Convert data and label lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=3
    )

    # Create a Support Vector Classifier object with probability estimates enabled
    svc = SVC(kernel=kernel, probability=True)

    # Train the model using 80% of the data
    svc.fit(X_train.reshape((len(X_train), -1)), y_train)

    # Predict target labels on the remaining test data
    y_pred = svc.predict(X_test.reshape((len(X_test), -1)))

    # Calculate the accuracy score of the model
    accuracy = accuracy_score(y_test, y_pred)

    return X_test, y_test, y_pred, accuracy


def show_confusion_matrix(y_test, y_pred):
    """Create and display a confusion matrix for given labels and targets.

    Parameters
    ----------
    y_test : numpy array
        Ground truth labels for given classes.
    y_pred : numpy array
        Predicted targets from SVC or other classifier.

    Returns
    -------
    None
    """
    cm = confusion_matrix(y_test, y_pred)
    categories = ["Normal", "AMD", "MH"]
    df_cm = pd.DataFrame(cm, index=categories, columns=categories)
    plt.figure(figsize=(6, 4))
    plt.title("Confusion Matrix")
    sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="g")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()

    return None
