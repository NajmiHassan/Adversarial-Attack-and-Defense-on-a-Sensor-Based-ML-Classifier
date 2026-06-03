import os
import numpy as np

# Standard label names for activities 1..6
DEFAULT_CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def _read_class_names(data_dir):
    """Read activity_labels.txt if present, else fall back to defaults."""
    path = os.path.join(data_dir, "activity_labels.txt")
    if not os.path.exists(path):
        return DEFAULT_CLASS_NAMES
    names = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                names.append(parts[1])
    return names if names else DEFAULT_CLASS_NAMES


def load_har(data_dir):
    """
    Load UCI HAR train/test splits.

    Returns:
        X_train (float32 [N, 561]), y_train (int64 [N]),
        X_test  (float32 [M, 561]), y_test  (int64 [M]),
        class_names (list[str])

    Labels are converted from the original 1..6 to 0..5.
    """
    train_X = os.path.join(data_dir, "train", "X_train.txt")
    train_y = os.path.join(data_dir, "train", "y_train.txt")
    test_X = os.path.join(data_dir, "test", "X_test.txt")
    test_y = os.path.join(data_dir, "test", "y_test.txt")

    for p in (train_X, train_y, test_X, test_y):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find '{p}'.\n"
                "Make sure --data_dir points to the unzipped 'UCI HAR Dataset' "
                "folder. See the download instructions at the top of data.py."
            )

    X_train = np.loadtxt(train_X, dtype=np.float32)
    y_train = np.loadtxt(train_y, dtype=np.int64) - 1  # 1..6 -> 0..5
    X_test = np.loadtxt(test_X, dtype=np.float32)
    y_test = np.loadtxt(test_y, dtype=np.int64) - 1

    class_names = _read_class_names(data_dir)
    return X_train, y_train, X_test, y_test, class_names
