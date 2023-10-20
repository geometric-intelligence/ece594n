import os
import warnings
import psutil
import torch
import pandas as pd
import numpy as np


class TimeWindowsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        partition="train",
        val_ratio=0.20,
        test_ratio=0.10,
        shuffle=False,
        random_seed=0,
        normalize=False,
        pin_memory=False,
        autoencoder=False,
    ):

        # parameters initialization and checks
        self.data_dir = data_dir
        self.partition = partition
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.normalize = normalize
        self.pin_memory = pin_memory
        self.autoencoder = autoencoder
        if not os.path.exists(self.data_dir):
            raise ValueError("{} does not exists!".format(self.data_dir))
        if (self.test_ratio + self.val_ratio) >= 1.0:
            raise ValueError(
                "Test and validation ratio are greater than one: {:.2f} > 1.0 !".format(
                    self.test_ratio + self.val_ratio
                )
            )
        valid_partition_names = ["train", "valid", "test"]
        if self.partition not in valid_partition_names:
            raise ValueError(
                "Invalid partition name '{}', available partition names are {}.".format(
                    self.partition, valid_partition_names
                )
            )

        # read file paths
        self._data_filepaths, self._label_filepath = self._read_file_list()
        # define indexes for the current partition
        self._partition_indexes = self._set_indexes_partition()
        self._partition_filepaths = self._data_filepaths[self._partition_indexes]
        # read partition data, filepaths or data directly
        if self.pin_memory:
            self.partition_data = [
                np.load(data_filepath) for data_filepath in self._partition_filepaths
            ]
            # check RAM usage
            avail_ram = psutil.virtual_memory().available
            predicted_ram = (
                len(self.partition_data)
                * self.partition_data[0].size
                * self.partition_data[0].itemsize
            )
            if (predicted_ram / avail_ram) > 0.2:
                warnings.warn(
                    "Data uses more than 20% of available RAM ({:.1f} MB), consider using `pin_memory=False`.".format(
                        predicted_ram / 1e6
                    )
                )
        else:
            self.partition_data = self._partition_filepaths
        # read partition targets
        if (self._label_filepath is None) | self.autoencoder:
            if self.autoencoder == False:
                warnings.warn("No labels file, assuming auto-encoder generator.")
            self.partition_targets = None
        else:
            self.partition_targets = self._read_labels()[self._partition_indexes]

    def __repr__(self):
        return "{}*({}, {})".format(
            self.__len__(), self.__getitem__(0)[0].shape, self.__getitem__(0)[1].shape
        )

    def __len__(self):
        """Return the length of the current generator."""
        return len(self._partition_filepaths)

    def __getitem__(self, idx):
        """Generate one generator item (data and targets)."""
        # reading numpy
        if not self.pin_memory:
            np_data = np.load(self.partition_data[idx])
        else:
            np_data = self.partition_data[idx]
        # normalization
        if self.normalize:
            np_data = self._normalize_data(np_data)
        # auto-encoder generator
        if self.partition_targets is None:
            #       outputs = (torch.from_numpy(np_data,dtype='float32'), torch.from_numpy(np_data,dtype='float32'))
            outputs = (torch.from_numpy(np_data), torch.from_numpy(np_data))
        else:
            #       outputs = (torch.from_numpy(np_data,dtype='float32'), self.partition_targets[idx])
            outputs = (torch.from_numpy(np_data), self.partition_targets[idx])

        return outputs[0], outputs[1]

    def get_item_path(self, idx):
        return self._partition_filepaths[idx]

    def _set_indexes_partition(self):
        """Partition indexes into train/valid/test data"""
        n_samples = len(self._data_filepaths)
        train_index = 1 - self.test_ratio - self.val_ratio
        val_index = 1 - self.test_ratio
        indexes = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(indexes)

        if self.partition == "train":
            range_idx = (0, int(train_index * n_samples))
        elif self.partition == "valid":
            range_idx = (int(train_index * n_samples), int(val_index * n_samples))
        elif self.partition == "test":
            range_idx = (int(val_index * n_samples), n_samples)

        return indexes[range_idx[0] : range_idx[1]]

    def _normalize_data(self, data):
        """Gaussian-normalization of the data, helps the training process for neural network models."""
        return (data - np.mean(data)) / np.std(data)

    def _read_file_list(self):
        """Return the list of data files and labels if exists."""
        list_files = []
        data_files = []
        label_file = None

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                list_files += [os.path.join(root, file)]
        list_files = sorted(list_files)

        for f in list_files:
            if f.split(".")[-1] == "npy":
                data_files += [f]
            elif "labels.csv" in f:
                label_file = f

        return np.array(data_files), label_file

    def _read_labels(self):
        """Read the labels, sorted by the data files."""
        labels = pd.read_csv(self._label_filepath)
        labels = labels.sort_values(by=["filename"])

        return np.array(labels["label"])


# if __name__ == "__main__":
#   data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data_shima", "interim")
#   random_seed = 0

#   # Pytorch generator test
#   torch.manual_seed(random_seed)
#   train_dataset = TimeWindowsDataset(data_dir=data_dir, partition="train", random_seed=random_seed, pin_memory=True)
#   train_gen = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
#   train_features, train_labels = next(iter(train_gen))
#   print(f"Feature batch shape: {train_features.size()}; mean {torch.mean(train_features)}")
#   print(f"Labels batch shape: {train_labels.size()}; mean {torch.mean(torch.Tensor.float(train_labels))}")
#   # check train, valid and test generator
#   valid_dataset = TimeWindowsDataset(data_dir=data_dir, partition="valid", pin_memory=True)
#   test_dataset = TimeWindowsDataset(data_dir=data_dir, partition="test", pin_memory=True)
#   print("Train generator object: {}".format(train_dataset))
#   for ii, data in enumerate(train_dataset):
#     print("\r\t#{} - ({}, {})".format(ii, data[0].shape, data[1].shape), end='')
#   print("")
#   print("Valid generator object: {}".format(valid_dataset))
#   for ii, data in enumerate(valid_dataset):
#     print("\r\t#{} - ({}, {})".format(ii, data[0].shape, data[1].shape), end='')
#   print("")
#   print("Test generator object: {}".format(test_dataset))
#   for ii, data in enumerate(test_dataset):
#     print("\r\t#{} - ({}, {})".format(ii, data[0].shape, data[1].shape), end='')
#   print("")
#   # test auto-encoder generator
#   data_gen = TimeWindowsDataset(data_dir=data_dir, partition="train", pin_memory=True, autoencoder=True)
#   print("Auto-encoder generator object: {}".format(data_gen))
#   for ii, data in enumerate(data_gen):
#     print("\r\t#{} - ({}, {})".format(ii, data[0].shape, data[1].shape), end='')
#   print("")
#   # benchmark time gain with pin_memory
#   import time
#   start = time.time()
#   for data in data_gen:
#     continue
#   print("Memory not-pinned elapsed time: {}s".format(time.time() - start))
#   data_gen = TimeWindowsDataset(data_dir=data_dir, partition="train", pin_memory=True)
#   start = time.time()
#   for data in data_gen:
#     continue
#   print("Memory pinned elapsed time: {}s".format(time.time() - start))