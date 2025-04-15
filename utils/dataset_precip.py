from torch.utils.data import Dataset
import h5py
import numpy as np


class precipitation_maps_h5(Dataset):
    """
    A PyTorch Dataset for handling precipitation maps stored in an HDF5 file. This class loads a sequence of images
    and returns the first `num_input_images` as input and the last image as target.
    """
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        """
        Initialize the dataset.

        Args:
            in_file (str): Path to the HDF5 file containing the data.
            num_input_images (int): Number of images to use as input.
            num_output_images (int): Number of images to use as output.
            train (bool, optional): Whether to load the training set. If False, the test set is loaded. Defaults to True.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        super().__init__()

        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, "r")["train" if train else "test"]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.sequence_length = num_input_images + num_output_images

        self.train = train
        self.size_dataset = self.n_images - (num_input_images + num_output_images)
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing the input images and the target image.
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024 ** 3)["train" if self.train else "test"]["images"]
        imgs = np.array(self.dataset[index: index + self.sequence_length], dtype="float32")

        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.size_dataset


class precipitation_maps_oversampled_h5(Dataset):
    """
    A PyTorch Dataset for handling oversampled precipitation maps stored in an HDF5 file.
    This class loads a sequence of images and returns the first `num_input_images` as input and the last image as target.
    """
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        """
        Initialize the dataset.

        Args:
            in_file (str): Path to the HDF5 file containing the data.
            num_input_images (int): Number of images to use as input.
            num_output_images (int): Number of images to use as output.
            train (bool, optional): Whether to load the training set. If False, the test set is loaded. Defaults to True.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        super().__init__()

        self.file_name = in_file
        self.samples, _, _, _ = h5py.File(self.file_name, "r")["train" if train else "test"]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images

        self.train = train
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing the input images and the target image.
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024 ** 3)["train" if self.train else "test"]["images"]
        imgs = np.array(self.dataset[index], dtype="float32")

        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        target_img = imgs[-1]

        return input_img, target_img

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.samples


class precipitation_maps_classification_h5(Dataset):
    """
    A PyTorch Dataset for handling classification of precipitation maps stored in an HDF5 file.
    This class loads a sequence of images and returns the first `num_input_images` as input and the last image as target.
    The target image is bucketed into categories based on precipitation levels.
    """
    def __init__(self, in_file, num_input_images, img_to_predict, train=True, transform=None):
        """
        Initialize the dataset.

        Args:
            in_file (str): Path to the HDF5 file containing the data.
            num_input_images (int): Number of images to use as input.
            img_to_predict (int): Index of the image to predict.
            train (bool, optional): Whether to load the training set. If False, the test set is loaded. Defaults to True.
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        super().__init__()

        self.file_name = in_file
        self.n_samples, self.n_images, self.nx, self.ny = h5py.File(self.file_name, "r")["train" if train else "test"]["images"].shape
        self.num_input = num_input_images
        self.img_to_predict = img_to_predict
        self.sequence_length = num_input_images + img_to_predict
        self.bins = np.array([0.0, 0.5, 1, 2, 5, 10, 30, 50, 100, 150, 200])

        self.train = train
        self.size_dataset = self.n_samples
        self.transform = transform
        self.dataset = None

    def __getitem__(self, index):
        """
        Get a sample from the dataset.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing the input images and the bucketed target image.
        """
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024 ** 3)["train" if self.train else "test"]["images"]
        imgs = np.array(self.dataset[index: index + self.sequence_length], dtype="float32")

        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        target_img = imgs[-1]
        buckets = np.digitize(target_img * 260.0, self.bins, right=True)

        return input_img, buckets

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.size_dataset
