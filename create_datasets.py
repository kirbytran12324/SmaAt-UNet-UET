import h5py
import numpy as np
from tqdm import tqdm

from root import ROOT_DIR


def create_dataset(input_length: int, image_ahead: int, rain_amount_thresh: float):
    """Create a dataset that has target images containing at least `rain_amount_thresh` (percent) of rain."""

    precipitation_folder = ROOT_DIR / "Radar" / "dataset"
    with h5py.File(
            precipitation_folder / "normalized.h5",
            "r",
    ) as orig_f:
        train_images = orig_f["train"]["images"]
        train_timestamps = orig_f["train"]["timestamps"]
        test_images = orig_f["test"]["images"]
        test_timestamps = orig_f["test"]["timestamps"]
        print("Train shape", train_images.shape)
        print("Test shape", test_images.shape)
        imgHeight = train_images.shape[1]
        imgWidth = train_images.shape[2]
        num_pixels = imgHeight * imgWidth

        filename = (
                precipitation_folder / f"train_test_input-length_{int(input_length)}_image-ahead_{int(image_ahead)}_rain-threshold_{int(rain_amount_thresh * 100)}.h5"
        )

        with h5py.File(filename, "w") as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            train_image_dataset = train_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgHeight, imgWidth),
                maxshape=(None, input_length + image_ahead, imgHeight, imgWidth),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            train_timestamp_dataset = train_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype="int32",
                compression="gzip",
                compression_opts=9,
            )
            test_image_dataset = test_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgHeight, imgWidth),
                maxshape=(None, input_length + image_ahead, imgHeight, imgWidth),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            test_timestamp_dataset = test_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype="int32",
                compression="gzip",
                compression_opts=9,
            )

            origin = [[train_images, train_timestamps], [test_images, test_timestamps]]
            datasets = [
                [train_image_dataset, train_timestamp_dataset],
                [test_image_dataset, test_timestamp_dataset],
            ]
            for origin_id, (images, timestamps) in enumerate(origin):
                image_dataset, timestamp_dataset = datasets[origin_id]
                first = True
                for i in tqdm(range(input_length + image_ahead, len(images))):
                    # If threshold of rain is bigger in the target image: add sequence to dataset
                    if np.sum(images[i] > 0) >= num_pixels * rain_amount_thresh:
                        imgs = images[i - (input_length + image_ahead): i]
                        timestamps_img = timestamps[i - (input_length + image_ahead): i]
                        #                     print(imgs.shape)
                        #                     print(timestamps_img.shape)
                        # extend the dataset by 1 and add the entry
                        if first:
                            first = False
                        else:
                            image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                            timestamp_dataset.resize(timestamp_dataset.shape[0] + 1, axis=0)
                        image_dataset[-1] = imgs
                        timestamp_dataset[-1] = timestamps_img.reshape(-1, 1)


if __name__ == "__main__":
    rain_amount_thresh = {0.0, 0.05, 0.1}
    for rain_thresh in rain_amount_thresh:
        create_dataset(6, 1, rain_thresh)
        print(f"Dataset with rain threshold {rain_thresh} created.")
