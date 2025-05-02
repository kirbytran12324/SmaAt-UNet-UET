
import tifffile
import h5py
import os
import random
from datetime import datetime
import numpy as np

from scipy.interpolate import griddata



tiff_dir = './dataset'
hdf5_file = './dataset/normalized.h5'  



def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]


def replace_inf_with_finite_image(image):
    """Replace -inf values with the minimum finite value in the image."""
    # min_finite = np.min(image[np.isfinite(image)])
    # image[np.isneginf(image)] = min_finite
    # return image
    """Replace -inf values using interpolation from the nearest valid points in the image."""
    h, w = image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    #take valid points
    valid_mask = np.isfinite(image)
    valid_points = np.column_stack((X[valid_mask], Y[valid_mask]))
    valid_values = image[valid_mask]

    #take inf points
    inf_mask = np.isneginf(image)
    inf_points = np.column_stack((X[inf_mask], Y[inf_mask]))

    #interpolation
    interpolated_values = griddata(valid_points, valid_values, inf_points, method='nearest')


    filled_image = image.copy()
    filled_image[inf_mask] = interpolated_values

    return filled_image

def normalize_image(image):
    """Divide all finite values in the image by 260.0."""
    finite_mask = np.isfinite(image)
    image[finite_mask] /= 260.0
    return image


def load_tiff_files(directory):
    images = {}
    timestamps = []

    def search_for_tiff_files(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.tif'):
                    # Load the image
                    img = tifffile.imread(os.path.join(root, file))

                    # Replace -inf values
                    img = replace_inf_with_finite_image(img)

                    # Crop the image to 240x80
                    img = crop_center(img, 240, 80)

                    # Normalize finite values
                    img = normalize_image(img)

                    # Extract the timestamp from the filename
                    timestamp_str = file[6:-4]  # Adjust these indices to match your filenames
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                        timestamp = timestamp.timestamp()  # Correct way to get epoch time

                        # Store the image in the dictionary with the timestamp as the key
                        images[timestamp] = img
                        timestamps.append(timestamp)
                    except ValueError:
                        print(f"Skipping file {file} due to invalid timestamp")
                        continue

        for root, dirs, _ in os.walk(directory):
            for d in dirs:
                search_for_tiff_files(os.path.join(root, d))  # Recursively search in subdirectories

    search_for_tiff_files(directory)
    return list(images.values()), timestamps


def save_to_hdf5(group_name, images, timestamps):
    with h5py.File(hdf5_file, 'a') as hf:
        if group_name in hf:
            group = hf[group_name]
        else:
            group = hf.create_group(group_name)
        group.create_dataset('images', data=images)
        group.create_dataset('timestamps', data=timestamps)



def process_directory(directory):
    # Load TIFF images and timestamps
    images, timestamps = load_tiff_files(directory)

    # Initialize lists to hold training and testing data
    train_images, train_timestamps = [], []
    test_images, test_timestamps = [], []

    # Days missing hours
    exclude_days = {
        (2019, 10): {'04','05','06','07','08','09' '10', '15', '16', '17', '18', '19', '20', '21','23', '26', '27', '29', '30'},
        (2020, 4): {'23','24','25','26','27','28','29','30'},
        (2020, 10): {'02','20','23','26','27','28','29' },
    }

    #Choosing 5 days to test
    random.seed(42)
    test_days = {}
    for year, month, max_day in [(2019, 4, 30), (2019, 10, 31), (2020, 4, 30), (2020, 10, 31)]:
        all_days = {f"{i:02d}" for i in range(1, max_day + 1)}
        bad_days = exclude_days.get((year, month), set())
        valid_days = list(all_days - bad_days)
        chosen_days = random.sample(valid_days, 5)
        test_days[(year, month)] = set(chosen_days)

    # Iterate over images and timestamps
    for img, timestamp in zip(images, timestamps):
        timestamp_dt = datetime.fromtimestamp(timestamp)
        year, month, day = timestamp_dt.year, timestamp_dt.month, f"{timestamp_dt.day:02d}"

        # Split the data into training and testing sets
        if (year, month) in test_days and day in test_days[(year, month)]:
            test_images.append(img)
            test_timestamps.append(timestamp)
        else:
            train_images.append(img)
            train_timestamps.append(timestamp)

    # Save training and testing sets in HDF5 file
    save_to_hdf5('train', train_images, train_timestamps)
    save_to_hdf5('test', test_images, test_timestamps)

    print("Test days per month:")
    for k, v in test_days.items():
        print(f"{k}: {sorted(v)}")


if __name__ == '__main__':
    process_directory(tiff_dir)
    print('Data saved to HDF5 file.')

