<<<<<<< HEAD

import seaborn as sns; sns.set(style='whitegrid')

import os
import numpy as np

def resize_array(array, target_shape):
    current_shape = array.shape

    # Create an empty array with the target shape
    resized_array = np.zeros(target_shape, dtype=array.dtype)

    # Determine the dimensions to crop
    crop_dims = [min(target_shape[i], current_shape[i]) for i in range(array.ndim)]

    # Crop the original array
    cropped_array = array[tuple(slice(0, dim) for dim in crop_dims)]

    # Assign the cropped array to the resized array
    resized_array[tuple(slice(0, dim) for dim in crop_dims)] = cropped_array

    return resized_array

# Define the directory containing the numpy arrays
directory = "./testing_image_CDH1"

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        filepath = os.path.join(directory, filename)

        # Load the numpy array
        array = np.load(filepath)

        print(array.shape)

        # Resize the array to the target shape
        resized_array = resize_array(array, (32, 128, 128))

        # Delete the original file
        os.remove(filepath)

        # Save the resized array as a new numpy file with the same filename
        np.save(filepath, resized_array)

=======

import seaborn as sns; sns.set(style='whitegrid')

import os
import numpy as np

def resize_array(array, target_shape):
    current_shape = array.shape

    # Create an empty array with the target shape
    resized_array = np.zeros(target_shape, dtype=array.dtype)

    # Determine the dimensions to crop
    crop_dims = [min(target_shape[i], current_shape[i]) for i in range(array.ndim)]

    # Crop the original array
    cropped_array = array[tuple(slice(0, dim) for dim in crop_dims)]

    # Assign the cropped array to the resized array
    resized_array[tuple(slice(0, dim) for dim in crop_dims)] = cropped_array

    return resized_array

# Define the directory containing the numpy arrays
directory = "./testing_image_CDH1"

# Iterate through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        filepath = os.path.join(directory, filename)

        # Load the numpy array
        array = np.load(filepath)

        print(array.shape)

        # Resize the array to the target shape
        resized_array = resize_array(array, (32, 128, 128))

        # Delete the original file
        os.remove(filepath)

        # Save the resized array as a new numpy file with the same filename
        np.save(filepath, resized_array)

>>>>>>> 959c822f3e294e28bffabe791f7b4ec2e6720746
