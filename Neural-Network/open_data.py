# Loads images and labels into numpy arrays for coordination with neuralnet.py
import numpy as np

def _get_labels(label_file_name, max_num):
    ''' Load image labels '''
    with open(label_file_name, 'rb') as label_file:

        # First 4 bytes are the "magic number", ignore.
        label_file.read(4)

        # Get number of desired labels
        num_labels = int.from_bytes(label_file.read(4), byteorder='big')
        num_labels = min(num_labels, max_num)

        # Initialize array
        labels = np.empty(num_labels)

        # Read through file converting each byte to integers
        for k in range(num_labels):
            num = label_file.read(1) 
            labels[k] = int.from_bytes(num, byteorder='big')
    return labels

def _get_images(image_file_name, max_num):
    ''' Load image vectors '''
    with open(image_file_name, 'rb') as image_file:

        # First 4 bytes are the "magic number", ignore.
        image_file.read(4)

        # Get number of desired images
        num_images = int.from_bytes(image_file.read(4), byteorder='big')
        num_images = min(num_images, max_num)

        # Could probably be hardcoded in as 784=28*28
        num_rows = int.from_bytes(image_file.read(4), byteorder='big')
        num_cols = int.from_bytes(image_file.read(4), byteorder='big')
        num_pixels = num_cols * num_rows

        # Should be the most efficient initialization
        images = np.empty((num_pixels, num_images), dtype=float)
        image = np.empty(num_pixels)
        
        for k in range(num_images):
            for i in range(num_pixels):
                
                #Convert byte to integer
                num = image_file.read(1) 
                image[i] = int.from_bytes(num, byteorder='big')

            # Cheaper to only make num_images writes to images,
            # as opposed to 784*num_images, since images is large
            images[:,k] = image
    return images

# Location of data files
training_label_file = 'mnist_data/train-labels.idx1-ubyte'
training_image_file = 'mnist_data/train-images.idx3-ubyte'
test_label_file = 'mnist_data/t10k-labels.idx1-ubyte'
test_image_file = 'mnist_data/t10k-images.idx3-ubyte'

def get_data(max_num, training=False, test=False, verbose=False):
    '''
    Retrieve data from the MNIST files
    max_num - retrieve only the first elements of each file
    training/test - flag which data sets will be returned
    verbose - print to console as each file finishes loading
    '''

    # If not requested, just return None
    training_labels, training_images = None, None
    test_labels, test_images = None, None

    if training:
        training_labels = _get_labels(training_label_file, max_num)
        if verbose: print('training labels loaded.')
        training_images = _get_images(training_image_file, max_num)
        if verbose: print('training images loaded.')
    if test:
        test_labels = _get_labels(test_label_file, max_num)
        if verbose: print('test labels loaded.')
        test_images = _get_images(test_image_file, max_num)
        if verbose: print('test images loaded.')

    return training_labels, training_images, test_labels, test_images
