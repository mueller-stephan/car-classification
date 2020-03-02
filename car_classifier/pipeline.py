import tensorflow as tf
import numpy as np
# from keras.applications import imagenet_utils
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import image as prep_image


FRACTION = 0.3

def get_label(filename, label_type='make'):
    """
    Function to get label from file_name

    Args:
        filename: file name to extract label from
        label_type: which type of label is needed (either 'make' or 'model')

    Returns:
        tf.tensor with label

    Raises:
        ValueError: If illegal value argument
    """
    parts = tf.strings.split(filename, '/')
    name = parts[-1]
    label = tf.strings.split(name, '_')

    if label_type == 'make':
        return tf.strings.lower(label[0])
    elif label_type == 'model':
        return tf.strings.lower(label[0] + '_' + label[1])
    else:
        raise ValueError('label must be either "make" or "model" and not ', label_type)


def get_image(filename, size=(212, 320)):
    """
    Function to lead image as tensor and resize it to specific size

    Args:
        filename: file name (path)
        size: size of image after resizing

    Returns:
        Image as tf.Tensor
    """
    # Prints only executed at initialization time -> no good for debugging
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    # print('Decoded image: \n' + 'minimum is: ' + str(tf.reduce_min(image)) + 'and maximum is: ' + str(tf.reduce_max(image)))
    image = tf.image.convert_image_dtype(image, tf.float32)
    # print('Converted to float: \n' + 'minimum is: ' + str(tf.reduce_min(image)) + 'and maximum is: ' + str(tf.reduce_max(image)))
    image = tf.image.resize_with_crop_or_pad(image, target_height=size[0], target_width=size[1])
    # print('resized image: \n' + 'minimum is: ' + str(tf.reduce_min(image)) + 'and maximum is: ' + str(tf.reduce_max(image)))
    # image = preprocess_input(image)
    # image = image * 255.0
    # print('Returned image: \n' + 'minimum is: ' + str(tf.reduce_min(image)) + 'and maximum is: ' + str(tf.reduce_max(image)))
    return image


def parse_file(filename, classes, input_size):
    """
    Function to parse files; loading images from file path and creating (one hot encoded) labels

    Args:
        filename: filename (including full path)
        classes: list of all classes for encoding
        input_size: size of images (output size)

    Returns:
        tuple of image and (one hot encoded) label
    """
    label = get_label(filename)
    image = get_image(filename, size=input_size)

    target = one_hot_encode(classes=classes, label=label)

    return image, target


def one_hot_encode(classes, label):
    """
    Function for one hot encoding of label

    Args:
        classes: list of all classes for encoding
        label: label to be encoded

    Returns:
        Encoded label
    """
    n_classes = len(classes)
    names = tf.constant(classes, shape=[1, n_classes])
    index = tf.argmax(tf.cast(tf.equal(names, label), tf.int32), axis=1)
    y = tf.one_hot(index, n_classes)
    y = tf.squeeze(y)
    return y


def image_augment(image, label):
    """
    Function for image augmentation

    Args:
        image: Images from tf.data.Dataset
        label: Labels from tf.data.Dataset

    Returns:
        Tuple of image, label
    """

    # Flip image with probability .5
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=0.05 / 255.0)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def construct_ds(input_files,
                 batch_size,
                 classes,
                 input_size=(212, 320),
                 prefetch_size=10,
                 shuffle_size=32,
                 augment=False):
    """
    Function to construct a tf.data set from files

    Args:
        input_files: list of files
        batch_size: number of observations in batch
        classes: list with all class labels
        input_size: size of images (output size)
        prefetch_size: buffer size (number of batches to prefetch)
        shuffle_size: shuffle size (size of buffer to shuffle from)
        augment: boolean if image augmentation is needed

    Returns:
        buffered and prefetched tf.data object with (image, label) tuple
    """
    # Create tf.data.Dataset from list of files
    ds = tf.data.Dataset.from_tensor_slices(input_files)

    # Shuffle files
    ds = ds.shuffle(buffer_size=shuffle_size)

    # Load image/labels
    ds = ds.map(lambda x: parse_file(x, classes=classes, input_size=input_size))

    # Image augmentation
    random_nr = np.random.rand()

    if augment and (random_nr < FRACTION):
        print("Doing augmentation...")
        ds = ds.map(image_augment)

    # Batch and prefetch data
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=prefetch_size)

    return ds





