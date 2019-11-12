import numpy as np
import matplotlib.pyplot as plt
import csv
import skimage.transform as trans
from skimage.util import random_noise
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import time


def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes

    for c in range(0, 43):
        # subdirectory for class
        prefix = rootpath + '/' + format(c, '05d') + '/'
        gtFile = open(prefix + 'GT-' + format(c, '05d') +
                      '.csv')  # annotations file
        # csv parser for annotations file
        gtReader = csv.reader(gtFile, delimiter=';')
        next(gtReader)  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            # the 1th column is the filename
            images.append(plt.imread(prefix + row[0]))
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()

    return images, labels


def readTestData(csv_path='GTSRB/Final_Test/Images/', csv_name='GT-final_test.csv'):
    '''Reads traffic sign test data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = []
    labels = []

    # Open csv reader
    gtReader = csv.reader(open(csv_path+csv_name), delimiter=';')
    next(gtReader)

    for row in gtReader:
        # Store the image and its label
        im_name, label = row[0], row[-1]
        images.append(plt.imread(csv_path + im_name))
        labels.append(label)

    return images, labels


def pad_images(images):
    '''
    Add padding to the images using numpy.pad method

    Arguments: images - array of images
    Return: None
    '''
    for i in range(len(images)):
        # Get the image and its height and width
        im, height, width = images[i], len(images[i]), len(images[i][0])

        if height == width:
            continue

        # If height and width differ from one another, pad the sides
        if height < width:
            images[i] = np.pad(
                im, [(0, width - height), (0, 0), (0, 0)], mode='constant')
        else:
            images[i] = np.pad(
                im, [(0, 0), (0, height - width), (0, 0)], mode='constant')


def scale_images(images, size=30):
    '''
    Scale the images to the `size` parameter

    Arguments: images - array of images, size - to which scale images
    Return: None
    '''
    for i in range(len(images)):
        images[i] = trans.resize(images[i], (size, size))


def split_data(images, labels, split_rate=0.8, image_per_class=30):
    '''
    Split images array into two arrays

    Arguments: images - array
               labels - array
               split_rate - float, how much divide data
               image_per_class - int, how much images in same class
    Return: train - numpy array of train data, test  - numpy array of test data 
    '''
    # Generate array of indexes of image classes and shuffle them
    arr = np.arange(int(len(images) / image_per_class))
    np.random.shuffle(arr)

    train_data, test_data = list(), list()
    train_labels, test_labels = list(), list()

    # Elements from `0:split_by` will be in train dataset,
    split_by = int(len(arr) * split_rate)
    for i in arr[:split_by]:
        train_data.extend(images[i*30: i*30 + 30])
        train_labels.extend(labels[i*30: i*30 + 30])

    # Elements from `split_by:end` will be in test dataset
    for i in arr[split_by:]:
        test_data.extend(images[i*30: i*30 + 30])
        test_labels.extend(labels[i*30: i*30 + 30])

    return np.array(train_data), np.array(test_data), np.array(train_labels), np.array(test_labels)


def augmentation(images, labels):
    '''
    Add images to all classes, thus making them equal in size

    Arguments:  images - array
                labels - array
    Return: new arrays of images and labels
    '''
    # Get unique labels in dataset and their  counts
    unique_labels, counts = np.unique(
        labels, return_counts=True)

    extra_images, extra_labels = list(), list()
    target = max(counts)
    for label in unique_labels:
        # Get all images with such label
        elems = images[np.where(labels == label)]
        # Randomly choice some of them
        rand_elems = elems[np.random.choice(
            elems.shape[0], target - len(elems))]
        for el in rand_elems:
            # Rotation and noise adding
            new_image = trans.rotate(el, np.random.uniform(-20, 20))
            new_image = random_noise(new_image)

            extra_images.append(new_image)
            extra_labels.append(label)

    return np.append(images, extra_images, axis=0), np.append(labels, extra_labels, axis=0)


def shuffle(images, labels):
    '''
    Shuffle images and labels in the same way

    Arguments: images - array
               labels - array
    Return, shuffled arrays
    '''
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    return images[indices], labels[indices]


def reshape(images):
    '''
    Create matrix of vectors

    Arguments: images - array
    Return: array
    '''
    res = list()
    for im in images:
        res.append(im.reshape(-1))

    return np.array(res)


def train_model_size(images, labels, test_images, test_labels, size):
    start = time.gmtime()
    scale_images(non_augmented_data, size=size)
    forest = RandomForestClassifier(
        n_estimators=50, max_depth=30, n_jobs=4, random_state=0)
    forest.fit(images, labels)
    return time.gmtime() - start, forest.score(test_images, test_labels)


if __name__ == "__main__":
    # Read the images and theirs labels
    images, labels = readTrafficSigns('GTSRB/Final_Training/Images')

    # Pad and scale images
    pad_images(images)
    scale_images(images)

    # Split data
    images, test_data, labels, test_labels = split_data(
        images, labels, 0.8, 30)

    # Non augmented data and labels
    non_augmented_data, non_augmented_labels = images, labels

    # Augmentation
    images, labels = augmentation(np.array(images), np.array(labels))

    # Data reshaping
    images = reshape(images)

    # Model training with augmentetion
    forest = RandomForestClassifier(
        n_estimators=50, max_depth=30, n_jobs=4, random_state=0)
    forest.fit(images, labels)

    # Shuffle test data, reshape it and score
    test_data, test_labels = shuffle(test_data, test_labels)
    test_data = reshape(test_data)
    print(
        f'Validation accuracy on augmented data: {forest.score(test_data, test_labels) * 100}')

    # Read test data, pad, scale, reshape and score
    test_data, test_labels = readTestData()
    pad_images(test_data)
    scale_images(test_data)
    test_data2 = reshape(test_data)
    print(
        f'Test accuracy on augmented data: {forest.score(test_data2, test_labels) * 100}', end='\n\n')

    # Get model's prediction
    prediction = forest.predict(test_data2)

    # Calculate overall precision and recall
    print(
        f'Precision: {precision_score(test_labels, prediction, average="micro")}')
    print(
        f'Recall: {recall_score(test_labels, prediction, average="micro")}', end='\n\n')

    # Show some missclasified images
    counter = 5
    for i in range(len(test_data2)):
        if counter == 0:
            break
        if prediction[i] != test_labels[i]:
            print(
                f'Real type: {test_labels[i]}, predicted {prediction[i]}, image {i}.ppm')
            counter -= 1
    print('\n')

    # Model without augmentation
    forest = RandomForestClassifier(
        n_estimators=50, max_depth=30, n_jobs=4, random_state=0)
    forest.fit(reshape(non_augmented_data), non_augmented_labels)
    print(
        f'Test accuracy on not augmented data: {forest.score(test_data2, test_labels) * 100}', end='\n\n')

    tm, accuracy = list(), list()
    images = reshape(non_augmented_data)
    labels = non_augmented_labels

    # Model with size 15x15
    t, a = train_model_size(images, labels, test_data2, test_labels, 15)
    tm.append(t)
    accuracy.append(a)

    # Model with size 30x30
    t, a = train_model_size(images, labels, test_data2, test_labels, 30)
    tm.append(t)
    accuracy.append(a)

    # Model with size 50x50
    t, a = train_model_size(images, labels, test_data2, test_labels, 50)
    tm.append(t)
    accuracy.append(a)

    # Model with size 65x65
    t, a = train_model_size(images, labels, test_data2, test_labels, 65)
    tm.append(t)
    accuracy.append(a)

    # Model with size 80x80
    t, a = train_model_size(images, labels, test_data2, test_labels, 80)
    tm.append(t)
    accuracy.append(a)

    print(tm)
    plt.hist(tm, label='Time for execution')
    plt.show()

    print(accuracy)
    plt.hist(accuracy, label='Accuracy')
    plt.show()
