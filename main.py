from readTrafficSigns import readTrafficSigns
import numpy as np

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
            images[i] = np.pad(im, [(0, width - height), (0, 0), (0, 0)], mode='constant')
        else:
            images[i] = np.pad(im, [(0, 0), (0, height - width), (0, 0)], mode='constant')

if __name__ == "__main__":
    # Read the images and theirs labels
    images, labels = readTrafficSigns('GTSRB/Final_Training/Images')
    
    pad_images(images)
    print([len(i) == len(i[0]) for i in images])