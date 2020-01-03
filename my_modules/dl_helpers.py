import os
# conda install -c anaconda pillow
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint # colored prints

img_dict = {'dog_1' : "./images/dogs/train/00a338a92e4e7bf543340dc849230e75.jpg",
            'dog_2' : "./images/dogs/train/0b345d4f2434903c374ad8b8513a289b.jpg"}


# The below functions are copy and pasted (very slightly modified) from the Kaggle/learntools repository for learning purposes

def load_my_image(key):
    '''returns array containing greyscale values for supplied file (at thumbnail size)'''
    image_color = Image.open(img_dict[key]).resize((135, 188), Image.ANTIALIAS)
    image_grayscale = image_color.convert('L')
    image_array = np.asarray(image_grayscale)
    return(image_array)
  
def visualize_conv(image, conv, title=''):
  '''visualizes the provided convolution based on the provided image'''
  conv_array = np.array(conv)
  vertical_padding = conv_array.shape[0] - 1
  horizontal_padding = conv_array.shape[1] - 1
  conv_out = scale_for_display(apply_conv_to_image(conv_array, image),
                               contrast_factor=350)
  show(np.hstack([image[:-vertical_padding, :-horizontal_padding], conv_out]), False, title)
    
def scale_for_display(image, contrast_factor=256):
    '''Scales numpy array containing image data to be integers in range [0, 256]'''
    out = image - image.min()
    out = (out / out.max() * contrast_factor).clip(0, 256)
    return out.astype(int)

def apply_conv_to_image(conv, image):
    '''Applies conv (supplied as list of lists) to image (supplied as numpy array). Returns output array'''
    assert(type(image) == np.ndarray)
    image_height, image_width = image.shape
    conv_array = np.array(conv)
    conv_height, conv_width = conv_array.shape
    filtered_image_height = image.shape[0] - conv_height + 1
    filtered_image_width = image.shape[1] - conv_width + 1
    # cprint('TESTING: image, conv, filtered height/width', 'red')
    # cprint('image', 'cyan')
    # print(f'image_shape: {image.shape}')
    # cprint('conv', 'cyan')
    # print(f'conv_height: {conv_height}, conv_width: {conv_width}')
    # cprint('filtered', 'cyan')
    # print(f'filtered_height: {filtered_image_height}, filtered_width: {filtered_image_width}')
    filtered_image = np.zeros((filtered_image_height, filtered_image_width))
    for i in range(filtered_image_height):
        for j in range(filtered_image_width):
            filtered_image[i, j] = apply_conv_locally(conv_array, image[i:i+conv_height, j:j+conv_width])
    return(filtered_image)

def apply_conv_locally(conv, image_section):
    '''Returns output of applying conv to image_section. Both inputs are numpy arrays.
    image_section is assumed to be same size/shape as conv.
    '''
    out = (conv * image_section).sum()
    return out

def show(image, scale_before_display=True, title=''):
    '''Displays numpy array as image.  Scale_before_display ensures values are integers in [0, 256]'''
    if scale_before_display:
        to_display = scale_for_display(image)
    else:
        to_display = image
    plt.imshow(to_display, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()