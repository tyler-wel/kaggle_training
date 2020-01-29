import os
# conda install -c anaconda pillow
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint # colored prints
import json

train_path = "../images/dogs/train/"

img_dict = {'dog_1' : "00a338a92e4e7bf543340dc849230e75.jpg",
            'dog_2' : "0b345d4f2434903c374ad8b8513a289b.jpg",
            'dog_3' : "0db44ddb42bf1f97de987abe2bf01839.jpg",
            'dog_4' : "01f8540fb1084107a6eb3e528f82c1aa.jpg"}


# The below functions are copy and pasted (very slightly modified) from the Kaggle/learntools repository for learning purposes

# Exercise 1
def load_my_image(key):
    '''returns array containing greyscale values for supplied file (at thumbnail size)'''
    image_color = Image.open(os.path.join(train_path, img_dict[key]) ).resize((135, 188), Image.ANTIALIAS)
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

# Exercise 3
def decode_predictions(preds, top=5, class_list_path='../pre-trained/resnet50/imagenet_class_index.json'):
  """Decodes the prediction of an ImageNet model.
  Arguments:
      preds: Numpy tensor encoding a batch of predictions.
      top: integer, how many top-guesses to return.
      class_list_path: Path to the canonical imagenet_class_index.json file
  Returns:
      A list of lists of top class prediction tuples
      `(class_name, class_description, score)`.
      One list of tuples per sample in batch input.
  Raises:
      ValueError: in case of invalid shape of the `pred` array
          (must be 2D).
  """
  if len(preds.shape) != 2 or preds.shape[1] != 1000:
    raise ValueError('`decode_predictions` expects '
                     'a batch of predictions '
                     '(i.e. a 2D array of shape (samples, 1000)). '
                     'Found array with shape: ' + str(preds.shape))
  CLASS_INDEX = json.load(open(class_list_path))
  results = []
  for pred in preds:
    top_indices = pred.argsort()[-top:][::-1]
    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
    result.sort(key=lambda x: x[2], reverse=True)
    results.append(result)
  return results