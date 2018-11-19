# Import Numpy
import numpy as np
# Matplot Lib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import pdb
import time
import csv

# Create a global variable to store all the data points
trained_data = []

# A function to plot a matrix
def plot_image(data, colorspace='binary'):
    global image_matrix
    image_matrix = data
    print('Displaying image...')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.imshow(data, cmap=colorspace)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def display_image(data, colormap='binary'):
    print('Displaying subset...')


def normalize_array(array):
    array_min = array.min(axis=(0, 1), keepdims=True)
    array_max = array.max(axis=(0, 1), keepdims=True)
    normalized_array = (array - array_min)/(array_max - array_min)

    return normalized_array


# A function that executes when the mouse is clicked
def onclick(event):
    global ix, iy
    # If the left mouse was clicked act accordingly
    if(event.button == 1):
        # Check to see if the data sent is just an integer or something else so that we can do any kind of slice.
        ix, iy = int(event.xdata), int(event.ydata)
        # Get the pixel value
        value = image_matrix[iy, ix]
        trained_data.append([ix, iy, 1])
        print(trained_data)
    # If button 2 was presed
    elif(event.button == 2):
        on_save_button(trained_data)

    # If button 3 was pressed label the zeros
    elif(event.button == 3):
        # Check to see if the data sent is just an integer or something else so that we can do any kind of slice.
        ix, iy = int(event.xdata), int(event.ydata)
        # Get the pixel value
        value = image_matrix[iy, ix]
        trained_data.append([ix, iy, 0])
        print(trained_data)

def on_save_button(data):
    print("Saving File")
    np_data = np.array(data)
    np.save('./training_sets/pixelwise/training_set_pixelwise_' + str(time.time()) + '.npy', np_data)


# A function to store values in a CSV file
#timestamp = str(time.time())
#def store_data(new_data):
#    print(new_data)
#    # with open(r'training_sets/training_set_' + timestamp + '.txt', 'a') as f:
#    #     writer = csv.writer(f)
#    #     writer.writerow(fields)
#    with open(r'training_sets/training_set_neighborhood_' + timestamp + '.txt', 'a') as f:
#        np.savetxt(f, new_data, fmt='%3.0f')
#    # print(training_data_neighborhood_array)
