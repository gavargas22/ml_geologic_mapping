# Import Numpy
import numpy as np
# Matplot Lib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import pdb
import time
import csv
import pickle

# Create a global variable to store all the data points

trained_data = {"positions":[], "neighbors":[], "labels":[]}

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
    neighborhood = 5
    global ix, iy

    # If the left mouse was clicked act accordingly
    if(event.button == 1):
        # Check to see if the data sent is just an integer or something else so that we can do any kind of slice.
        ix, iy = int(event.xdata), int(event.ydata)
        # Extract the neighborhood block
        neighborhood_block = image_matrix[(iy-neighborhood):(iy+neighborhood), (ix-neighborhood):(ix+neighborhood)]
        print(neighborhood_block.shape)
        trained_data["positions"].append([ix, iy])
        trained_data["neighbors"].append([neighborhood_block.tolist()])
        trained_data["labels"].append([1])
        print(trained_data)

    # If button 2 was presed
    elif(event.button == 2):
        on_save_button(trained_data)

    # If button 3 was pressed label the zeros
    elif(event.button == 3):
        # Check to see if the data sent is just an integer or something else so that we can do any kind of slice.
        ix, iy = int(event.xdata), int(event.ydata)
        # Extract the neighborhood block
        neighborhood_block = image_matrix[(iy-neighborhood):(iy+neighborhood), (ix-neighborhood):(ix+neighborhood)]
        print(neighborhood_block.shape)
        trained_data["positions"].append([ix, iy])
        trained_data["neighbors"].append([neighborhood_block.tolist()])
        trained_data["labels"].append([0])
        print(trained_data)

def on_save_button(data):
    timestamp = str(time.time())
    pickle_out = open("./training_sets/neighborhood/training_set_neighborhood_" + timestamp + ".pickle", "wb")
    # pdb.set_trace()
    pickle.dump(trained_data, pickle_out)
    pickle_out.close()
    print("Saved trained dataset")
    # np_data = np.array(data)
    # timestamp = str(time.time())
    # np.save('./training_sets/neighborhood/training_set_neighborhood_' + timestamp + '.npy', data)
