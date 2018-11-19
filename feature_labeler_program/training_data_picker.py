from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand

import tkinter as tk
from tkinter import filedialog

# Import the seismic utilities
import pixelwise_operations as pixelwise_ops
import neighborhood_operations as neighborhood_ops

# Import PDB
import pdb


from tkinter import Tk, Label, Button

class TrainingDataPicker:
    def __init__(self, master):
        self.master = master
        master.title("Training Data Picker")

        self.label = Label(master, text="Training Data Picker")
        self.label.pack()

        load_image_button = Button(master, text="Load Image", command=self.get_image_path)
        load_image_button.pack()

        plot_image_button = Button(master, text='Pixelwise Classifier', command=self.classify_pixels)
        plot_image_button.pack()

        plot_image_button = Button(master, text='Neighborhood Classifier', command=self.classify_neighbors)
        plot_image_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def get_image_path(self):
        # Open a dialog to allow the user to select the 3D seismic volume
        root = tk.Tk()
        root.withdraw()
        # Numpy Array load
        file_path = filedialog.askopenfilename()
        # Open Numpy file
        data = np.load(file_path)
        print(data.shape)
        # Insert raw data into memory
        global global_raw_data
        global_raw_data = data

    # Classify the pixels
    def classify_pixels(self):
        pixelwise_ops.plot_image(global_raw_data, colorspace='binary')

    # Classify neighbors
    def classify_neighbors(self):
        neighborhood_ops.plot_image(global_raw_data, colorspace='binary')

root = Tk()
my_gui = TrainingDataPicker(root)
root.mainloop()
