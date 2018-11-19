# How to use the feature labeler.

This program will run a window where you can select the image file that you want to label for generating a labeled dataset. 

Open up an image file and click with `mouse button 1` or `mouse button 2` to label 1 or 0 respectively.

There are two kinds of labeling, one labels each individual pixel and the other labels a square pixel neighborhood of size n, which stores a pixel square and label it, just like the MNIST digits.

To save measurements, click the mouse middle button (MB3) and a timestamped file will be stored in `./training_sets/[desired_classification-scheme]` subdirectory.

## Requirements

- python 3
- matplotlib
- numpy
- tkinter

## Running the app

`$ python training_data_picker.py`
