import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras import layers
from keras import models
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
import pdb
from time import time

np.random.seed(123)


#Configure TensorFlow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# A function to plot a matrix
def plot_image(data, colorspace='binary'):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.imshow(data, cmap=colorspace)


def normalize_array(array):
    array_min = array.min(axis=(0, 1), keepdims=True)
    array_max = array.max(axis=(0, 1), keepdims=True)
    normalized_array = (array - array_min)/(array_max - array_min)

    return normalized_array

def normalize_with_preset(array, max_value, min_value):
    normalized_array = (array - min_value)/(max_value - min_value)

    return normalized_array


def recreate_image(labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = labels[label_idx]
            label_idx += 1
    return image

def overlay_images(image1, image2, colorspace1='binary', colorspace2='magma', transparency=0.5):
    plt.imshow(image1, cmap=colorspace1, interpolation='nearest')
    plt.imshow(image2, cmap=colorspace2, alpha=transparency, interpolation='bilinear')
    plt.show()

def flatten_normalize_neighbors(training_set, global_max=233, global_min=69):
    flattened_neighbors = []
    neighbors = training_set['neighbors']

    for i in range(len(neighbors)):
        numpy_array = np.array(neighbors[i]).reshape(-1, 1)
        normalized_neighbors = normalize_with_preset(numpy_array, global_max, global_min)
        flattened_neighbors.append(normalized_neighbors)

    return np.array(flattened_neighbors).reshape((len(flattened_neighbors), -1))

def extract_values_from_training_data_locations(training_data, image_channel):
    data_values = []

    for row in training_data:
        data_values.append(np.array([(image_channel[row[0], row[1]]), row[2]]))

    data_values = np.array(data_values)

    return data_values

def extract_neighborhood_values_from_training_data_locations(training_data, image_channel, neighborhood_size):
    data_values = []

    for row in training_data:
        data = image_channel[(row[0]-neighborhood_size):(row[0]+neighborhood_size), (row[1]-neighborhood_size):(row[1]+neighborhood_size)]
        data_values.append(np.array([data, row[2]]))

    data_values = np.array(data_values)

    return data_values

# Load up the data

#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

data = np.load('./data/neighborhood_training_data_with_labels.npy')

# Extract the data
data_list = []
label_list = []
for item in data:
    data_list.append(item[0])
    label_list.append(item[1])
    
data_list_array = np.array(data_list)
label_list_array = np.array(label_list)

# Precondition the data contents
#train_images = train_images.reshape((-1, 10, 10, 1))
# Preconditioning data
data_list_set = data_list_array.reshape((-1, 10, 10, 1))
label_list_set = to_categorical(label_list_array)

#Split the data
X_train, X_test, y_train, y_test = train_test_split(data_list_set, label_list_set, test_size=0.05, random_state=123)

model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(10, 10, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=200, verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy', test_acc)

print('Executing a classification...')

# Load up the RGB image
rgb_image = np.load('./data/rgb_subset.npy')
red_channel = rgb_image[:, :, 0]
pdb.set_trace()
# test_block = rgb_image[:, :, 0][2000-5:2000+5, 2000-5:2000+5].reshape((-1, 10, 10, 1))
width = rgb_image.shape[0]
height = rgb_image.shape[1]

# Do the classification
t0 = time()
prediction_map = np.zeros((width, height))
for i in range(5, width-5):
    for j in range(5, height-5):
        # test = subset[i-5:i+5, j-5:j+5].reshape(1, -1)
        test = red_channel[i-5:i+5, j-5:j+5].reshape(-1, 10, 10, 1)
        prediction = model.predict(test)
        # Get the max probability
        prediction_map[i][j] = prediction.argmax(axis=1)[0]
print("done in %0.3fs." % (time() - t0))

pdb.set_trace()
