#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf

# Define the 5x5 input matrix
input_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

# Reshape to match TensorFlow's input format: (batch_size, height, width, channels)
input_tensor = input_matrix.reshape((1, 5, 5, 1))

# Define the 3x3 kernel
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

# Reshape kernel to match TensorFlow format: (height, width, input_channels, output_channels)
kernel_tensor = kernel.reshape((3, 3, 1, 1))

# Perform convolution with different parameters
def apply_convolution(stride, padding):
    return tf.nn.conv2d(input=input_tensor, filters=kernel_tensor, strides=[1, stride, stride, 1], padding=padding).numpy()

# Compute feature maps
feature_map_valid_1 = apply_convolution(stride=1, padding='VALID')
feature_map_same_1 = apply_convolution(stride=1, padding='SAME')
feature_map_valid_2 = apply_convolution(stride=2, padding='VALID')
feature_map_same_2 = apply_convolution(stride=2, padding='SAME')

# Print results
print("Stride = 1, Padding = 'VALID':\n", feature_map_valid_1.squeeze())
print("\nStride = 1, Padding = 'SAME':\n", feature_map_same_1.squeeze())
print("\nStride = 2, Padding = 'VALID':\n", feature_map_valid_2.squeeze())
print("\nStride = 2, Padding = 'SAME':\n", feature_map_same_2.squeeze())
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Task 1: Edge Detection Using Sobel Filter

def edge_detection(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Define Sobel filters
    sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    # Apply Sobel filter in X and Y direction
    sobel_x = cv2.filter2D(image, -1, sobel_x_filter)
    sobel_y = cv2.filter2D(image, -1, sobel_y_filter)
    
    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel-X')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel-Y')
    plt.axis('off')
    
    plt.show()


# In[8]:


get_ipython().system('pip install opencv-python')


# 3)TASK_1

# In[20]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the sample image (ensure you have a sample image in the same directory or provide the full path)
# For now, let's use OpenCV's built-in image as an example.
image = cv2.imread(cv2.samples.findFile('th.jpeg'), cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if image is None:
    print("Error loading image")
else:
    # Apply Sobel filter for edge detection in x-direction (horizontal edges)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)

    # Apply Sobel filter for edge detection in y-direction (vertical edges)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Display the original image and the Sobel filtered images
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel-X Edge Detection')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel-Y Edge Detection')
    plt.axis('off')

    plt.show()


# TASK_2

# In[22]:


import tensorflow as tf
import numpy as np

# Create a random 4x4 matrix
input_matrix = np.random.randint(0, 10, size=(1, 4, 4, 1))

# Convert to TensorFlow tensor
input_tensor = tf.constant(input_matrix, dtype=tf.float32)

# Apply Max Pooling operation with 2x2 filter and stride 2
max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
max_pooled = max_pooling(input_tensor)

# Apply Average Pooling operation with 2x2 filter and stride 2
average_pooling = tf.keras.layers.AvgPool2D(pool_size=(2, 2), strides=2)
avg_pooled = average_pooling(input_tensor)

# Print original, max-pooled, and average-pooled matrices
print("Original Matrix:")
print(input_matrix[0, :, :, 0])

print("\nMax Pooling Matrix:")
print(max_pooled[0, :, :, 0].numpy())

print("\nAverage Pooling Matrix:")
print(avg_pooled[0, :, :, 0].numpy())


# In[23]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Define the AlexNet architecture
def create_alexnet(input_shape=(224, 224, 3)):
    model = models.Sequential()

    # Conv2D Layer 1
    model.add(layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Conv2D Layer 2
    model.add(layers.Conv2D(256, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Conv2D Layer 3
    model.add(layers.Conv2D(384, (3, 3), activation='relu'))

    # Conv2D Layer 4
    model.add(layers.Conv2D(384, (3, 3), activation='relu'))

    # Conv2D Layer 5
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))

    # Flatten
    model.add(layers.Flatten())

    # Fully Connected Layer 1
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Fully Connected Layer 2
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Output Layer
    model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for classification

    return model

# Create the AlexNet model
alexnet_model = create_alexnet()

# Print the model summary
alexnet_model.summary()


# In[24]:


import tensorflow as tf
from tensorflow.keras import layers, models

# Define the Residual Block function
def residual_block(input_tensor, filters):
    # First Conv2D layer
    x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(input_tensor)
    x = layers.Conv2D(filters, (3, 3), activation=None, padding='same')(x)
    
    # Add the input tensor to the output of the second Conv2D layer (skip connection)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    
    return x

# Define the ResNet-like architecture
def create_resnet(input_shape=(224, 224, 3)):
    input_tensor = layers.Input(shape=input_shape)

    # Initial Conv2D Layer
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(input_tensor)

    # Apply two residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Flatten the output
    x = layers.Flatten()(x)

    # Dense Layer
    x = layers.Dense(128, activation='relu')(x)

    # Output Layer
    output_tensor = layers.Dense(10, activation='softmax')(x)  # Assuming 10 classes for classification

    # Create the model
    model = models.Model(inputs=input_tensor, outputs=output_tensor)

    return model

# Create the ResNet-like model
resnet_model = create_resnet()

# Print the model summary
resnet_model.summary()


# In[ ]:




