
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the Xception model
model = keras.applications.xception.Xception(weights="imagenet", include_top=False)

# Function to get an image array (preprocessing step)
def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = keras.applications.xception.preprocess_input(array)  # Preprocessing for Xception
    return array

# Define the loss function for a specific filter index
def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We take the mean of the activation of the filter we are trying to visualize
    filter_activation = activation[:, :, :, filter_index]
    return tf.reduce_mean(filter_activation)

# Gradient ascent step for optimizing the image to maximize the filter activation
@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return image

# Generate filter patterns for a specific filter
def generate_filter_pattern(filter_index, size=299, iterations=30, learning_rate=10.0):
    image = tf.random.uniform((1, size, size, 3), minval=0.4, maxval=0.6)
    for _ in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0].numpy()

# De-process the image (post-processing step)
def deprocess_image(image):
    image -= image.mean()
    image /= (image.std() + 1e-5)
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    return image

# Define the layer for which we want to visualize filters
layer_name = "block3_sepconv1"
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

# Visualize the first 64 filters in the layer
all_images = []
for filter_index in range(64):
    print(f"Processing filter {filter_index}")
    image = generate_filter_pattern(filter_index)
    image = deprocess_image(image)
    all_images.append(image)

# Create a grid to display all the filter visualizations
n = 8
margin = 5
cropped_width = 299
cropped_height = 299
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

for i in range(n):
    for j in range(n):
        image = all_images[i * n + j]
        row_start = i * (cropped_height + margin)
        row_end = row_start + cropped_height
        col_start = j * (cropped_width + margin)
        col_end = col_start + cropped_width
        stitched_filters[row_start:row_end, col_start:col_end, :] = image

# Save the stitched image of filters
keras.utils.save_img(f"filters_for_layer_{layer_name}.png", stitched_filters)

# Show the stitched image of filters
plt.figure(figsize=(10, 10))
plt.imshow(stitched_filters.astype("uint8"))
plt.axis("off")
plt.show()
