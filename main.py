# Import necessary libraries
from tensorflow import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------
# Loading a pretrained model
# ---------------------------------
model = keras.applications.Xception(weights="imagenet")

#model = keras.models.load_model("convnet_from_scratch_with_augmentation.keras")

# ---------------------------------
# Pre-processing a single image (cat)
# ---------------------------------
img_path = keras.utils.get_file(
    fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg")

def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)  # Resize image
    array = keras.utils.img_to_array(img)  # Convert to array
    array = np.expand_dims(array, axis=0)  # Add a batch dimension
    return array

img_tensor = get_img_array(img_path, target_size=(180, 180))

# ---------------------------------
# Instantiating a model that returns activations for specific layers
# ---------------------------------
from tensorflow.keras import layers

layer_outputs = []  # Store layer outputs (activations)
layer_names = []    # Store layer names

# Extract activations for Conv2D and MaxPooling2D layers
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)

# Create a new model that outputs activations
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# ---------------------------------
# Computing activations for the cat image
# ---------------------------------
activations = activation_model.predict(img_tensor)

# ---------------------------------
# Using Xception model for feature extraction
# ---------------------------------
model_xception = keras.applications.xception.Xception(weights="imagenet", include_top=False)

for layer in model_xception.layers:
    if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
        print(layer.name)

layer_name = "block3_sepconv1"
layer = model_xception.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model_xception.input, outputs=layer.output)

activation_xception = feature_extractor(keras.applications.xception.preprocess_input(img_tensor))

# ---------------------------------
# Filter Visualization - Generating patterns
# ---------------------------------
img_width = 200
img_height = 200

@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)
        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image += learning_rate * grads
    return image

def generate_filter_pattern(filter_index):
    iterations = 30
    learning_rate = 10.0
    image = tf.random.uniform(minval=0.4, maxval=0.6, shape=(1, img_width, img_height, 3))
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0].numpy()

def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = np.clip(image, 0, 255).astype("uint8")
    image = image[25:-25, 25:-25, :]  # Crop the image
    return image

# Visualizing multiple filters
all_images = []
for filter_index in range(64):
    print(f"Processing filter {filter_index}")
    image = deprocess_image(generate_filter_pattern(filter_index))
    all_images.append(image)

# Stitch all filter images together in a grid
margin = 5
n = 8
cropped_width = img_width - 25 * 2
cropped_height = img_height - 25 * 2
width = n * cropped_width + (n - 1) * margin
height = n * cropped_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 3))

for i in range(n):
    for j in range(n):
        image = all_images[i * n + j]
        row_start = (cropped_width + margin) * i
        row_end = (cropped_width + margin) * i + cropped_width
        column_start = (cropped_height + margin) * j
        column_end = (cropped_height + margin) * j + cropped_height
        stitched_filters[row_start: row_end, column_start: column_end, :] = image

keras.utils.save_img(f"filters_for_layer_{layer_name}.png", stitched_filters)

# ---------------------------------
# Grad-CAM: Visualizing class-specific activations
# ---------------------------------
model = keras.applications.xception.Xception(weights="imagenet")
img_path = keras.utils.get_file(
    fname="elephant.jpg", origin="https://img-datasets.s3.amazonaws.com/elephant.jpg")

def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = keras.applications.xception.preprocess_input(array)
    return array

img_array = get_img_array(img_path, target_size=(299, 299))

# Prepare model for Grad-CAM
last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = ["avg_pool", "predictions"]

last_conv_layer = model.get_layer(last_conv_layer_name)
last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
for layer_name in classifier_layer_names:
    x = model.get_layer(layer_name)(x)
classifier_model = keras.Model(classifier_input, x)

# Generate heatmap for Grad-CAM
with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_array)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
last_conv_layer_output = last_conv_layer_output.numpy()[0]

for i in range(pooled_grads.shape[-1]):
    last_conv_layer_output[:, :, i] *= pooled_grads[i]

heatmap = np.mean(last_conv_layer_output, axis=-1)

# Processing the heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

# Overlay heatmap on the original image
img = keras.utils.load_img(img_path)
img = keras.utils.img_to_array(img)
heatmap = np.uint8(255 * heatmap)
jet = cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
jet_heatmap = keras.utils.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = keras.utils.img_to_array(jet_heatmap)

superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = keras.utils.array_to_img(superimposed_img)
save_path = "elephant_cam.jpg"
superimposed_img.save(save_path)

# ---------------------------------
# Displaying the final image with heatmap
# ---------------------------------
plt.imshow(superimposed_img)
plt.axis('off')
plt.show()
