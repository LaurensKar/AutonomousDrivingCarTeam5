import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import random
import imgaug.augmenters as iaa
import keras
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense


directory = "C:\Thomas More 2023-2024\AI\AutonomousDrivingCarTeam5\DeepLearning"
columns = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
data = pd.read_csv(os.path.join(directory, "driving_log.csv"), names=columns)
pd.set_option("display.max_colwidth", None)
data.head()

def pathleaf(path):
    head, tail = os.path.split(path)
    return tail

data["center"] = data["center"].apply(pathleaf)
data["left"] = data["left"].apply(pathleaf)
data["right"] = data["right"].apply(pathleaf)
data.head()

plt.hist(data["steering"], bins=25, edgecolor="black")
plt.xlabel("Steering Angle")
plt.ylabel("Number of Samples")
plt.title("Distribution of Steering Angles")
plt.show()

zero_indices = data[data["steering"] == 0].index

# Randomly select a portion of the zero values to keep
selected_zero_indices = np.random.choice(zero_indices, size=600, replace=False)

# Combine the selected zero indices with the indices of non-zero values
selected_indices = np.concatenate([selected_zero_indices, data[data["steering"] != 0].index])

# Create a new DataFrame with the selected indices
selected_data = data.loc[selected_indices].copy()


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'data' is your DataFrame and 'steering' is the column of interest
plt.hist(selected_data["steering"], bins=25, edgecolor="black")
plt.xlabel("Steering Angle")
plt.ylabel("Number of Samples")
plt.title("Distribution of Steering Angles")
plt.show()

def load_img_steering(datadir, df):
    image_paths = []
    steerings = []

    for i in range(len(df)):
        center, left, right = df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2]
        
        image_paths.extend([
            os.path.join(datadir, center.strip()),
            os.path.join(datadir, left.strip()),
            os.path.join(datadir, right.strip())
        ])
        
        steering = float(df.iloc[i, 3])
        steerings.extend([steering, steering + 0.15, steering - 0.15])

    return np.asarray(image_paths), np.asarray(steerings)


image_paths, steerings = load_img_steering(dir + "/IMG", selected_data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

print("Training Samples: {}\nValid Samples: {}".format(len(X_train), len(X_valid)))

plt.figure(figsize=(12, 6))

# Plot histogram for the training set
plt.subplot(1, 2, 1)
plt.hist(y_train, bins=25, edgecolor="black", color='blue')
plt.xlabel("Steering Angle")
plt.ylabel("Number of Samples")
plt.title("Training Set")

# Plot histogram for the validation set
plt.subplot(1, 2, 2)
plt.hist(y_valid, bins=25, edgecolor="black", color='red')
plt.xlabel("Steering Angle")
plt.ylabel("Number of Samples")
plt.title("Validation Set")

plt.tight_layout() 
plt.show()

def zoom(image, scale_factor):
    zoom_augmenter = iaa.Affine(scale=(1, scale_factor))
    augmented_image = zoom_augmenter.augment_image(image)
    return augmented_image


image_path = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image_path)

zoom_factor = 2
zoomed_image = zoom(original_image, zoom_factor)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title("Original Image")

axs[1].imshow(zoomed_image)
axs[1].set_title("Zoomed Image (Scale Factor: {})".format(zoom_factor))

plt.show()

def random_flip(image, steering_angle):
    flipped_image = cv2.flip(image, 1)
    flipped_steering_angle = -steering_angle
    return flipped_image, flipped_steering_angle

random_index = random.randint(0, 1000)
image_path = image_paths[random_index]
steering_angle = steerings[random_index]

original_image = mpimg.imread(image_path)
flipped_image, flipped_steering_angle = random_flip(original_image, steering_angle)

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(original_image)
axs[0].set_title(f"Original Image - Steering Angle: {steering_angle}")

axs[1].imshow(flipped_image)
axs[1].set_title(f"Flipped Image - Steering Angle: {flipped_steering_angle}")

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        scale_factor = 1.8  # You can adjust this value
        image = zoom(image, scale_factor)
    if np.random.rand() < 0.5:
        image, steering_angle = random_flip(image, steering_angle)
    return image, steering_angle

    def img_preprocess(img):
    ## Crop image to remove unnecessary features
    img = img[60:135, :, :]
    ## Change to YUV image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    ## Gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    ## Decrease size for easier processing
    img = cv2.resize(img, (200, 66))
    ## Normalize values
    img = img / 255
    return img

    def visualize_original_and_preprocessed(image_paths, index):
    image_path = image_paths[index]
    original_image = mpimg.imread(image_path)
    preprocessed_image = img_preprocess(original_image)

    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()

    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")

    axs[1].imshow(preprocessed_image)
    axs[1].set_title("Preprocessed Image")

    plt.show()

random_index = random.randint(0, len(image_paths) - 1)
visualize_original_and_preprocessed(image_paths, random_index)

def generate_batch(image_paths, steering_ang, batch_size, istraining):
    while True:
        batch_img, batch_steering = [], []

        for _ in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            im, steering = random_augment(image_paths[random_index], steering_ang[random_index]) if istraining else (mpimg.imread(image_paths[random_index]), steering_ang[random_index])
            im = img_preprocess(im)

            batch_img.append(im)
            batch_steering.append(steering)

        yield np.asarray(batch_img), np.asarray(batch_steering)


# Create training batch
x_train_gen, y_train_gen = next(generate_batch(X_train, y_train, batch_size=1, istraining=True))

# Generate validation batch
x_valid_gen, y_valid_gen = next(generate_batch(X_valid, y_valid, batch_size=1, istraining=False))

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(x_train_gen[0])
axs[0].set_title("Training Image")

axs[1].imshow(x_valid_gen[0])
axs[1].set_title("Validation Image")

plt.show()

def NvidiaModel():
  model = Sequential()
  model.add(Convolution2D(24,(5,5),strides=(2,2),input_shape=(66,200,3),activation="elu"))
  model.add(Convolution2D(36,(5,5),strides=(2,2),activation="elu"))
  model.add(Convolution2D(48,(5,5),strides=(2,2),activation="elu")) 
  model.add(Convolution2D(64,(3,3),activation="elu"))   
  model.add(Convolution2D(64,(3,3),activation="elu"))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100,activation="elu"))
  model.add(Dropout(0.5))
  model.add(Dense(50,activation="elu"))
  model.add(Dropout(0.5))
  model.add(Dense(10,activation="elu"))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.compile(optimizer=Adam(learning_rate=1e-3),loss="mse")
  return model

model = NvidiaModel()
print(model.summary())

history = model.fit(
    batch_generator(X_train, y_train, 100, 1),
    steps_per_epoch=300,
    epochs=10,
    validation_data=batch_generator(X_valid, y_valid, 100, 0),
    validation_steps=200,
    verbose=1,
    shuffle=1,
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["training", "validation"])
plt.title("Loss")
plt.xlabel("Epoch")

model.save('model.h5')