# %% [markdown]
# # **LAST PROJECT: IMAGE CLASSIFICATION MODEL DEPLOYMENT 👨🏽‍💻**

# %% [markdown]
# * Name         : Lintang Nagari
# * Email        : unggullintangg@gmail.com
# * Linkedin     : <a href='https://www.linkedin.com/in/lintangnagari/'>Lintang Nagari</a>
# * Github       : <a href='https://github.com/lnt-ngr'>lnt-ngr</a>

# %% [markdown]
# **Here are the submission criteria you must meet:**
# 
# * Use any dataset, but it must have **at least 1000 images**.
# * The dataset should not have been used in any previous machine learning submissions.
# * Divide the dataset into **80% training set and 20% test set**.
# * Model must use a `sequential model`.
# * Model must include `Conv2D Maxpooling Layer`.
# * Achieve **at least 80% accuracy** on both training and validation sets.
# * Use Callbacks.
# * Create plots for model accuracy and loss.
# * Write code to save the model in *TF-Lite format*.
# 
# **Dataset : https://www.kaggle.com/datasets/shiv28/animal-5-mammal**
# 
# **About Dataset**
# 
# It contains about 15K medium quality animal images belonging to 10 categories: **dog, cat, horse, elephant ,lion**. All the images have been collected from `google images` and have been checked by human. There is some erroneous data to simulate real conditions (eg. images taken by users of your app).

# %% [markdown]
# ### __IMPORT LIBRARY__

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications import InceptionV3, Xception, ResNet50V2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

print(tf.__version__)

# %% [markdown]
# ### __DATASET__

# %%
# Dataset Train
dataset = '/content/drive/MyDrive/DATASET/animals/dataframe'

# Loop through each folder within the dataset directory
def list_dirs_and_files(directory_path):
    print('Folder and its number of files:')

    folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    total_files = 0
    total_folders = len(folders)

    for folder in folders:
        folder_path = os.path.join(directory_path, folder)
        files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        file_count = len(files)
        total_files += file_count
        print(f"{folder} [{file_count}] files")

    print(f"\nTotal files in all folders: {total_files}")
    print(f"Total folders (label): {total_folders}")

print('\nDirectory dataset info')
list_dirs_and_files(dataset)

def list_various_resolutions(directory):
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
    image_sizes = []

    for folder in folders:
        folder_path = os.path.join(directory, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                with Image.open(file_path) as image:
                    width, height = image.size
                    image_sizes.append(f'{width}x{height}')

    unique_sizes = set(image_sizes)
    first_size_unique = 8

    print(f'Size all images: {len(image_sizes)}')
    print(f'Size unique images: {len(unique_sizes)}')
    print(f'First {first_size_unique} unique images: {list(unique_sizes)[:first_size_unique]}')

print('\nList various resolutions')
list_various_resolutions(dataset)

# %% [markdown]
# ## __DARA PREPROCESSING__

# %%
# Dataset
baseDir = '/content/drive/MyDrive/DATASET/animals/dataframe'
os.listdir(baseDir)

# %%
!pip install split-folders
import splitfolders as sf

# Split directory
sf.ratio(
    baseDir,
    output = os.path.join('/content/drive/MyDrive/DATASET/animals'),
    seed   = None,
    ratio  = (0.8, 0.2)
)

# %%
imageDir = '/content/drive/MyDrive/DATASET/animals'

trainDirelephant  = os.path.join(imageDir, 'train/elephant')
trainDirlion      = os.path.join(imageDir, 'train/lion')
trainDirdog       = os.path.join(imageDir, 'train/dog')
trainDircat       = os.path.join(imageDir, 'train/cat')
trainDirhorse     = os.path.join(imageDir, 'train/horse')

valDirelephant    = os.path.join(imageDir, 'val/elephant')
valDirlion        = os.path.join(imageDir, 'val/lion')
valDirdog         = os.path.join(imageDir, 'val/dog')
valDircat         = os.path.join(imageDir, 'val/cat')
valDirhorse       = os.path.join(imageDir, 'val/horse')

# %%
# Count train and val image
trainSet = (
      len(os.listdir(trainDirelephant))
    + len(os.listdir(trainDirlion))
    + len(os.listdir(trainDirdog))
    + len(os.listdir(trainDircat))
    + len(os.listdir(trainDirhorse))
)

valSet = (
      len(os.listdir(valDirelephant))
    + len(os.listdir(valDirlion))
    + len(os.listdir(valDirdog))
    + len(os.listdir(valDircat))
    + len(os.listdir(valDirhorse))
)

print(f'Train Set      : {trainSet}')
print(f'Validation Set : {valSet}')

# %%
# List directory of train and validation image
train_dir = os.path.join(imageDir, 'train')
val_dir   = os.path.join(imageDir, 'val')

print(os.listdir(train_dir))
print(os.listdir(val_dir))

# %%
#Image Generator
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 45,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    fill_mode = 'nearest',
    validation_split = 0.2,)

test_datagen = ImageDataGenerator(rescale=1./255)

# %%

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 64,
    shuffle = True,
    color_mode = 'rgb',
    class_mode = 'categorical',
)

val_gen = test_datagen.flow_from_directory(
    val_dir,
    target_size = (224, 224),
    batch_size = 64,
    shuffle = True,
    color_mode = 'rgb',
    class_mode = 'categorical',
)

# %% [markdown]
# ## __MODELLING__

# %%
# Transfer Learning from ResNetv50
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

# %%
# model's architecture
model = tf.keras.models.Sequential([
    base_model,
    Conv2D(128, (3, 3), activation = 'relu', padding = 'same'),
    MaxPooling2D((2,2)),
    Conv2D(256, (3, 3), activation = 'relu', padding = 'same'),
    MaxPooling2D((2,2)),
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(2048, activation = 'relu'),
    Dense(1024, activation = 'relu'),
    Dropout(.3),
    Dense(512, activation = 'relu'),
    Dense(256, activation = 'relu'),
    Dropout(.3),
    Dense(128, activation = 'relu'),
    Dense(5, activation='softmax')
])

model.summary()

# %%
# Compiling the model
model.compile(
    loss ='categorical_crossentropy',
    optimizer = tf.optimizers.Adam(learning_rate = 0.001),
    metrics = ['accuracy']
)

# %% [markdown]
# ## __CALLBACK__

# %%
# Stop training callback
class stopCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.92 and logs.get('val_accuracy') > 0.92):
            print('\nAccuracy and Validation Accuracy reach > 92%')
            self.model.stop_training = True

MyCallbacks = stopCallback()

# ReduceLROnPlateau callback
reduceLROP   = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)


# %% [markdown]
# ## __FUNGSI FIT__

# %%
epoch_number = 50
history = model.fit(
    train_gen,
    epochs = epoch_number,
    steps_per_epoch = 16,
    validation_data = val_gen,
    validation_steps = 16,
    callbacks = [MyCallbacks, reduceLROP],
    verbose = 2,
)

# %% [markdown]
# ## __PLOTTING__

# %%
#loss Plot
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc="upper right")
plt.show()

# %%
#Accuracy Plot
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()

# %% [markdown]
# ## __SAVE MODEL TF-LITE__

# %%
import warnings

warnings.filterwarnings('ignore') # Ignore Warning

# Convert model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with tf.io.gfile.GFile('Animal_ClassV1.tflite', 'wb') as f:
  f.write(tflite_model)

# %%
from google.colab import files

# Download the flower model
files.download('/content/Animal_ClassV1.tflite')

print('`model.tflite` has been downloaded')


