import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Image Data Generators with adjusted augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,               # Normalize pixel values to [0, 1]
    rotation_range=10,               # Reduced rotation for better control
    width_shift_range=0.1,           # Reduced width shift
    height_shift_range=0.1,          # Reduced height shift
    shear_range=0.1,                 # Reduced shear
    zoom_range=0.1,                  # Reduced zoom
    brightness_range=[0.8, 1.2],     # Adjust brightness to simulate lighting variations
    horizontal_flip=True,            # Horizontal flip
    fill_mode='nearest',             # Fill missing pixels after transformations
    validation_split=0.2             # 20% split for validation
)
train_datagen