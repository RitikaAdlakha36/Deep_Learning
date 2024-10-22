# Validation data generator (only rescale, no augmentation)
valid_datagen = ImageDataGenerator(
    rescale=1.0/255.0,               # Normalize pixel values for validation set
    validation_split=0.2
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/extracted_faces',
    target_size=(224, 224),          # Resize all images to 224x224
    batch_size=32,
    class_mode='categorical',
    subset='training'                # Use the training subset
)

# Validation data generator
valid_generator = valid_datagen.flow_from_directory(
    '/content/drive/MyDrive/extracted_faces',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'              # Use the validation subset
)
