# Define a lightweight CNN model
def create_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional Block 1
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Block 2
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Block 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))  # Dropout to prevent overfitting
    model.add(layers.BatchNormalization())


    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting
    model.add(layers.BatchNormalization())


    return model

# Parameters
input_shape = (224, 224, 3)  # input size
num_classes = 24 # 3 classes: dementia, mci, normal

# Create the model
model = create_model(input_shape, num_classes)

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()