from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add learning rate scheduler and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-5)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

import wandb
# Import the callback from the correct location
from wandb.integration.keras import WandbCallback

# Train the model with WandB logging
wandb.init(project="dementia-classification")
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=valid_generator,
    class_weight=class_weights_dict,
    callbacks=[early_stopping,
               reduce_lr,
               WandbCallback(monitor="val_loss", mode="min", save_model=False, save_graph=False)]  # Use WandbCallback for logging
)


# Evaluate the model
loss, accuracy = model.evaluate(valid_generator)
print('Validation loss:', loss)
print('Validation accuracy:', accuracy)

# Log the final validation loss and accuracy to WandB
wandb.log({"validation_loss": loss, "validation_accuracy": accuracy})

# Get training loss and accuracy from the history object
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']


# Log training loss and accuracy to WandB
for epoch in range(len(train_loss)):
  wandb.log({
      "epoch": epoch,
      "train_loss": train_loss[epoch],
      "train_accuracy": train_accuracy[epoch],
      "val_loss": val_loss[epoch],
      "val_accuracy": val_accuracy[epoch],
  })

wandb.finish()