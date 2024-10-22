import matplotlib.pyplot as plt
# Plot accuracy and loss graphs
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.title('Training and Validation Loss')

plt.tight_layout()

# Initialize a new WandB run to log the plots
# You can use the same project name or a different one
with wandb.init(project="dementia-classification", name="plots") as run:
    # Log the plots to Weights & Biases
    wandb.log({"Accuracy vs Epoch": wandb.Image(plt)})