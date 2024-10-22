import os
import cv2
import wandb
import time

# API Key :- 69cfa95b6857e105a17c646eb69b8bbb16879de4

# Define the target directory for organized images
target_dir = '/content/drive/MyDrive/preprocess_faces'
image_size = (150, 150)
# Define the source directory where the extracted faces are located
source_dir = "/content/drive/MyDrive/extracted_faces"

# Initialize WandB project for image preprocessing tracking
wandb.init(project="image-preprocessing", name="dementia-classification", config={
    "image_size": image_size,
    "source_dir": source_dir,
    "target_dir": target_dir
})

# Create target subdirectories for each class
classes = ['dementia', 'mci', 'normal']
for class_name in classes:
    os.makedirs(os.path.join(target_dir, class_name), exist_ok=True)

# Function to classify, resize, and move images based on folder names
def classify_resize_and_move_images(source_dir, target_dir, image_size):
    total_images_processed = 0
    start_time = time.time()

    # Loop through all folders in the source directory
    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)

        if os.path.isdir(folder_path):
            if folder_name.startswith('d'):  # Dementia
                class_dir = os.path.join(target_dir, 'dementia')
            elif folder_name.startswith('m'):  # MCI
                class_dir = os.path.join(target_dir, 'mci')
            elif folder_name.startswith('n'):  # Normal
                class_dir = os.path.join(target_dir, 'normal')
            else:
                continue  # Skip if the folder name doesn't start with 'd', 'm', or 'n'

            # Create the folder inside the class directory
            target_folder = os.path.join(class_dir, folder_name)
            os.makedirs(target_folder, exist_ok=True)

            # Process each image in the folder
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                if image_file.endswith('.png'):  # images are in .png format
                    try:
                        # Open the image using OpenCV
                        img = cv2.imread(image_path)

                        # Check if resizing is necessary
                        if img.shape[:2] != image_size:
                            # Resize the image using OpenCV
                            img_resized = cv2.resize(img, image_size)
                        else:
                            img_resized = img

                        # Save the resized image using OpenCV
                        cv2.imwrite(os.path.join(target_folder, image_file), img_resized)

                        # Update the processed image count
                        total_images_processed += 1

                        # Log progress to WandB
                        wandb.log({"Processed Images": total_images_processed})

                    except KeyboardInterrupt:
                        print("Process interrupted by user. Exiting gracefully.")
                        wandb.finish()
                        return

    end_time = time.time()
    duration = end_time - start_time

    # Log summary metrics to WandB
    wandb.log({
        "Total Images Processed": total_images_processed,
        "Total Processing Time (seconds)": duration

    })

    wandb.finish()

# Define the source directory where the extracted faces are located
source_dir = "/content/drive/MyDrive/extracted_faces"

# Call the function to classify, resize, and move images
classify_resize_and_move_images(source_dir, target_dir, image_size)

# List the contents of the target directory to confirm
os.listdir(target_dir)
