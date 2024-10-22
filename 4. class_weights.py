import numpy as np  # Ensure this is imported
from sklearn.utils import class_weight

# Get class labels from the train_generator
class_labels = list(train_generator.class_indices.keys())  

# Get the true labels from the train_generator
true_labels = train_generator.classes

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(true_labels),  # np.unique requires numpy to be imported
    y=true_labels
)

# Convert class weights into a dictionary
class_weights_dict = dict(enumerate(class_weights))
class_weights_dict