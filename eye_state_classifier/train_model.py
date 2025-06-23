# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout




base_path = '../input/newdataset/dataset 2'  

train_dir = os.path.join(base_path, 'train')
#val_dir = os.path.join(base_path, 'val')
test_dir = os.path.join(base_path, 'test')




from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 32

# Create ImageDataGenerator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize the images to [0, 1]
    rotation_range=30,         # Random rotation between 0 and 30 degrees
    zoom_range=0.2,            # Zoom images by up to 20%
    width_shift_range=0.2,     # Randomly shift images horizontally by 20%
    height_shift_range=0.2,    # Randomly shift images vertically by 20%
    shear_range=0.2,           # Apply shear transformation
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest'        # Fill in newly created pixels during transformations
)

# Load Train Data
train_generator = train_datagen.flow_from_directory(
    train_dir,  # path to 'train' folder
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)




#Build the CNN Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # 1 neuron for binary classification
])




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential([
    # First Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Second Convolutional Layer
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Third Convolutional Layer
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Fourth Convolutional Layer (Optional)
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten the 3D output to 1D
    Flatten(),
    
    # Fully connected (Dense) layer
    Dense(512, activation='relu'),
    Dropout(0.5),  # Increase Dropout to prevent overfitting
    Dense(1, activation='sigmoid')  # 1 neuron for binary classification
])




#compile model (Use binary crossentropy loss for binary classification):

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Print model summary
model.summary()




#Train the Model
#Use training dataset:

history = model.fit(
    train_generator,
    epochs=100,  # You can increase it later if needed
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    verbose=1
)




# Get final training accuracy and loss
final_train_accuracy = history.history['accuracy'][-1]
final_train_loss = history.history['loss'][-1]

print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")





from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create ImageDataGenerator for test data (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load Test Data
test_generator = test_datagen.flow_from_directory(
    test_dir,  # path to 'test' folder
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important: Do not shuffle for evaluation
)




# Helper function to load and preprocess an image
def load_and_preprocess_image(img_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))  # Load image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)  # Reshape for model
    return img_array

# ---- Predict on All Test Images ----

all_preds = []

print("\n--- Predictions for All Test Images ---")
for img_path in test_generator.filepaths:
    img = load_and_preprocess_image(img_path)
    pred_value = model.predict(img)[0][0]
    pred_label = "Open Eye" if pred_value > 0.5 else "Closed Eye"
    img_name = img_path.split("/")[-1]
    print(f"{img_name} → {pred_label} ({pred_value:.4f})")
    all_preds.append((img_name, pred_label, pred_value))




# True labels
true_labels = test_generator.classes  # Actual labels from generator

# Predicted labels
predicted_labels = []

for img_path in test_generator.filepaths:
    img = load_and_preprocess_image(img_path)
    pred_value = model.predict(img)[0][0]
    pred_class = 1 if pred_value > 0.5 else 0  # 1 for Open Eye, 0 for Closed Eye
    predicted_labels.append(pred_class)

# Now calculate accuracy
predicted_labels = np.array(predicted_labels)
accuracy = np.mean(predicted_labels == true_labels)
print(f"\nTest Accuracy (manual calculation): {accuracy*100:.2f}%")





# Assuming all_preds is a list of tuples: (img_name, pred_label, pred_value)
# Example: [('img1.jpg', 'Open Eye', 0.8324), ('img2.jpg', 'Closed Eye', 0.2312), ...]

# Convert to DataFrame
df = pd.DataFrame(all_preds, columns=["Image Name", "Predicted Label", "Predicted Value"])

# Save to CSV
df.to_csv("eye_state_predictions.csv", index=False)

print("\nPredictions saved to 'eye_state_predictions.csv'")




# Load predictions from CSV
df = pd.read_csv("eye_state_predictions.csv")

# Count the occurrences of each predicted label
label_counts = df["Predicted Label"].value_counts()

# Print the result
print("\nPrediction Counts from CSV:")
print(label_counts)




# Evaluate model on the test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")




# Create lists of repeated test values
epochs_range = range(len(history.history['accuracy']))  # number of epochs
test_accuracy_list = [test_accuracy] * len(epochs_range)
test_loss_list = [test_loss] * len(epochs_range)




from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Get true labels
true_labels = test_generator.classes

# Step 2: Get predicted probabilities
pred_probs = model.predict(test_generator, verbose=1)

# Step 3: Convert probabilities to binary labels (assuming sigmoid output)
pred_labels = (pred_probs > 0.5).astype(int).reshape(-1)

# Step 4: Generate classification report
target_names = list(test_generator.class_indices.keys())  # e.g., ['Closed Eye', 'Open Eye']
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=target_names))




import matplotlib.pyplot as plt

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy', color='blue', marker='o')
plt.plot(epochs_range, test_accuracy_list, label='Test Accuracy', color='green', linestyle='--')
plt.title('Training vs Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history.history['loss'], label='Training Loss', color='red', marker='o')
plt.plot(epochs_range, test_loss_list, label='Test Loss', color='orange', linestyle='--')
plt.title('Training vs Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()




# Save as HDF5 file
model.save("/kaggle/working/eye_state_model_new2.h5")

model.save("/kaggle/working/eye_state_model_new2.keras")
