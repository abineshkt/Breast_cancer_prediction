import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
# Download the dataset from the UCI Machine Learning Repository
file_path = "wdbc.data"
# Define column names for the dataset
column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

# Load the dataset
data = pd.read_csv(file_path, header=None, names=column_names)

# Separate features and labels
X = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})
# Convert 'M' to 1 and 'B' to 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the feedforward neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model and store the history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Create a single figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot the training and validation accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Training and Validation Accuracy')
axes[0, 0].legend()

# Plot the training and validation loss
axes[0, 1].plot(history.history['loss'], label='Train Loss')
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Training and Validation Loss')
axes[0, 1].legend()

# Plot predicted probabilities vs. actual labels
y_pred = model.predict(X_test)
axes[1, 0].scatter(y_test, y_pred)
axes[1, 0].set_xlabel('Actual Labels')
axes[1, 0].set_ylabel('Predicted Probabilities')
axes[1, 0].set_title('Predicted Probabilities vs. Actual Labels')

# Hide the last empty subplot
axes[1, 1].axis('off')

# Adjust layout and show the combined plot
plt.tight_layout()
plt.show()
