import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop

# Set visible devices (important for multiple GPUs)
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0" refers to the first GPU. If you had multiple GPUs and wanted to use the second one, you would set it to "1" etc.

# Check for GPU availability and set memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        print("Using GPU:", tf.config.list_physical_devices('GPU')[0]) #prints the first physical GPU
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPUs found, running on CPU.")

# Initialize the input and output
X = np.array([[0.4, 0.3], [0.3, 0.4], [0.35, 0.45], [0.45, 0.45], [0.5, 0.45],[0.45, 0.5],[0.6, 0.5],[0.8, 0.9],[1, 1],[0, 0]])
y = np.array([0,0,0,0,1,1,1,1,1,0])

# Initialize the model
model = Sequential()
model.add(Dense(20, input_dim=2, activation='tanh'))  # Hidden layer with 20 nodes
model.add(Dense(20, activation='tanh'))  # Another hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model
opt = RMSprop(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=2000, batch_size=1)

# Verify if the model is running on GPU
print("Device being used:")
print(model.trainable_variables[0].device)