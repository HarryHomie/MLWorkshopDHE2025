from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,RMSprop
import numpy as np

X = np.array([
    [0.1, 0.2, 0.1],
    [0.3, 0.4, 0.05],
    [0.5, 0.4, 0.2],
    [0.5, 0.5, 0.2],
    [0.6, 0.4, 0.3],
    [0.65, 0.45, 0.2],
    [0.6, 0.4, 0.2],
    [0.5, 0.4, 0.2],
    [0.6, 0.3, 0.2],
    [0.7, 0.5, 0.4],
    [0.9, 0.8, 0.1],
    [1, 1, 1],
    [0, 0, 0],
    [0.9, 0.8, 0.3],
    [0.8, 0.7, 0.5],
])

y = np.array([
    0, 0, 0, 0, 1, 
    1, 1, 0, 0, 1, 
    0, 1, 0, 0, 1
])

model = Sequential()
model.add(Dense(20, input_dim=3, activation='tanh')) 
model.add(Dense(20, activation='tanh')) 
model.add(Dense(1, activation='sigmoid')) 


opt = RMSprop(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X, y, epochs=2000, batch_size=1)

