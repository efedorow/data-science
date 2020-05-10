import tensorflow as tf
import logging
import numpy as np

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#these are the input values in kilograms
meters_in = np.array([1, 5, 8, 15, 23, 34, 49, 69, 82, 96])

#these are the output values in pounds determined via the conversion formula
pounds_out = np.array([2.205, 11.023, 17.637, 33.069, 50.706, 74.957, 108.027, 152.119, 180.779, 211.644])

#for printing them out to see the data better
for ivar, cvar in enumerate(meters_in):
    print("{} kilograms = {} pounds".format(ivar, pounds_out[ivar]))

#the model used, it has only one layer since it is linear
model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
        ])

#compile the model with a loss function in terms of mean squared error
#and using the optimize function Adam for a learning rate of 0.05
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.05))

#this gives a plot of the loss magnitude over time
#it quickly goes to around 0
history = model.fit(meters_in, pounds_out, epochs=600, verbose=False)
import matplotlib.pyplot as plt
plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

#this determines the error for some random inputs
y_true = np.random.randint(0, 2, size=(2,3))
y_pred = np.random.random(size=(2, 3))
loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
assert loss.shape == (2,)
assert np.array_equal(loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))

tf.keras.losses.MAE(
    y_true, y_pred)

#to find how accurate it is print the layer weight (real conversion is K = 2.20462*P)
print("These are the layer vaiables:{}".format(model.get_weights())) #gave value of 2.17817
