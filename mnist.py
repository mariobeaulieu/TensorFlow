#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#%matplotlib inline
import matplotlib.pyplot as plt

def display_image(image):
    plt.imshow(image.reshape(28,28),cmap='Greys', interpolation='nearest')

def plot_original_reconstructed(original, reconstructed):
    fig = plt.figure(figsize=(8,6))
    plt.subplot(221)
    display_image(original)
    plt.subplot(222)
    display_image(reconstructed)

mnist = input_data.read_data_sets('fashion-mnist')
display_image(mnist.train.images[np.random.randint(1000)])

print ('Starting Neural Network evaluation')

NUM_INPUTS  = NUM_OUTPUTS = 28*28
NUM_HIDDEN1 = NUM_HIDDEN3 = 400
NUM_HIDDEN2 = 200
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS])
dropout_rate = 0.3
X_drop = tf.layers.dropout(X, dropout_rate, training=True)
hidden_layer1 = tf.layers.dense(X_drop       , NUM_HIDDEN1, activation=tf.nn.relu)
hidden_layer2 = tf.layers.dense(hidden_layer1, NUM_HIDDEN2, activation=tf.nn.relu)
hidden_layer3 = tf.layers.dense(hidden_layer2, NUM_HIDDEN3, activation=tf.nn.relu)
outputs       = tf.layers.dense(hidden_layer3, NUM_OUTPUTS, activation=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
optimizer     = tf.train.AdamOptimizer(0.01)
training_op   = optimizer.minimize(reconstruction_loss)
init          = tf.global_variables_initializer()

n_epochs   = 10
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples
        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict = {X: X_batch})
        loss_train     = reconstruction_loss.eval(feed_dict = {X: X_batch})
        outputs_eval   = outputs.eval(feed_dict={X:X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)

n=50
plot_original_reconstructed(mnist.test.images[n], outputs_eval[n])

