import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import math
import matplotlib.pyplot as plt


def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = int(num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(name="X", shape=[n_x, None], dtype=tf.float32)
    Y = tf.placeholder(name="Y", shape=[n_y, None], dtype=tf.float32)

    return X, Y


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [153, 18]
                        b1 : [153, 1]
                        W2 : [70, 153]
                        b2 : [70, 1]
                        W3 : [18, 70]
                        b3 : [18, 1]
                        W4 : [1, 18]
                        b4 : [1, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    W1 = tf.get_variable("W1", shape=[816, 18], initializer=tf.contrib.layers.xavier_initializer())
    # W1 = tf.Print(W1,[tf.shape(W1), W1], message="my W1-values:")

    b1 = tf.get_variable("b1", shape=[816, 1], initializer=tf.zeros_initializer())
    # b1 = tf.Print(b1,[tf.shape(b1), b1], message="my b1-values:")

    W2 = tf.get_variable("W2", [153, 816], initializer=tf.contrib.layers.xavier_initializer())
    # W2 = tf.Print(W2,[tf.shape(W2), W2], message="my W2-values:")

    b2 = tf.get_variable("b2", [153, 1], initializer=tf.zeros_initializer())
    # b2 = tf.Print(b2,[tf.shape(b2), b2], message="my b2-values:")

    W3 = tf.get_variable("W3", [18, 153], initializer=tf.contrib.layers.xavier_initializer())
    # W3 = tf.Print(W3,[tf.shape(W3), W3], message="my W3-values:")

    b3 = tf.get_variable("b3", [18, 1], initializer=tf.zeros_initializer())
    # b3 = tf.Print(b3,[tf.shape(b3), b3], message="my b3-values:")

    W4 = tf.get_variable("W4", [1, 18], initializer=tf.contrib.layers.xavier_initializer())
    # W3 = tf.Print(W3,[tf.shape(W3), W3], message="my W3-values:")

    b4 = tf.get_variable("b4", [1, 1], initializer=tf.zeros_initializer())
    # b3 = tf.Print(b3,[tf.shape(b3), b3], message="my b3-values:")

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    # X = tf.Print(X,[tf.shape(X), X[0]], message="my X[0]-values:")
    with tf.device('/cpu:0'):
        # W1 = tf.Print(W1,[tf.shape(W1), W1[0]], message="my W1[0]-values:")
        Z1 = tf.add(tf.matmul(W1, X), b1)                        # Z1 = np.dot(W1, X) + b1
        # Z1 = tf.Print(Z1,[tf.shape(Z1), Z1[0]], message="my Z1[0]-values:")
        A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
        Z2 = tf.add(tf.matmul(W2, A1), b2)                       # Z2 = np.dot(W2, a1) + b2
        A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
        Z3 = tf.add(tf.matmul(W3, A2), b3)                       # Z3 = np.dot(W3,Z2) + b3 it should be # Z3 = np.dot(W3,A2) + b3

        A3 = tf.nn.relu(Z3)
        Z4 = tf.add(tf.matmul(W4, A3), b4)

    return Z4


def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    # Y = tf.Print(Y, [Y], message="Y:")

    # logits = tf.transpose(Z3)
    # labels = tf.transpose(Y)
    # cost = tf.reduce_mean(tf.squared_difference(logits, labels))
    # cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y, Z3))))
    cost = tf.div(tf.abs(tf.subtract(Y, Z3)), Y)
    cost = tf.reduce_mean(cost)
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=400000, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Z3 = tf.Print(Z3,[tf.shape(Z3)], message="my Z-values:")
    # Y = tf.Print(Y,[tf.shape(Y)], message="my Y-values:")
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    check_op = tf.add_check_numerics_ops()
    # Start the session to compute the tensorflow graph
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        # saver.restore(sess, "./model.ckpt")

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                # print(minibatch_X.shape)
                _ , minibatch_cost, _ = sess.run([optimizer, cost, check_op], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                # print(minibatch_cost)
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                save_path = saver.save(sess, "./model.ckpt")
                print("Model saved in file: %s" % save_path)
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)


        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        # correct_prediction = tf.sqrt(tf.squared_difference(Z3, Y))

        # Calculate accuracy on the test set
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy = tf.reduce_mean(tf.div(tf.abs(tf.subtract(Y, Z3)), Y))
        print ("Train Error:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Error:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters
