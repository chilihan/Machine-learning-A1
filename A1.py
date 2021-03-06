""" ENGG*6500 A1
    Train a MLP on the StumbleUpon Evergreen dataset
"""

import time
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import multigrad

import pickle
import matplotlib.pyplot as plt

from numpy.random import uniform

from scipy.linalg import norm

# Global variables
dtype = 'float32'
eps = np.finfo(np.double).eps  # -- a small number


def load_evergreen(dtype=dtype):
    with open('evergreen.pkl') as f:
        train_set, val_set, test_set = pickle.load(f)
    train_X, train_y = train_set
    val_X, val_y = val_set

    return (train_X.astype(dtype), train_y.astype('int8'),
            val_X.astype(dtype), val_y.astype('int8'))


def logistic(z):
    """return logistic sigmoid of float or ndarray `z`"""
    return 1.0 / (1.0 + np.exp(-z))


def p_y_given_x(W, b, x):
    return logistic(np.dot(x, W) + b)


def logreg_prediction(W, b, x):
    return p_y_given_x(W, b, x) > 0.5


def cross_entropy(x, z):
    # note we add a small epsilon for numerical stability
    return -(x * np.log(z + eps) + (1 - x) * np.log(1 - z + eps))


def logreg_cost(W, b, x, y):
    z = p_y_given_x(W, b, x)
    l = cross_entropy(y, z).mean(axis=0)
    return l


def accuracy(y, y_pred):
    return 1.0 * np.sum(y == y_pred) / y.shape[0]


def mlp_cost(X, y, W_hid, b_hid, W_out, b_out):
    # forward pass
    # hidden activations
    act_hid = p_y_given_x(W_hid, b_hid, X)
    # output activation
    act_out = p_y_given_x(W_out, b_out, act_hid)
    return cross_entropy(y, act_out).mean(axis=0)


def mlp_predict(X, W_hid, b_hid, W_out, b_out):
    act_hid = p_y_given_x(W_hid, b_hid, X)
    act_out = p_y_given_x(W_out, b_out, act_hid)
    return act_out > 0.5


def initialize_model(n_inputs, n_hidden, dtype=dtype):
    W_hid = uniform(low=-4 * np.sqrt(6.0 / (n_inputs + n_hidden)),
                    high=4 * np.sqrt(6.0 / (n_inputs + n_hidden)),
                    size=(n_inputs, n_hidden)).astype(dtype)
    b_hid = np.zeros(n_hidden, dtype=dtype)

    # now allocate the logistic regression model at the top
    W_out = uniform(low=-4 * np.sqrt(6.0 / (n_inputs + n_hidden)),
                    high=4 * np.sqrt(6.0 / (n_inputs + n_hidden)),
                    size=(n_hidden,)).astype(dtype)
    b_out = np.array(0.0)

    return W_hid, b_hid, W_out, b_out


def my_grads(X, y, W_hid, b_hid, W_out, b_out):
    # print("------------------------------")
    #
    # print("X:",X,X.shape)
    # print ("this is x:", X[1],X[1].shape)
    d = X.shape[1] #shape =36
    samplesize = X.shape[0] #shape = 6500
    #samplesize = 20
    hidu = len(W_out) #shape = 100
    dE_dWout = np.zeros(hidu) #shape = (100,)
    dE_dWhid = np.zeros((d,hidu)) #shape  = (36,100)

    dE_dbout = 0 #()
    dE_dbhid = np.zeros((100,))
    # dE_dbhid = np.zeros()

    #
    # #test dE_dWout
    # x = X[0] #(36,)
    # print ("x:",x.shape)
    # ytrue = y[0] #()
    # print ("ytrue:",ytrue.shape)
    # y_hid = p_y_given_x(W_hid, b_hid, x) #(100,)
    # y_out = p_y_given_x(W_out, b_out, y_hid) #shape = ()
    #
    # print ("yhid:",y_hid,y_hid.shape)
    #
    #
    #
    # dE_dyout = np.divide(ytrue, y_out) #()
    # print ("dedout:", dE_dyout, dE_dyout.shape)
    #
    # dyout_dzout = np.multiply(np.subtract(1,y_out), y_out) # ()
    # dE_dzout = np.multiply(dE_dyout,dyout_dzout) #()
    # print ("dyout_dzout:", dyout_dzout,dyout_dzout.shape)
    # print ("dE_dzout:", dE_dzout,dE_dzout.shape)
    #
    # dE_dWoutchange = np.multiply(dE_dzout,y_hid) #(100,)
    # dE_dWout = np.add(dE_dWout, dE_dWoutchange) #(100,)+(100,)=(100,)
    # print ("dE_dWoutchange:", dE_dWoutchange, dE_dWoutchange.shape)
    # print ("dE_dWout:", dE_dWout,dE_dWout.shape)
    #
    # #test dE_dWhid
    # dzout_dyhid = W_out #(100,)
    # print ("dzout_dyhid:", dzout_dyhid,dzout_dyhid.shape)
    #
    # yhid_1 = 1-y_hid #(100,)
    # print ("1_yhid", yhid_1,yhid_1.shape)
    # dyhid_dzhid = y_hid*yhid_1  #(100,)
    # print ("dyhid_dzhid:",dyhid_dzhid,dyhid_dzhid.shape)
    #
    # #dzo/dyh * dyh/dzh = dzo/dzh
    # dzout_dzhid = dzout_dyhid * dyhid_dzhid
    # dE_dzhid = dE_dzout * np.array(dzout_dzhid)
    # print ("dzout_dzhid:",dzout_dzhid,dzout_dzhid.shape)
    # print ("dE_dzhid:",dE_dzhid,dE_dzhid.shape)
    #
    # dE_dzhid_mat = np.reshape(dE_dzhid,(1,len(dE_dzhid)))
    # print ("dE_dzhid_vec:",dE_dzhid_mat,dE_dzhid_mat.shape)
    # x_vec = np.reshape(x,(len(x),1))
    # print("x_vec:",x_vec,x_vec.shape)
    # dE_dWhidchange = np.dot(x_vec,dE_dzhid_mat) #(36, 100)
    # print("dE_dWhidchange:",dE_dWhidchange,dE_dWhidchange.shape)
    # dE_dWhid = np.add(dE_dWhid,dE_dWhidchange)
    # print ("dE_dWhid:",dE_dWhid,dE_dWhid.shape)
    # dE_dWhid = (1.0/(samplesize))*np.array(dE_dWhid)
    # dE_dWout = (1.0/(samplesize))*np.array(dE_dWout)



    # print ("dyh_dzh",dyhdzh, dyhdzh.shape)



    #loop
    for i in range(samplesize):
        # output to hidden layer
        x = X[i] # shape = (36,)
        dzhid_dWhid = X[i] #(36,)
        dzout_dyhid = W_out #
        ytrue = y[i] #shape = ()
        y_hid = p_y_given_x(W_hid, b_hid, x) #shape = (100,)
        y_out =p_y_given_x(W_out,b_out, y_hid) #shape = ()

        ## first calculate dE/dE_dWout
        part1 = np.divide(np.subtract(1,ytrue),np.add(np.subtract(1,y_out),eps))
        part2 = np.divide(ytrue,np.add(y_out,eps))
        dE_dyout = (part1-part2)



        #dE_dyout = np.subtract(np.divide(ytrue, y_out),np.divide(np.subtract(1,ytrue),np.subtract(1,y_out)))
        dyout_dzout = np.multiply(np.subtract(1,y_out), y_out)
        dE_dzout = np.multiply(dE_dyout,dyout_dzout)
        dE_dWoutchange = np.multiply(dE_dzout,y_hid) #(100,)
        dE_dWout = np.add(dE_dWout, dE_dWoutchange) #(100,)+(100,)=(100,)

        # output to hidden layer bias gradient
        dE_dboutchange = dE_dzout
        dE_dbout = np.add(dE_dbout, dE_dboutchange)
        #dE_dbout =(1.0/samplesize)*dE_dbout


        # hidden layer to input
        x = X[i] #shape = (36,)
        dzout_dyhid = W_out #(100,)
        yhid_1 = 1-y_hid  #(100,)
        dyhid_dzhid = y_hid*yhid_1  #(100,)
        dzout_dzhid = dzout_dyhid * dyhid_dzhid  #(100,); dzo/dyh * dyh/dzh =dzo/dzh
        dE_dzhid = dE_dzout * np.array(dzout_dzhid) #(100,)
        dE_dzhid_mat = np.reshape(dE_dzhid,(1,len(dE_dzhid))) #(1, 100)
        x_vec = np.reshape(x,(len(x),1)) #(36, 1)
        dE_dWhidchange = np.dot(x_vec,dE_dzhid_mat) #(36, 100)
        dE_dWhid = np.add(dE_dWhid,dE_dWhidchange) #(36, 100)

        # hidden layer to output layer bias gradient
        dE_dbhidchange = dE_dzhid
        dE_dbhid = np.add(dE_dbhid, dE_dbhidchange)


    dE_dWhid = (1.0/(samplesize))*np.array(dE_dWhid)
    dE_dWout = (1.0/(samplesize))*np.array(dE_dWout)
    dE_dbout =(1.0/samplesize)*dE_dbout
    dE_dbhid = (1.0/samplesize)*dE_dbhid
    return dE_dWhid,dE_dbhid , dE_dWout, dE_dbout




def check_gradients(cost_fn, grad_fn, X, y, W_hid, b_hid, W_out, b_out,
                    min_diff=0.0001):
    # generate autograd gradient function
    # note that we specify the arguments with respect to the gradient is taken

    auto_grad_fn = multigrad(cost_fn, [2, 3, 4, 5])
    weights = [W_hid, b_hid, W_out, b_out]

    grads_auto = auto_grad_fn(X, y, *weights)
    grads_manual = grad_fn(X, y, *weights)

    norms = [norm(auto_g - manual_g) for auto_g, manual_g in zip(grads_auto,
                                                                 grads_manual)]
    ret = True
    print "Checking gradients:"
    wrt = ('W_hid', 'b_hid', 'W_out', 'b_out')
    for name, n in zip(wrt, norms):
        if n < min_diff:
            msg = 'OK  '
        else:
            msg = 'FAIL'
            ret = False
        print '%s: %s - norm of gradient difference with autograd: %s' % \
              (name, msg, n)

    if ret:
        print "All gradients OK!"
    else:
        print "Some gradients failed."

    return ret


def train_model(train_X, train_y, test_X, test_y, W_hid, b_hid, W_out, b_out,
                learning_rate, epochs, mlp_grads=None, dtype=dtype):
    train_costs = np.zeros(epochs, dtype=dtype)
    test_costs = np.zeros(epochs, dtype=dtype)

    # Make a list of the weights
    weights = [W_hid, b_hid, W_out, b_out]

    if mlp_grads is None:
        mlp_grads = multigrad(mlp_cost, [2, 3, 4, 5])
    for epoch in xrange(epochs):
        print "Epoch", epoch

        train_cost = mlp_cost(train_X, train_y, *weights)
        test_cost = mlp_cost(test_X, test_y, *weights)
        train_costs[epoch] = train_cost
        test_costs[epoch] = test_cost

        grads = mlp_grads(train_X, train_y, *weights)

        # returns a CudaNDarray when running on GPU
        # creating a np.array copies the data back from the GPU
        if not isinstance(grads[0], np.ndarray):
            grads = [np.array(g) for g in grads]

        # update the weights
        for W, dW in zip(weights, grads):
            W -= learning_rate * dW

        print "Training set cost:", train_cost
        print "Test set cost:    ", test_cost

    return train_costs, test_costs


def run_training(n_hidden, learning_rate, epochs, data=None, model=None,
                 grad_fn=None, show_plot=True):
    t0 = time.time()
    train_X, train_y, test_X, test_y = data

    # initialize input layer parameters
    n_inputs = train_X.shape[1]  # -- aka D_0
    print "NUM input dimensions:", n_inputs

    if model is None:
        model = initialize_model(n_inputs, n_hidden)
    W_hid, b_hid, W_out, b_out = model

    if grad_fn is not None:
        check = check_gradients(mlp_cost, grad_fn, train_X, train_y,
                                W_hid, b_hid, W_out, b_out)
        if not check:
            print "Failed gradient check. Aborting training"
            return [], []

    print "Before training"
    print 'train accuracy: %6.4f' % \
          accuracy(train_y, mlp_predict(train_X, W_hid, b_hid, W_out, b_out))
    print 'train cross entropy: %6.4f' % \
          mlp_cost(train_X, train_y, W_hid, b_hid, W_out, b_out)
    print 'test accuracy: %6.4f' % \
          accuracy(test_y, mlp_predict(test_X, W_hid, b_hid, W_out, b_out))
    print 'test cross entropy: %6.4f' % mlp_cost(test_X, test_y, W_hid, b_hid,
                                                 W_out, b_out)

    train_costs, test_costs = train_model(train_X, train_y, test_X, test_y,
                                          W_hid, b_hid, W_out, b_out,
                                          learning_rate, epochs, grad_fn)

    print "After training, n_hidden: %s, learning_rate: %s" % (n_hidden,
                                                               learning_rate)
    print 'train accuracy: %6.4f' % \
          accuracy(train_y, mlp_predict(train_X, W_hid, b_hid, W_out, b_out))
    print 'train cross entropy: %6.4f' % \
          mlp_cost(train_X, train_y, W_hid, b_hid, W_out, b_out)
    print 'test accuracy: %6.4f' % \
          accuracy(test_y, mlp_predict(test_X, W_hid, b_hid, W_out, b_out))
    print 'test cross entropy: %6.4f' % mlp_cost(test_X, test_y, W_hid, b_hid,
                                                 W_out, b_out)

    print "training took: %s sec" % (time.time() - t0)

    if show_plot:
        plt.plot(train_costs, '-b', label="Training data")
        plt.plot(test_costs, '-r', label="Test data")
        plt.legend(loc='upper right')
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy")
        plt.show()

    return train_costs, test_costs

if __name__ == '__main__':

    # swap the lines below once you have implemented my_grads
    grad_fn = my_grads

    #grad_fn = None

    epochs = 250
    learning_rate = 0.01
    n_hidden = 100  # -- aka D_1

    data = load_evergreen()
    n_inputs = data[0].shape[1]  # -- aka D_0

    model = initialize_model(n_inputs, n_hidden)
    train_costs, test_costs = run_training(n_hidden, learning_rate, epochs,
                                           data, model, grad_fn,
                                           show_plot=True)
