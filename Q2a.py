import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.examples.tutorials


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Hyperparameters
def questiontwoa(n_input, n_sample):
    lr = 0.01
    max_epoch = 80

    trX = np.random.random((n_sample,n_input))
    #creat a zero vector trY, dimension is (number of sample)*1，be prepared for update it
    trY = np.zeros((n_sample,))

    #create a empty list of weights
    W = []
    #create a random b which is less than 3
    btr= 3*np.random.ranf()
    #create a noise
    noise = np.random.randn(*(trX[:,0]).shape) * 0.033


    #random generate w for each input and put them in empty W list
    for i in range(0,n_input):
        r=5*np.random.ranf()
        W.append(r)
        #update trY
        trY += r * trX[:,i]

    # create a y value which is approximately linear but with some random noise
    trY += btr + noise

    print (R)
    print btr

    #placeholder
    X = tf.placeholder("float",[None, n_input])
    Y = tf.placeholder("float",[None])


    def Linear_Regression_model(X, w, b):
        return tf.add(tf.matmul(X, w),b)

    # create a shared variable for the weight matrix
    w = tf.Variable(tf.zeros([n_input,1]), name="weights")
    b = tf.Variable(tf.zeros([1]), name="bias")
    y_model = Linear_Regression_model(X, w, b)

    cost = tf.square(Y - y_model) # use square error for cost function

    # construct an optimizer to minimize cost and fit line to my data
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)


    ##Launch the graph in a session
    with tf.Session() as sess:
        init=sess.run(tf.initialize_all_variables())

        for i in range(max_epoch):
            for (x,y) in zip(trX, trY):
                #print(x,y)
                sess.run(train_op, feed_dict={X: x[np.newaxis, ...], Y: y[np.newaxis, ...]})
                cur_cost = sess.run(cost, feed_dict = {X: trX, Y: trY})
                print ("epoch %d  cost: %f"% (i, np.sum(cur_cost)/len(trY)))
        #
        print ("Training Finished")
        #
        wpredict = sess.run(w)
        print(wpredict)
        bpredict = sess.run(b)
        print(bpredict)


questiontwoa(3,100)
        #

        #
        #
        #
