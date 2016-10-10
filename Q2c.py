import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.examples.tutorials


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Hyperparameters
def questiontwoc(n_input, n_sample,n_output):
    lr = 0.01
    max_epoch = 150

    trX = np.random.random((n_sample,n_input))
    trY = np.zeros((n_sample,n_output))

    W = []
    btr= 3*np.random.ranf(n_output,)
    noise = np.random.randn(*(trX[:,0]).shape) * 0.033
    # create a y value which is approximately linear but with some random noise
    for p in range(n_output):
        W.append([])
        for i in range(0,n_input):
            r=5*np.random.ranf()
            W[-1].append(r)
            trY[:,p] += r * trX[:,i]
        trY[:,p] += btr[p] + noise

    W = np.array(W).T

    print W
    print btr


    # print trY
    X = tf.placeholder("float",[None, n_input])

    Y = tf.placeholder("float",[None,n_output])


    #
    def Linear_Regression_model(X, w, b):
        # linear regression can be simply defined by X*w + b.
        #return tf.add(tf.matmul(X, w), b)
        return tf.add(tf.matmul(X, w),b)

    # create a shared variable for the weight matrix
    w = tf.Variable(tf.zeros([n_input,n_output]), name="weights")
    b = tf.Variable(tf.zeros([n_output]), name="bias")
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
        #         print ("epoch %d  cost: %f"% (i, np.sum(cur_cost)/len(trY)))
        # #
        # print ("Training Finished")
        # #
        wpredict = sess.run(w)
        print(wpredict)  # It should be something around 2
        bpredict = sess.run(b)
        print(bpredict)

questiontwoc(6,50,4)
