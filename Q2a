import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.examples.tutorials


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# Hyperparameters
def questiontwo(n_input, n_sample):
    lr = 0.1
    max_epoch = 80

    trX = np.random.random((n_sample,n_input))
    trY = np.zeros((n_sample,))

    R = []
    btr= 3*np.random.ranf()
    noise = np.random.randn(*(trX[:,0]).shape) * 0.033
    # create a y value which is approximately linear but with some random noise

    for i in range(0,n_input):
        # r = np.random.random_integers(4)
        r=5*np.random.ranf()
        R.append(r)
        trY += r * trX[:,i]

    trY += btr + noise

    # trY = 5 * trX[:,0] +3*trX[:,1]+4+np.random.randn(*(trX[:,0]).shape) * 0.33

    # n_input = trX.shape[1]
    # n_samples = trY.shape[0]
    # prin[trY])
    print (R)
    print btr

    # print trY
    X = tf.placeholder("float",[None, n_input])

    Y = tf.placeholder("float",[None])


    #
    def Linear_Regression_model(X, w, b):
        # linear regression can be simply defined by X*w + b.
        #return tf.add(tf.matmul(X, w), b)
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
        print(wpredict)  # It should be something around 2
        bpredict = sess.run(b)
        print(bpredict)


questiontwo(3,100)
        #

        #
        #
        #
