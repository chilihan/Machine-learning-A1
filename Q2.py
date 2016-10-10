import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.examples.tutorials


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Hyperparameters
lr = 0.01
max_epoch = 50

# Create Data
# trX1 = np.linspace(-1, 1, 10)
# trX2=np.linspace(-1, 1, 10)
trX = np.random.random((101, 2))
batch_size = 3
# print trX


# create a y value which is approximately linear but with some random noise
trY = 5 * trX[:, 0] + 3 * trX[:, 1] + 4 + np.random.randn(*(trX[:, 0]).shape) * 0.33
n_input = trX.shape[1]
n_samples = trY.shape[0]
# print n_input
# print n_samples
# trY = []
# trX=[]
# for i in range(10):
#     y = 2 * trX1[i] +3*trX2[i]
#     trY.append(y)
#     trX.append([trX1, trX2])
# trY=np.transpose([trY])
print(trY)

# print trY
X = tf.placeholder("float", [None, n_input])

Y = tf.placeholder("float", [None])


#
def Linear_Regression_model(X, w, b):
    # linear regression can be simply defined by X*w + b.
    # return tf.add(tf.matmul(X, w), b)
    return tf.add(tf.matmul(X, w), b)

# create a shared variable for the weight matrix
w = tf.Variable(tf.zeros([2, 1]), name="weights")
b = tf.Variable(tf.zeros([1]), name="bias")
y_model = Linear_Regression_model(X, w, b)

cost = tf.square(Y - y_model)  # use square error for cost function

# construct an optimizer to minimize cost and fit line to my data
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


# Launch the graph in a session
with tf.Session() as sess:
    init = sess.run(tf.initialize_all_variables())

    for i in range(max_epoch):
        for (x, y) in zip(trX, trY):
            # print(x,y)
            sess.run(train_op, feed_dict={X: x[np.newaxis, ...], Y: y[np.newaxis, ...]})
            cur_cost = sess.run(cost, feed_dict={X: trX, Y: trY})
            print("epoch %d  cost: %f" % (i, np.sum(cur_cost) / len(trY)))
    #
    print("Training Finished")
    #
    wpredict = sess.run(w)
    print(wpredict)  # It should be something around 2
    bpredict = sess.run(b)
    print(bpredict)

    xp = trX[:, 0]
    xpp = trX[:, 1]


    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for l in range(101):
        ax.scatter(xp[l], xpp[l], trY[l])

    x_surf = np.arange(0, 1.2, 0.05)
    y_surf = np.arange(0, 1.2, 0.05)
    xm, xmm = np.meshgrid(x_surf, y_surf)
    z = xm * wpredict[0] + xmm * wpredict[1] + bpredict
    surf = ax.plot_surface(xm, xmm, z, rstride=3, cstride=3, alpha=0.2)

    plt.show()
