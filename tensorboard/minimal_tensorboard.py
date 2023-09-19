import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
    W = tf.Variable(tf.zeros([784, 10]), name='W')
    b = tf.Variable(tf.zeros([10]), name='b')
    y = tf.matmul(x, W) + b
    y = tf.nn.softmax(y)

with tf.Session(graph=graph) as sess:
    tf.train.write_graph(sess.graph_def, '.', 'graph.pb', as_text=False)

