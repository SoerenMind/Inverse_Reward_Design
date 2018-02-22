import numpy as np
import tensorflow as tf

from gridworld import Direction

class Model(object):
    def __init__(self, feature_dim, height, width, gamma, num_iters):
        self.feature_dim = feature_dim
        self.height = height
        self.width = width
        self.gamma = gamma
        self.num_iters = num_iters
        self.num_actions = 4
        self.build_tf_graph()

    def build_tf_graph(self):
        self.build_planner()
        # Initializing the variables
        self.initialize_op = tf.global_variables_initializer()

    def build_planner(self):
        self.image = tf.placeholder(
            tf.float32, name="image", shape=[self.height, self.width])
        self.features = tf.placeholder(
            tf.float32, name="features", shape=[self.height, self.width, self.feature_dim])
        
        self.weights_list = [
            self._weight_var(i) for i in range(self.feature_dim)]
        self.weight_inputs = [
            self._weight_input(i) for i in range(self.feature_dim)]
        self.weight_assignments = [
            w.assign(w_in) for w, w_in in zip(self.weights_list, self.weight_inputs)]
        self.weights = tf.reshape(
            tf.concat(self.weights_list, axis=0),
            [self.feature_dim, 1])

        image_batch = tf.reshape(self.image, [1, self.height, self.width, 1])
        feature_batch = tf.reshape(
            self.features, [1, self.height, self.width, self.feature_dim])

        # To deal with walls, give a -1000000 reward whenever the agent takes an
        # action that would move it into a wall. Change the gridworld so that
        # actions that move the agent into a wall are not available (rather than
        # causing the agent to stay in place, as it is now).
        # TODO: Think about episode termination and EXIT actions.
        # q = r + conv(w, v) + conv(w, self.image)
        # v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
        north = Direction.get_number_from_direction(Direction.NORTH)
        south = Direction.get_number_from_direction(Direction.SOUTH)
        east = Direction.get_number_from_direction(Direction.EAST)
        west = Direction.get_number_from_direction(Direction.WEST)

        bellman_kernel_value = np.zeros([3, 3, 2, self.num_actions])
        # No matter what action we take, we get the reward for the current state
        bellman_kernel_value[1,1,0,:] = [1] * self.num_actions
        # If you move north, add the discounted value from the state north of
        # you. Similarly for the other actions.
        bellman_kernel_value[0,1,1,north] = self.gamma
        bellman_kernel_value[2,1,1,south] = self.gamma
        bellman_kernel_value[1,2,1,east] = self.gamma
        bellman_kernel_value[1,0,1,west] = self.gamma
        bellman_kernel = tf.constant(bellman_kernel_value, dtype=tf.float32)

        r = tf.tensordot(feature_batch, self.weights, [[3], [0]])
        v = -1000000 * image_batch
        for _ in range(self.num_iters):
            rv = tf.concat([r, v], 3)
            q = self._conv2d(rv, bellman_kernel, name="q")
            v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")
            v -= 1000000 * image_batch

        self.q = self._conv2d(rv, bellman_kernel, name="q_final")

    def compute_qvals(self, sess, mdp):
        image, features, _ = mdp.convert_to_numpy_input()
        fd = {}
        for i in range(len(mdp.goals)):
            fd["weight_in" + str(i) + ":0"] = np.array([mdp.feature_weights[i]])
        sess.run(self.weight_assignments, feed_dict=fd)

        fd = {
            "image:0": image,
            "features:0": features
        }
        (qvals,) = sess.run([self.q], feed_dict=fd)
        return qvals

    def _weight_var(self, i):
        return tf.Variable(tf.zeros([1], name="weight"+str(i)), trainable=False)

    def _weight_input(self, i):
        return tf.placeholder(tf.float32, name="weight_in"+str(i), shape=[1])

    def _conv2d(self, x, k, name=None, strides=(1,1,1,1),pad='SAME'):
        return tf.nn.conv2d(x, k, name=name, strides=strides, padding=pad)
