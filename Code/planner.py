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
        height, width, num_actions = self.height, self.width, self.num_actions
        dim, gamma = self.feature_dim, self.gamma

        self.image = tf.placeholder(
            tf.float32, name="image", shape=[height, width])
        self.features = tf.placeholder(
            tf.float32, name="features", shape=[height, width, dim])
        
        self.weights_list = [
            self._weight_var(i) for i in range(dim)]
        self.weight_inputs = [
            self._weight_input(i) for i in range(dim)]
        self.weight_assignments = [
            w.assign(w_in) for w, w_in in zip(self.weights_list, self.weight_inputs)]

        self.wall_weight = tf.constant([-1000000.0], dtype=tf.float32)
        self.weights = tf.concat(self.weights_list + [self.wall_weight], axis=0)
        dim += 1

        image_batch = tf.expand_dims(tf.expand_dims(self.image, 0), 3)
        feature_batch = tf.expand_dims(self.features, 0)
        feature_batch = tf.concat([feature_batch, image_batch], axis=3)

        # To deal with walls, give a -1000000 reward whenever the agent takes an
        # action that would move it into a wall.
        north = Direction.get_number_from_direction(Direction.NORTH)
        south = Direction.get_number_from_direction(Direction.SOUTH)
        east = Direction.get_number_from_direction(Direction.EAST)
        west = Direction.get_number_from_direction(Direction.WEST)

        bellman_kernel_value = np.zeros([3, 3, 2*dim, num_actions*dim])
        for i in range(dim):
            # For every action, we get the features for the current state
            bellman_kernel_value[1,1,i,i::dim] = [1.0] * num_actions
            # If you move north, add the discounted features from the state
            # north of you. Similarly for the other actions.
            bellman_kernel_value[0,1,dim+i,north*dim+i] = gamma
            bellman_kernel_value[2,1,dim+i,south*dim+i] = gamma
            bellman_kernel_value[1,2,dim+i,east*dim+i] = gamma
            bellman_kernel_value[1,0,dim+i,west*dim+i] = gamma
        bellman_kernel = tf.constant(bellman_kernel_value, dtype=tf.float32)

        index_prefixes = tf.constant(
            [[[[0, i, j] for j in range(width)] for i in range(height)]],
            dtype=tf.int64)

        feature_expectations = tf.zeros([1, height, width, dim])
        for _ in range(self.num_iters):
            fv = tf.concat([feature_batch, feature_expectations], axis=3)
            # TODO: Fix Bellman kernel
            q_flattened_fes = self._conv2d(fv, bellman_kernel, "q_flat_fe")
            q_fes = tf.reshape(
                q_flattened_fes, [1, height, width, num_actions, dim])
            q_values = tf.tensordot(q_fes, self.weights, [[4], [0]])
            policy = tf.expand_dims(tf.argmax(q_values, axis=3), -1)
            indexes = tf.concat([index_prefixes, policy], axis=3)
            feature_expectations = tf.gather_nd(q_fes, indexes)

        self.feature_expectations = feature_expectations
        fv = tf.concat([feature_batch, feature_expectations], axis=3)
        q_flattened_fes = self._conv2d(fv, bellman_kernel, "q_flat_fe")
        q_fes = tf.reshape(
            q_flattened_fes, [1, height, width, num_actions, dim])
        self.q_values = tf.tensordot(q_fes, self.weights, [[4], [0]])

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
        (qvals,) = sess.run([self.q_values], feed_dict=fd)
        return qvals

    def _weight_var(self, i):
        return tf.Variable(tf.zeros([1], name="weight"+str(i)), trainable=False)

    def _weight_input(self, i):
        return tf.placeholder(tf.float32, name="weight_in"+str(i), shape=[1])

    def _conv2d(self, x, k, name=None, strides=(1,1,1,1),pad='SAME'):
        return tf.nn.conv2d(x, k, name=name, strides=strides, padding=pad)
