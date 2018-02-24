import numpy as np
import tensorflow as tf

from gridworld import Direction

class Model(object):
    def __init__(self, feature_dim, height, width, gamma, num_iters, query,
                 proxy_reward_space, true_reward_matrix, true_reward):
        self.feature_dim = feature_dim
        self.height = height
        self.width = width
        self.gamma = gamma
        self.num_iters = num_iters
        self.num_actions = 4
        self.query = query
        # List of possible settings for the query features
        # Eg. If we are querying features 1 and 4, and discretizing each into 3
        # buckets (-1, 0 and 1), the proxy reward space would be
        # [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.proxy_reward_space = proxy_reward_space
        self.true_reward_matrix = true_reward_matrix
        self.true_reward = true_reward
        self.build_tf_graph()

    def build_tf_graph(self):
        self.build_planner()
        # Initializing the variables
        self.initialize_op = tf.global_variables_initializer()

    def build_planner(self):
        self.name_to_op = {}
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
        self.name_to_op['feature_exps'] = feature_expectations

        fv = tf.concat([feature_batch, feature_expectations], axis=3)
        q_flattened_fes = self._conv2d(fv, bellman_kernel, "q_flat_fe")
        q_fes = tf.reshape(
            q_flattened_fes, [1, height, width, num_actions, dim])

        self.q_values = tf.tensordot(q_fes, self.weights, [[4], [0]])
        self.name_to_op['q_values'] = self.q_values

    def compute(self, outputs, sess, mdp, weight_inits, gradient_steps=0):
        """
        Takes gradient steps to set the non-query features to the values that
        best optimize the objective. After optimization, calculates the values
        specified in outputs and returns them.

        :param outputs: List of strings, each specifying a value to compute.
        :param sess: tf.Session() object.
        :param mdp: An instance of GridworldMdpWithDistanceFeatures.
        :param query: List of features (integers) to ask the user to set.
        :param weight_inits: Initialization for the non-query features.
        :param gradient_steps: Number of gradient steps to take.
        :return: List of the same length as parameter `outputs`.
        """
        image, features, _ = mdp.convert_to_numpy_input()
        fd = {}
        # for i in range(len(mdp.goals)):
        #     if i not in self.query:
        #         name = "weight_in" + str(i) + ":0"
        #         try: fd[name] = np.array([weight_inits[i]])
        #         except: fd[name] = np.array([0.])
        #     sess.run(self.weight_assignments, feed_dict=fd)

        # if weight_inits is not None:
        #     for i in range(len(mdp.goals)):
        #         if i not in self.query:
        #             name = "weight_in" + str(i) + ":0"
        #             fd[name] = np.array([weight_inits[i]])
        #     sess.run(self.weight_assignments, feed_dict=fd)
        #     # sess.run(self.weight_assignments, feed_dict=fd)

        if weight_inits is not None:
            fd = {}
            assign_ops = []
            # for i in range(len(mdp.goals)):
            for i in range(self.feature_dim):

                if i not in self.query:
                    # name = "weight_in" + str(i) + ":0"
                    # fd[name] = np.array([weight_inits[i]])
                    assign_ops.append(self.weight_assignments[i])
                    fd[self.weight_inputs[i]] = np.array([weight_inits[i]])
            sess.run(assign_ops, feed_dict=fd)

        # fd = {
        #     "image:0": image,
        #     "features:0": features
        # }
        fd = {
            self.image: image,
            self.features: features
        }
        def get_op(name):
            if name == 'entropy':
                return 0.0
            elif name == 'answer':
                idx = np.random.choice(len(self.proxy_reward_space))
                return self.proxy_reward_space[idx]
            elif name == 'true_posterior':
                K = len(self.proxy_reward_space)
                N = len(self.true_reward_matrix)
                result = np.random.rand(N, K)
                return (result / result.sum(0)).T
            elif name == 'optimal_weights':
                return np.zeros(self.feature_dim - len(self.query))
            elif name not in self.name_to_op:
                raise ValueError("Unknown op name: " + str(name))
            return sess.run([self.name_to_op[name]], feed_dict=fd)[0]

        return [get_op(name) for name in outputs]
        # output_ops = [get_op(name) for name in outputs]
        # return sess.run(output_ops, feed_dict=fd)

    def _weight_var(self, i):
        return tf.Variable(tf.zeros([1], name="weight"+str(i)), trainable=False)

    def _weight_input(self, i):
        return tf.placeholder(tf.float32, name="weight_in"+str(i), shape=[1])

    def _conv2d(self, x, k, name=None, strides=(1,1,1,1),pad='SAME'):
        return tf.nn.conv2d(x, k, name=name, strides=strides, padding=pad)
