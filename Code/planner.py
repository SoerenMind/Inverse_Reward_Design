import numpy as np
import tensorflow as tf

from gridworld import Direction

class Model(object):
    def __init__(self, feature_dim, height, width, gamma, num_iters, query,
                 proxy_reward_space, true_reward_matrix, true_reward, beta, objective):
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
        self.query_size = len(self.proxy_reward_space)
        self.true_reward_matrix = true_reward_matrix
        self.true_reward = true_reward
        self.beta = beta
        self.build_tf_graph(objective)

    def build_tf_graph(self, objective):
        self.build_weights()
        self.build_planner()
        self.build_map_to_posterior()
        self.build_map_to_objective(objective)
        # Initializing the variables
        self.initialize_op = tf.global_variables_initializer()

    def build_weights(self):
        query_size = len(self.query)
        num_fixed = self.feature_dim - query_size
        self.human_weights= tf.placeholder(tf.float32, shape=[query_size])
        self.weights_to_optimize = tf.Variable(tf.zeros([num_fixed]))
        self.weight_inputs = tf.placeholder(tf.float32, shape=[num_fixed])
        self.assign_op = self.weights_to_optimize.assign(self.weight_inputs)

        # TODO(rohinmshah): Order of weights is now wrong.
        self.weights = tf.concat([self.human_weights, self.weights_to_optimize], axis=0)

    def build_planner(self):
        raise NotImplemented('Should be implemented in subclass')

    def build_map_to_posterior(self):
        """
        Maps self.feature_exp (created by planner) to self.log_posterior.
        """
        'for testing purposes'
        self.feature_exp_test_input = tf.placeholder(
            tf.float32, name='feature_exp_test_input', shape=(None, self.feature_dim))
        self.true_reward_tensor = tf.constant(
            self.true_reward_matrix, dtype=tf.float32, name="true_reward_tensor", shape=self.true_reward_matrix.shape)

        avg_reward_matrix = tf.matmul(
            self.feature_exp_test_input, tf.transpose(self.true_reward_tensor), name='avg_reward_matrix')
        log_likelihoods_new = self.beta * avg_reward_matrix

        # Calculate posterior
        self.prior = tf.placeholder(tf.float32, name="prior", shape=(len(self.true_reward_matrix)))
        log_Z_w = tf.reduce_logsumexp(log_likelihoods_new, axis=0, name='log_Z_q')
        log_P_q_z = log_likelihoods_new - log_Z_w
        self.log_Z_q, max_a, max_b = logdot(log_P_q_z, tf.log(self.prior))
        self.log_posterior = log_P_q_z + tf.log(self.prior) - self.log_Z_q
        self.posterior = tf.exp(self.log_posterior, name="posterior")

        post_sum_to_1 = tf.reduce_sum(tf.exp(self.log_posterior), axis=1, name='post_sum_to_1')
        tf.assert_equal(post_sum_to_1, 1., name='posteriors_normalized')

        # Fill name to ops dict
        self.name_to_op['true_reward_tensor'] = self.true_reward_tensor
        self.name_to_op['prior'] = self.prior
        self.name_to_op['posterior'] = self.posterior

    def build_map_to_objective(self, objective):
        """

        :param objective:
        """
        if 'entropy' in objective:
            post_ent = - tf.reduce_sum(
                tf.multiply(tf.exp(self.log_posterior), self.log_posterior), axis=1, keep_dims=True, name='post_ent')
            self.exp_post_ent = tf.reduce_sum(
                tf.multiply(post_ent, tf.exp(self.log_Z_q)), axis=0, keep_dims=True, name='exp_post_entropy')
            self.name_to_op['entropy'] = self.exp_post_ent

        if 'variance' in objective:
            posterior = tf.exp(self.log_posterior, name="posterior")
            post_avg, post_var = tf.nn.moments(posterior, axes=[1], keep_dims=False)
            self.generalized_var = tf.matrix_determinant(post_var, name="generalized_variance")
            self.name_to_op['variance'] = self.generalized_var

        if 'regret' in objective:
            pass

        if 'avg_reward' in objective:
            pass

        if 'query_entropy' in objective:
            pass

    # @profile
    # TODO: Remove the feature_expectations_test_input argument
    def compute(self, outputs, sess, mdp, weight_inits, feature_expectations_test_input = None, gradient_steps=0):
        """
        Takes gradient steps to set the non-query features to the values that
        best optimize the objective. After optimization, calculates the values
        specified in outputs and returns them.

        :param outputs: List of strings, each specifying a value to compute.
        :param sess: tf.Session() object.
        :param mdp: The MDP whose true reward function we want to identify.
        :param query: List of features (integers) to ask the user to set.
        :param weight_inits: Initialization for the non-query features.
        :param gradient_steps: Number of gradient steps to take.
        :return: List of the same length as parameter `outputs`.
        """
        # if weight_inits is not None:
        #     fd = {}
        #     assign_ops = []
        #     for i in range(self.feature_dim):
        #         if i not in self.query:
        #             assign_ops.append(self.weight_assignments[i])
        #             fd[self.weight_inputs[i]] = np.array([weight_inits[i]])
        #     sess.run(assign_ops, feed_dict=fd)

        fd = self.create_mdp_feed_dict(mdp)
        if feature_expectationis_test_input is not None:
            fd[self.feature_expectations] = feature_expectations_test_input

        def get_op(name):
            K = len(self.proxy_reward_space)
            if name == 'entropy':
                return 0.0
            elif name == 'answer':
                return self.proxy_reward_space[np.random.choice(K)]
            elif name == 'true_posterior':
                N = len(self.true_reward_matrix)
                result = np.random.rand(N, K)
                return (result / result.sum(0)).T
            elif name == 'optimal_weights':
                return np.zeros(self.feature_dim - len(self.query))
            elif name == 'q_values':
                return np.random.rand(K, self.height, self.width, self.num_actions)
            elif name == 'feature_exps':
                return np.random.rand(K, self.height, self.width, self.feature_dim)
            elif name not in self.name_to_op:
                raise ValueError("Unknown op name: " + str(name))
            return sess.run([self.name_to_op[name]], feed_dict=fd)[0]

        return [get_op(name) for name in outputs]
        # output_ops = [get_op(name) for name in outputs]
        # return sess.run(output_ops, feed_dict=fd)

    def compute_from_reward_weights(self, outputs, sess, mdp, weights):
        fd = self.create_mdp_feed_dict(mdp)
        fd[self.weights] = weights
        # TODO(rohinmshah): Handle other outputs as well
        return sess.run([self.q_values], feed_dict=fd)

    def create_mdp_feed_dict(self, mdp):
        raise NotImplemented('Should be implemented in subclass')


class BanditsModel(Model):
    def build_planner(self):
        self.name_to_op = {}

        self.features = tf.placeholder(
            tf.float32, name="features", shape=[None, self.feature_dim])

        intermediate_tensor = tf.multiply(self.features, self.weights)
        self.reward_per_state = tf.reduce_sum(intermediate_tensor, axis=1, keep_dims=False, name="rewards_per_state")

        # (This is for one particular setting of the weights)
        self.state_probs = tf.nn.softmax(self.reward_per_state, axis=-1, name="state_probs")
        self.feature_expectations = tf.reduce_sum(
            tf.multiply(self.features, self.state_probs), axis=-2 ,name="feature_expectations")

        self.name_to_op['features'] = self.features
        self.name_to_op['weights_unsorted'] = self.weights
        self.name_to_op['state_probs'] = self.state_probs
        self.name_to_op['reward_per_state'] = self.reward_per_state
        self.name_to_op['feature_exps'] = self.feature_expectations


class GridworldModel(Model):
    def build_planner(self):
        self.name_to_op = {}
        height, width, dim = self.height, self.width, self.feature_dim
        num_actions = self.num_actions

        self.image = tf.placeholder(
            tf.float32, name="image", shape=[height, width])
        self.features = tf.placeholder(
            tf.float32, name="features", shape=[height, width, dim])

        features_wall = tf.concat(
            [self.features, tf.expand_dims(self.image, -1)], axis=-1)
        weights_wall = tf.concat([self.weights, [-1000000]], axis=-1)
        dim += 1

        index_prefixes = tf.constant(
            [[[[i, j, k]
               for k in range(dim)]
              for j in range(width)]
             for i in range(height)],
            dtype=tf.int64)

        feature_expectations = tf.zeros([height, width, dim])
        for _ in range(self.num_iters):
            q_fes = self.bellman_update(feature_expectations, features_wall)
            q_values = tf.tensordot(q_fes, weights_wall, [[-2], [0]])
            policy = tf.expand_dims(tf.argmax(q_values, axis=-1), -1)
            repeated_policy = tf.stack([policy] * dim, axis=-2)
            indexes = tf.concat([index_prefixes, repeated_policy], axis=-1)
            feature_expectations = tf.gather_nd(q_fes, indexes)

        self.feature_expectations = feature_expectations[:,:,:-1]
        self.name_to_op['feature_exps'] = self.feature_expectations

        q_fes = self.bellman_update(feature_expectations, features_wall)
        q_values = tf.tensordot(q_fes, weights_wall, [[-2], [0]])
        self.q_values = q_values
        self.name_to_op['q_values'] = q_values

    def bellman_update(self, fes, features):
        height, width, dim = self.height, self.width, self.feature_dim + 1
        gamma = self.gamma
        extra_row = tf.zeros((1, width, dim))
        extra_col = tf.zeros((height, 1, dim))

        north_lookahead = tf.concat([extra_row, fes[:-1]], axis=0)
        north_fes = features + gamma * north_lookahead
        south_lookahead = tf.concat([fes[1:], extra_row], axis=0)
        south_fes = features + gamma * south_lookahead
        east_lookahead = tf.concat([fes[:,1:], extra_col], axis=1)
        east_fes = features + gamma * east_lookahead
        west_lookahead = tf.concat([extra_col, fes[:,:-1]], axis=1)
        west_fes = features + gamma * west_lookahead
        return tf.stack([north_fes, south_fes, east_fes, west_fes], axis=-1)

    def create_mdp_feed_dict(self, mdp):
        image, features, _ = mdp.convert_to_numpy_input()
        return {
            self.image: image,
            self.features: features
        }


class GridworldModelUsingConvolutions(GridworldModel):
    def build_planner(self):
        self.name_to_op = {}
        height, width, dim = self.height, self.width, self.feature_dim
        num_actions, gamma = self.num_actions, self.gamma

        self.image = tf.placeholder(
            tf.float32, name="image", shape=[height, width])
        self.features = tf.placeholder(
            tf.float32, name="features", shape=[height, width, dim])

        image_batch = tf.expand_dims(tf.expand_dims(self.image, 0), 3)
        feature_batch = tf.expand_dims(self.features, 0)
        feature_batch = tf.concat([feature_batch, image_batch], axis=3)
        wall_weight = tf.constant([-1000000.0], dtype=tf.float32)
        new_weights = tf.concat([self.weights, wall_weight], axis=0)
        dim += 1

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
            q_flattened_fes = self._conv2d(fv, bellman_kernel, "q_flat_fe")
            q_fes = tf.reshape(
                q_flattened_fes, [1, height, width, num_actions, dim])
            q_values = tf.tensordot(q_fes, new_weights, [[4], [0]])
            policy = tf.expand_dims(tf.argmax(q_values, axis=3), -1)
            indexes = tf.concat([index_prefixes, policy], axis=3)
            feature_expectations = tf.gather_nd(q_fes, indexes)

        self.feature_expectations = feature_expectations
        self.name_to_op['feature_exps'] = feature_expectations

        fv = tf.concat([feature_batch, feature_expectations], axis=3)
        q_flattened_fes = self._conv2d(fv, bellman_kernel, "q_flat_fe")
        q_fes = tf.reshape(
            q_flattened_fes, [1, height, width, num_actions, dim])

        self.q_values = tf.tensordot(q_fes, new_weights, [[4], [0]])
        self.name_to_op['q_values'] = self.q_values

    def _conv2d(self, x, k, name=None, strides=(1,1,1,1),pad='SAME'):
        return tf.nn.conv2d(x, k, name=name, strides=strides, padding=pad)


def logdot(a,b):
    max_a, max_b = tf.reduce_max(a), tf.reduce_max(b)   # TODO: make sure broadcasting is right. Don't let max_a be max over whole matrix.
    exp_a, exp_b = a - max_a, b - max_b
    exp_a = tf.exp(exp_a)
    exp_b = tf.exp(exp_b)
    # c = tf.tensordot(exp_a, exp_b, axes=1)
    c = tf.reduce_sum(tf.multiply(exp_a,exp_b), axis=1, keep_dims=True)
    c = tf.log(c) + max_a + max_b
    return c, max_a, max_b
