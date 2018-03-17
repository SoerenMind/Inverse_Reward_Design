import numpy as np
import tensorflow as tf
from itertools import product

from gridworld import Direction

class Model(object):
    def __init__(self, feature_dim, gamma, query_size, discretization_size,
                 true_reward_space_size, num_unknown, beta, beta_planner,
                 objective, lr, discrete, optimize):
        self.initialized = False
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.query_size = query_size
        self.true_reward_space_size = true_reward_space_size
        self.beta = beta
        self.beta_planner = beta_planner
        self.lr = lr
        self.discrete = discrete
        self.optimize = optimize
        if discrete:
            self.K = query_size
            if optimize:
                self.num_unknown = num_unknown
        else:
            assert query_size <= 5
            num_posneg_vals = (discretization_size // 2)
            const = 9 // num_posneg_vals
            f_range = range(-num_posneg_vals * const, 10, const)
            print 'Using', f_range, 'to discretize the feature'
            assert len(f_range) == discretization_size
            # proxy_space = np.random.randint(-4,3,size=[30 * query_size, query_size])
            self.proxy_reward_space = list(product(f_range, repeat=query_size))
            self.K = len(self.proxy_reward_space)
        self.build_tf_graph(objective)

    def initialize(self, sess):
        if not self.initialized:
            self.initialized = True
            sess.run(self.initialize_op)

    def build_tf_graph(self, objective):
        self.name_to_op = {}
        self.build_weights()
        self.build_planner()
        self.build_map_to_posterior()
        self.build_map_to_objective(objective)
        # Initializing the variables
        self.initialize_op = tf.global_variables_initializer()

    def build_weights(self):
        if self.discrete and self.optimize:
            self.build_discrete_weights_for_optimization()
        elif self.discrete:
            self.build_discrete_weights()
        else:
            self.build_continuous_weights()

    def build_discrete_weights_for_optimization(self):
        K, N = self.K, self.num_unknown
        dim = self.feature_dim
        self.weights_to_train = tf.Variable(
            tf.zeros([N, dim]), name="weights_to_train")
        self.weight_inputs = tf.placeholder(
            tf.float32, shape=[N, dim], name="weight_inputs")
        self.assign_op = self.weights_to_train.assign(self.weight_inputs)

        if N < K:
            self.known_weights = tf.placeholder(
                tf.float32, shape=[K - N, dim], name="known_weights")
            self.weights = tf.concat(
                [self.known_weights, self.weights_to_train], axis=0, name="weights")
        else:
            self.weights = self.weights_to_train

        self.name_to_op['weights'] = self.weights
        self.name_to_op['weights_to_train'] = self.weights_to_train

    def build_discrete_weights(self):
        self.weights = tf.placeholder(
            tf.float32, shape=[self.K, self.feature_dim], name="weights")

    def build_continuous_weights(self):
        query_size, dim, K = self.query_size, self.feature_dim, self.K
        num_fixed = dim - query_size
        self.query_weights= tf.constant(
            self.proxy_reward_space, dtype=tf.float32, name="query_weights")

        if self.optimize:
            weight_inits = tf.random_normal([num_fixed], stddev=4)
            self.weights_to_train = tf.Variable(
                weight_inits, name="weights_to_train")
            self.weight_inputs = tf.placeholder(
                tf.float32, shape=[num_fixed], name="weight_inputs")
            self.assign_op = self.weights_to_train.assign(self.weight_inputs)
            self.fixed_weights = self.weights_to_train
            self.name_to_op['weights_to_train'] = self.weights_to_train
            self.name_to_op['weights_to_train[:3]'] = self.weights_to_train[:3]
        else:
            self.fixed_weights = tf.constant(
                np.zeros([num_fixed], dtype=np.float32))

        # Let's say query is [1, 3] and there are 6 features.
        # query_weights = [10, 11] and weight_inputs = [12, 13, 14, 15].
        # Then we want self.weights to be [12, 10, 13, 11, 14, 15].
        # Concatenate to get [10, 11, 12, 13, 14, 15]
        repeated_weights = tf.stack([self.fixed_weights] * K, axis=0)
        unordered_weights = tf.concat(
            [self.query_weights, repeated_weights], axis=1)
        # Then permute using gather to get the desired result.
        # The permutation can be computed from the query [1, 3] using
        # get_permutation_from_query.
        self.permutation = tf.placeholder(tf.int32, shape=[dim])
        self.weights = tf.gather(unordered_weights, self.permutation, axis=-1)

        self.name_to_op['weights'] = self.weights
        self.name_to_op['query_weights'] = self.query_weights


    def build_planner(self):
        raise NotImplemented('Should be implemented in subclass')

    def build_map_to_posterior(self):
        """
        Maps self.feature_exp (created by planner) to self.log_posterior.
        """
        # Get log likelihoods for true reward matrix
        true_reward_space_size = self.true_reward_space_size
        dim = self.feature_dim
        self.true_reward_matrix = tf.placeholder(
            tf.float32, [true_reward_space_size, dim], name="true_reward_matrix")
        self.log_true_reward_matrix = tf.log(self.true_reward_matrix, name='log_true_reward_matrix')

        # TODO: Inefficient to recompute this matrix on every forward pass.
        # We can cache it and feed in true reward indeces instead of true_reward_matrix. The matrix multiplication has
        # size_proxy x size_true x feature_dim complexity. The other calculations in this map have a factor feature_dim
        # less. However, storing this matrix takes size_proxy / feature_dim more memory. That's good for large feature_dim.
        self.avg_reward_matrix = tf.tensordot(
            self.feature_expectations, self.true_reward_matrix, axes=[-1, -1], name='avg_reward_matrix')

        log_likelihoods_new = self.beta * self.avg_reward_matrix


        # Calculate posterior
        # self.prior = tf.placeholder(tf.float32, name="prior", shape=(true_reward_space_size))
        self.log_prior = tf.placeholder(tf.float32, name="log_prior", shape=(true_reward_space_size))
        log_Z_w = tf.reduce_logsumexp(log_likelihoods_new, axis=0, name='log_Z_w')
        log_P_q_z = log_likelihoods_new - log_Z_w
        # self.log_Z_q, max_a, max_b = logdot(log_P_q_z, tf.log(self.prior))
        self.log_Z_q = tf.reduce_logsumexp(log_P_q_z + self.log_prior, axis=1, name='log_Z_q', keep_dims=True)
        # self.log_posterior = log_P_q_z + tf.log(self.prior) - self.log_Z_q
        self.log_posterior = log_P_q_z + self.log_prior - self.log_Z_q
        self.posterior = tf.exp(self.log_posterior, name="posterior")

        self.post_sum_to_1 = tf.reduce_sum(tf.exp(self.log_posterior), axis=1, name='post_sum_to_1')


        # Get log likelihoods for actual true reward
        self.true_reward = tf.placeholder(
            tf.float32, shape=[dim], name="true_reward")
        self.true_reward_tensor = tf.expand_dims(
            self.true_reward, axis=0, name="true_reward_tensor")
        self.avg_true_rewards = tf.tensordot(
            self.feature_expectations, tf.transpose(self.true_reward_tensor), axes=[-1, -2], name='avg_true_rewards')
        true_log_likelihoods = self.beta * self.avg_true_rewards
        log_true_Z_w = tf.reduce_logsumexp(true_log_likelihoods, axis=0, name='log_true_Z_w')
        log_answer_probs = true_log_likelihoods - log_true_Z_w

        self.name_to_op['true_reward_tensor'] = self.true_reward_tensor
        self.name_to_op['avg_true_rewards'] = self.avg_true_rewards
        self.name_to_op['true_log_likelihoods'] = true_log_likelihoods
        self.name_to_op['log_answer_probs'] = log_answer_probs
        self.name_to_op['log_true_Z_w'] = log_true_Z_w
        self.name_to_op['log_answer_probs'] = log_answer_probs

        # # Sample answer
        log_answer_probs = tf.reshape(log_answer_probs, shape=[1,-1])
        sample = tf.multinomial(log_answer_probs, num_samples=1)
        sample = sample[0][0]
        self.true_log_posterior = self.log_posterior[sample]
        self.true_posterior = self.posterior[sample]

        self.name_to_op['sample'] = sample
        self.name_to_op['true_posterior'] = self.true_posterior
        self.name_to_op['true_log_posterior'] = self.true_log_posterior
        self.name_to_op['probs'] = tf.exp(log_answer_probs)

        # Get true posterior entropy
        scaled_log_posterior = self.true_log_posterior - 0.0001
        interm_tensor = scaled_log_posterior + tf.log(- scaled_log_posterior)
        self.true_ent = tf.exp(tf.reduce_logsumexp(
            interm_tensor, axis=0, name="true_entropy", keep_dims=True))
        self.name_to_op['true_entropy'] = self.true_ent

        # Get true posterior_avg
        ## Not in log space
        self.post_weighted_true_reward_matrix = tf.multiply(self.true_posterior, tf.transpose(self.true_reward_matrix))
        self.post_avg = tf.reduce_sum(self.post_weighted_true_reward_matrix, axis=1, name='post_avg', keep_dims=False)

        ## In log space (necessary?)
        # log_true_posterior_times_true_reward = self.true_log_posterior + tf.transpose(self.log_true_reward_matrix) # TODO: log true posteriors are log of negative
        # self.log_post_avg = tf.reduce_logsumexp(log_true_posterior_times_true_reward, axis=1, keep_dims=False)
        # self.name_to_op['log_post_avg'] = self.log_post_avg
        # self.post_avg = tf.exp(self.log_post_avg, name='post_avg')


        # Fill name to ops dict
        self.name_to_op['post_avg'] = self.post_avg
        self.name_to_op['avg_reward_matrix'] = self.avg_reward_matrix
        self.name_to_op['true_reward_matrix'] = self.true_reward_matrix
        # self.name_to_op['prior'] = self.prior
        self.name_to_op['log_prior'] = self.log_prior
        self.name_to_op['posterior'] = self.posterior
        self.name_to_op['log_posterior'] = self.log_posterior
        self.name_to_op['post_sum_to_1'] = self.post_sum_to_1


    def build_map_to_objective(self, objective):
        """
        :param objective: string that specifies the objective function
        """
        if 'entropy' == objective:
            # # Calculate exp entropy without log space trick
            # post_ent = - tf.reduce_sum(
            #     tf.multiply(tf.exp(self.log_posterior), self.log_posterior), axis=1, keep_dims=True, name='post_ent')
            # self.exp_post_ent = tf.reduce_sum(
            #     tf.multiply(post_ent, tf.exp(self.log_Z_q)), axis=0, keep_dims=True, name='exp_post_entropy')
            # self.name_to_op['entropy'] = self.exp_post_ent


            # Calculate entropy as exp logsumexp (log p + log (-log p))
            scaled_log_posterior = self.log_posterior - 0.0001
            interm_tensor = scaled_log_posterior + tf.log(- scaled_log_posterior)
            self.log_post_ent_new = tf.reduce_logsumexp(
                interm_tensor, axis=1, name="entropy_per_answer", keep_dims=True)
            self.post_ent_new = tf.exp(self.log_post_ent_new)
            self.name_to_op['entropy_per_answer'] = self.post_ent_new
            self.log_exp_post_ent = tf.reduce_logsumexp(
                self.log_post_ent_new + self.log_Z_q, axis=0, keep_dims=True, name='entropy')
            self.exp_post_ent = tf.exp(self.log_exp_post_ent)
            self.name_to_op['entropy'] = self.exp_post_ent

            self.objective = self.log_exp_post_ent

        if 'total_variation' == objective:

            self.post_averages, self.post_var = tf.nn.weighted_moments(
                self.true_reward_matrix, [1, 1], tf.stack([self.posterior] * self.feature_dim, axis=2),
                name="moments", keep_dims=False)
            self.name_to_op['post_var'] = self.post_var

            self.total_variations = tf.reduce_sum(self.post_var, axis=-1, keep_dims=False)
            self.name_to_op['total_variations'] = self.total_variations
            self.total_variations, self.log_Z_q = tf.reshape(self.total_variations, [-1]), tf.reshape(self.log_Z_q,[-1])
            self.total_variation = tf.tensordot(
                self.total_variations, tf.exp(self.log_Z_q), axes=[-1,-1] ,name='total_var')
            self.total_variation = tf.reshape(self.total_variation, shape=[1,1,-1])
            self.name_to_op['total_variation'] = self.total_variation


            self.objective = self.total_variation

        # Set up optimizer
        if self.optimize:
            # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) # Make sure the momentum is reset for each model call
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            gradients, vs = zip(*optimizer.compute_gradients(self.objective))
            # self.gradient_norm = tf.norm(tf.stack(gradients, axis=0))
            self.train_op = optimizer.apply_gradients(zip(gradients, vs))
            self.name_to_op['gradients'] = gradients
            # self.name_to_op['gradients[:4]'] = gradients[0][:4]
            # self.name_to_op['gradient_norm'] = self.gradient_norm
            self.name_to_op['minimize'] = self.train_op


    def compute(self, outputs, sess, mdp, query=None, log_prior=None, weight_inits=None, feature_expectations_input=None,
                gradient_steps=0, gradient_logging_outputs=[], true_reward=None, true_reward_matrix=None):
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
        if weight_inits is not None:
            fd = {self.weight_inputs: weight_inits}
            sess.run([self.assign_op], feed_dict=fd)

        fd = {}
        self.update_feed_dict_with_mdp(mdp, fd)
        if feature_expectations_input is not None:
            fd[self.feature_expectations] = feature_expectations_input

        if log_prior is not None:
            fd[self.log_prior] = log_prior

        if query:
            if self.discrete and self.optimize:
                fd[self.known_weights] = query
            elif self.discrete:
                fd[self.weights] = query
            else:
                fd[self.permutation] = self.get_permutation_from_query(query)

        if true_reward is not None:
            fd[self.true_reward] = true_reward
        if true_reward_matrix is not None:
            fd[self.true_reward_matrix] = true_reward_matrix

        def get_op(name):
            if name not in self.name_to_op:
                raise ValueError("Unknown op name: " + str(name))
            return self.name_to_op[name]

        if gradient_steps > 0:
            ops = [get_op(name) for name in gradient_logging_outputs]
            other_ops = [self.train_op]
            for step in range(gradient_steps):
                results = sess.run(ops + other_ops, feed_dict=fd)
                if ops and step % 9 == 0:
                    print 'Gradient step {0}: {1}'.format(step, results[:-1])

        return sess.run([get_op(name) for name in outputs], feed_dict=fd)


    def update_feed_dict_with_mdp(self, mdp, fd):
        raise NotImplemented('Should be implemented in subclass')

    def get_permutation_from_query(self, query):
        dim = self.feature_dim
        # Running example: query = [1, 3], and we want indexes that will permute
        # weights [10, 11, 12, 13, 14, 15] to [12, 10, 13, 11, 14, 15].
        # Compute the feature numbers for unordered_weights.
        # This can be thought of as an unordered_weight -> feature map
        # In our example, this would be [1, 3, 0, 2, 4, 5]
        feature_order = query[:]
        for i in range(dim):
            if i not in feature_order:
                feature_order.append(i)
        # Invert the previous map to get the feature -> unordered_weight map.
        # This gives us [2, 0, 3, 1, 4, 5]
        indexes = [None] * dim
        for i in range(dim):
            indexes[feature_order[i]] = i
        return indexes


class BanditsModel(Model):

    def build_planner(self):
        self.features = tf.placeholder(
            tf.float32, name="features", shape=[None, self.feature_dim])
        self.name_to_op['features'] = self.features

        # Calculate state probabilities
        # TODO(soerenmind): Calculate q-values instead of rewards
        weights_expand = tf.expand_dims(self.weights,axis=0)
        weights_reshaped = tf.transpose(weights_expand, perm=[0,2,1])
        intermediate_tensor = tf.multiply(tf.stack([self.features]*self.K,axis=2), weights_reshaped)
        self.reward_per_state = tf.reduce_sum(intermediate_tensor, axis=1, keep_dims=False, name="rewards_per_state")
        self.name_to_op['reward_per_state'] = self.reward_per_state
        self.name_to_op['q_values'] = self.reward_per_state
        self.state_probs = tf.nn.softmax(self.beta_planner * self.reward_per_state, dim=0, name="state_probs")
        self.name_to_op['state_probs'] = self.state_probs
        self.name_to_op['state_probs_cut'] = self.state_probs[:5]


        # Calculate feature expectations
        probs_stack = tf.stack([self.state_probs] * self.feature_dim, axis=1)
        features_stack = tf.multiply(tf.stack([self.features] * self.K, axis=2), probs_stack, name='multi')
        self.feature_expectations = tf.reduce_sum(features_stack, axis=0, keep_dims=False, name="feature_exps")
        self.feature_expectations = tf.transpose(self.feature_expectations)
        self.name_to_op['feature_exps'] = self.feature_expectations


    def update_feed_dict_with_mdp(self, mdp, fd):
        fd[self.features] = mdp.convert_to_numpy_input()


class GridworldModel(Model):
    def __init__(self, feature_dim, gamma, query_size, discretization_const,
                 true_reward_space_size, num_unknown, beta, beta_planner,
                 objective, lr, discrete, optimize, height, width, num_iters):
        self.height = height
        self.width = width
        self.num_iters = num_iters
        self.num_actions = 4
        super(GridworldModel, self).__init__(
            feature_dim, gamma, query_size, discretization_const,
            true_reward_space_size, num_unknown, beta, beta_planner,
            objective, lr, discrete, optimize)

    def build_planner(self):
        height, width, dim = self.height, self.width, self.feature_dim
        num_actions, K = self.num_actions, self.K

        self.image = tf.placeholder(
            tf.float32, name="image", shape=[height, width])
        self.features = tf.placeholder(
            tf.float32, name="features", shape=[height, width, dim])
        self.start_x = tf.placeholder(tf.int32, name="start_x", shape=[])
        self.start_y = tf.placeholder(tf.int32, name="start_y", shape=[])

        features_wall = tf.concat(
            [self.features, tf.expand_dims(self.image, -1)], axis=-1)
        features_wall = tf.stack([features_wall] * K, axis=0)
        wall_constant = [[-1000000.0] for _ in range(K)]
        weights_wall = tf.concat([self.weights, wall_constant], axis=-1)
        # Change from K by dim to K by 1 by dim
        weights_wall = tf.expand_dims(weights_wall, axis=1)
        # Change to K by height by width by 1 by dim
        weights_wall = tf.stack([weights_wall] * width, axis=1)
        weights_wall = tf.stack([weights_wall] * height, axis=1)
        dim += 1

        feature_expectations = tf.zeros([K, height, width, dim])
        for i in range(self.num_iters):
            q_fes = self.bellman_update(feature_expectations, features_wall)
            q_values = tf.squeeze(tf.matmul(weights_wall, q_fes), [-2])
            policy = tf.nn.softmax(self.beta_planner * q_values, dim=-1)
            repeated_policy = tf.stack([policy] * dim, axis=-2)
            feature_expectations = tf.reduce_sum(
                tf.multiply(repeated_policy, q_fes), axis=-1)
            self.name_to_op['policy'+str(i)] = policy

        # Remove the wall feature
        self.feature_expectations_grid = feature_expectations[:,:,:,:-1]
        dim -= 1
        self.name_to_op['feature_exps_grid'] = self.feature_expectations_grid

        x, y = self.start_x, self.start_y
        self.feature_expectations = self.feature_expectations_grid[:,y,x,:]
        self.name_to_op['feature_exps'] = self.feature_expectations

        q_fes = self.bellman_update(feature_expectations, features_wall)
        q_values = tf.squeeze(tf.matmul(weights_wall, q_fes), [-2])
        self.q_values = q_values
        self.name_to_op['q_values'] = q_values

    def bellman_update(self, fes, features):
        height, width, dim = self.height, self.width, self.feature_dim + 1
        gamma, K = self.gamma, self.K
        extra_row = tf.zeros((K, 1, width, dim))
        extra_col = tf.zeros((K, height, 1, dim))

        north_lookahead = tf.concat([extra_row, fes[:,:-1]], axis=1)
        north_fes = features + gamma * north_lookahead
        south_lookahead = tf.concat([fes[:,1:], extra_row], axis=1)
        south_fes = features + gamma * south_lookahead
        east_lookahead = tf.concat([fes[:,:,1:], extra_col], axis=2)
        east_fes = features + gamma * east_lookahead
        west_lookahead = tf.concat([extra_col, fes[:,:,:-1]], axis=2)
        west_fes = features + gamma * west_lookahead
        return tf.stack([north_fes, south_fes, east_fes, west_fes], axis=-1)

    def update_feed_dict_with_mdp(self, mdp, fd):
        image, features, start_state = mdp.convert_to_numpy_input()
        x, y = start_state
        fd[self.image] = image
        fd[self.features] = features
        fd[self.start_x] = x
        fd[self.start_y] = y


class NoPlanningModel(Model):

    def build_weights(self):
        pass

    def build_planner(self):
        self.feature_expectations = tf.placeholder(
            tf.float32, shape=[self.K, self.feature_dim], name='feature_exps')
        self.name_to_op['feature_exps'] = self.feature_expectations

    def update_feed_dict_with_mdp(self, mdp, fd):
        pass
