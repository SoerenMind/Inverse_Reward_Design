import numpy as np
import tensorflow as tf

from gridworld import Direction

class Model(object):
    def __init__(self, feature_dim, gamma, query_size, proxy_reward_space,
                 true_reward_matrix, true_reward, beta, beta_planner, objective, lr, no_planning=False):
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.query_size = query_size
        # List of possible settings for the query features
        # Eg. If we are querying features 1 and 4, and discretizing each into 3
        # buckets (-1, 0 and 1), the proxy reward space would be
        # [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
        self.proxy_reward_space = proxy_reward_space
        self.K = len(self.proxy_reward_space)
        self.true_reward_matrix = np.array(true_reward_matrix, dtype=np.float32)
        self.true_reward = true_reward
        self.beta = beta
        self.beta_planner = beta_planner
        self.lr = lr
        self.build_tf_graph(objective, no_planning)

    def build_tf_graph(self, objective, no_planning):
        self.name_to_op = {}
        if not no_planning:
            self.build_weights()
            self.build_planner()
        else:
            self.feature_expectations = tf.placeholder(tf.float32, shape=[None, self.feature_dim], name='feature_exps')
            self.name_to_op['feature_exps'] = self.feature_expectations
        self.build_map_to_posterior()
        self.build_map_to_objective(objective)
        # Initializing the variables
        self.initialize_op = tf.global_variables_initializer()

    def build_weights(self):
        query_size, dim, K = self.query_size, self.feature_dim, self.K
        num_fixed = dim - query_size
        self.query_weights= tf.constant(self.proxy_reward_space, dtype=tf.float32, name="query_weights")
        self.other_weights = tf.Variable(tf.zeros([num_fixed]), name="other_weights")
        self.weight_inputs = tf.placeholder(tf.float32, shape=[num_fixed], name="weight_inputs")
        self.assign_op = self.other_weights.assign(self.weight_inputs)



        # Let's say query is [1, 3] and there are 6 features.
        # query_weights = [10, 11] and weight_inputs = [12, 13, 14, 15].
        # Then we want self.weights to be [12, 10, 13, 11, 14, 15].
        # Concatenate to get [10, 11, 12, 13, 14, 15]
        repeated_weights = tf.stack([self.other_weights] * K, axis=0)
        unordered_weights = tf.concat(
            [self.query_weights, repeated_weights], axis=1)
        # Then permute using gather to get the desired result.
        # The permutation can be computed from the query [1, 3] using
        # get_permutation_from_query.
        self.permutation = tf.placeholder(tf.int32, shape=[dim])
        self.weights = tf.gather(unordered_weights, self.permutation, axis=-1)

        self.name_to_op = {}    # Remove
        self.name_to_op['weights'] = self.weights
        self.name_to_op['query_weights'] = self.query_weights
        self.name_to_op['other_weights'] = self.other_weights


        # print self.query_weights.shape
        # print self.other_weights.shape
        # print self.weights.shape


    def build_planner(self):
        raise NotImplemented('Should be implemented in subclass')

    def build_map_to_posterior(self):
        """
        Maps self.feature_exp (created by planner) to self.log_posterior.
        """
        # Get log likelihoods for true reward matrix


        self.feature_expectations_input = tf.identity(self.feature_expectations, name='feature_expectations_input')

        self.true_reward_matrix_tensor = tf.constant(
            self.true_reward_matrix, dtype=tf.float32, name="true_reward_matrix_tensor", shape=self.true_reward_matrix.shape)
        self.avg_reward_matrix = tf.tensordot(
            self.feature_expectations, tf.transpose(self.true_reward_matrix_tensor), axes=[-1, -2], name='avg_reward_matrix')
        log_likelihoods_new = self.beta * self.avg_reward_matrix


        # Calculate posterior
        self.prior = tf.placeholder(tf.float32, name="prior", shape=(len(self.true_reward_matrix)))
        log_Z_w = tf.reduce_logsumexp(log_likelihoods_new, axis=0, name='log_Z_q')
        log_P_q_z = log_likelihoods_new - log_Z_w
        self.log_Z_q, max_a, max_b = logdot(log_P_q_z, tf.log(self.prior))
        self.log_posterior = log_P_q_z + tf.log(self.prior) - self.log_Z_q
        self.posterior = tf.exp(self.log_posterior, name="posterior")

        self.post_sum_to_1 = tf.reduce_sum(tf.exp(self.log_posterior), axis=1, name='post_sum_to_1')


        # Get log likelihoods for actual true reward
        if self.true_reward is not None:
            self.true_reward_tensor = tf.constant(
                self.true_reward.reshape(1,-1), dtype=tf.float32, name="true_reward_tensor", shape=[1, self.feature_dim])
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
            interm_tensor = tf.exp(self.true_log_posterior + tf.log(- self.true_log_posterior))
            self.true_ent = tf.reduce_sum(interm_tensor, axis=0, name="true_entropy", keep_dims=True)
            self.name_to_op['true_entropy'] = self.true_ent

        # Fill name to ops dict
        self.name_to_op['avg_reward_matrix'] = self.avg_reward_matrix
        self.name_to_op['true_reward_matrix_tensor'] = self.true_reward_matrix_tensor
        self.name_to_op['prior'] = self.prior
        self.name_to_op['posterior'] = self.posterior
        self.name_to_op['post_sum_to_1'] = self.post_sum_to_1


    def build_map_to_objective(self, objective):
        """
        :param objective: string that specifies the objective function
        """
        if 'entropy' in objective:
            # # Calculate exp entropy without log space trick
            # post_ent = - tf.reduce_sum(
            #     tf.multiply(tf.exp(self.log_posterior), self.log_posterior), axis=1, keep_dims=True, name='post_ent')
            # self.exp_post_ent = tf.reduce_sum(
            #     tf.multiply(post_ent, tf.exp(self.log_Z_q)), axis=0, keep_dims=True, name='exp_post_entropy')
            # self.name_to_op['entropy'] = self.exp_post_ent


            # Calculate entropy as sum exp (log p + log (-log p))
            interm_tensor = tf.exp(self.log_posterior + tf.log(- self.log_posterior))
            self.post_ent_new = tf.reduce_sum(interm_tensor, axis=1, name="entropy_per_answer", keep_dims=True)
            self.name_to_op['entropy_per_answer'] = self.post_ent_new
            self.exp_post_ent = tf.reduce_sum(
                tf.multiply(self.post_ent_new, tf.exp(self.log_Z_q)), axis=0, keep_dims=True, name='entropy')
            self.name_to_op['entropy'] = self.exp_post_ent

            # Set up optimizer
            if self.query_size < self.feature_dim:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.minimize_op = optimizer.minimize(
                    self.exp_post_ent, var_list=[self.other_weights])
                self.name_to_op['minimize'] = self.minimize_op

        if 'variance' in objective:
            true_rewards = tf.constant(self.true_reward_matrix, dtype=tf.float32, name='true_rewards')
            # post_avg, post_var = tf.nn.moments(true_rewards, axes=[0], keep_dims=False)
            posterior_stack = tf.stack([self.posterior[0]] * self.feature_dim, axis=1)
            posterior_stack = tf.expand_dims(self.posterior[0], axis=1)

            # true_rewards_stack = tf.stack([true_rewards] * self.K, axis=0)
            post_avg, post_var = tf.nn.weighted_moments(
                true_rewards, [0, 0], posterior_stack, name="moments", keep_dims=False)

            data = tf.constant([[0.,2.,4.],[0.,2.,4.]], dtype=tf.float32)
            avg, var = tf.nn.moments(data, axes=[1,0])

            # self.generalized_var = tf.matrix_determinant(post_var, name="generalized_variance")
            # self.name_to_op['variance'] = self.generalized_var
            self.name_to_op['post_avg'] = post_avg
            self.name_to_op['post_var'] = post_var
            self.name_to_op['posterior_stack'] = posterior_stack
            self.name_to_op['var'] = var


        if 'regret' in objective:
            pass

        if 'avg_reward' in objective:
            pass

        if 'query_entropy' in objective:
            pass


    # TODO: Remove the feature_expectations_test_input argument
    def compute(self, outputs, sess, mdp, query, prior=None, weight_inits=None, feature_expectations_input = None,
                gradient_steps=0, discrete_query=None):
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
        fd = self.create_mdp_feed_dict(mdp)

        if weight_inits is not None:
            fd[self.weight_inputs] = weight_inits   # Should this be done after running assign_op?
            # fd = {self.weight_inputs: weight_inits}
            sess.run([self.assign_op], feed_dict=fd)
        elif discrete_query is not None:
            fd[self.query_weights] = discrete_query   # Should this be done with another assign op?
        elif feature_expectations_input is not None:
            fd[self.feature_expectations] = feature_expectations_input

        if prior is not None:
            fd[self.prior] = prior

        fd[self.permutation] = self.get_permutation_from_query(query)

        if gradient_steps > 0:
            for step in range(gradient_steps):
                # print sess.run(self.name_to_op['entropy'], feed_dict=fd)
                sess.run(self.minimize_op, feed_dict=fd)

        def get_op(name):
            if name not in self.name_to_op:
                raise ValueError("Unknown op name: " + str(name))
            return sess.run([self.name_to_op[name]], feed_dict=fd)[0]

        return [get_op(name) for name in outputs]
        # output_ops = [get_op(name) for name in outputs]
        # return sess.run(output_ops, feed_dict=fd)


    # TODO: Remove the feature_expectations_test_input argument
    def compute_no_planning(self, outputs, sess, mdp, query, prior=None, weight_inits=None, feature_expectations_input = None,
                gradient_steps=0, discrete_query=None, true_reward=None):
        """
        Computes outputs from feature_expectations_input
        """
        fd = {}

        fd[self.feature_expectations] = feature_expectations_input
        if true_reward is not None:
            fd[self.true_reward_tensor] = true_reward.reshape(1,-1)

        if prior is not None:
            fd[self.prior] = prior


        def get_op(name):
            if name not in self.name_to_op:
                raise ValueError("Unknown op name: " + str(name))
            return sess.run([self.name_to_op[name]], feed_dict=fd)[0]

        return [get_op(name) for name in outputs]
        # output_ops = [get_op(name) for name in outputs]
        # return sess.run(output_ops, feed_dict=fd)



    def create_mdp_feed_dict(self, mdp):
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

        # Calculate feature expectations
        probs_stack = tf.stack([self.state_probs] * self.feature_dim, axis=1)
        features_stack = tf.multiply(tf.stack([self.features] * self.K, axis=2), probs_stack, name='multi')
        self.feature_expectations = tf.reduce_sum(features_stack, axis=0, keep_dims=False, name="feature_exps")
        self.feature_expectations = tf.transpose(self.feature_expectations)
        self.name_to_op['feature_exps'] = self.feature_expectations


    def create_mdp_feed_dict(self, mdp):
        return {self.features: mdp.convert_to_numpy_input()}


class GridworldModel(Model):
    def __init__(self, feature_dim, gamma, query_size, proxy_reward_space,
                 true_reward_matrix, true_reward, beta, beta_planner, objective,
                 lr, height, width, num_iters, no_planning=False):
        self.height = height
        self.width = width
        self.num_iters = num_iters
        self.num_actions = 4
        super(GridworldModel, self).__init__(
            feature_dim, gamma, query_size, proxy_reward_space, true_reward_matrix,
            true_reward, beta, beta_planner, objective, lr, no_planning=no_planning)

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
        for _ in range(self.num_iters):
            q_fes = self.bellman_update(feature_expectations, features_wall)
            q_values = tf.squeeze(tf.matmul(weights_wall, q_fes), [-2])
            policy = tf.nn.softmax(self.beta_planner * q_values, dim=-1)
            repeated_policy = tf.stack([policy] * dim, axis=-2)
            feature_expectations = tf.reduce_sum(
                tf.multiply(repeated_policy, q_fes), axis=-1)

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

    def create_mdp_feed_dict(self, mdp):
        image, features, start_state = mdp.convert_to_numpy_input()
        x, y = start_state
        return {
            self.image: image,
            self.features: features,
            self.start_x: x,
            self.start_y: y
        }


def logdot(a,b):
    max_a, max_b = tf.reduce_max(a), tf.reduce_max(b)   # TODO: make sure broadcasting is right. Don't let max_a be max over whole matrix.
    exp_a, exp_b = a - max_a, b - max_b
    exp_a = tf.exp(exp_a)
    exp_b = tf.exp(exp_b)
    # c = tf.tensordot(exp_a, exp_b, axes=1)
    c = tf.reduce_sum(tf.multiply(exp_a,exp_b), axis=1, keep_dims=True)
    c = tf.log(c) + max_a + max_b
    return c, max_a, max_b
