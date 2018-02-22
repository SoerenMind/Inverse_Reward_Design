import tensorflow as tf
import numpy as np


def logdot(a,b):
    max_a, max_b = tf.reduce_max(a), tf.reduce_max(b)   # TODO: make sure broadcasting is right. Don't let max_a be max over whole matrix.
    exp_a, exp_b = a - max_a, b - max_b # TODO: fix axes
    exp_a = tf.exp(exp_a)
    exp_b = tf.exp(exp_b)
    # c = tf.tensordot(exp_a, exp_b, axes=1)
    c = tf.reduce_sum(tf.multiply(exp_a,exp_b), axis=1, keepdims=True) # TODO: adapt to get matrix diagonal instead
    c = tf.log(c) + max_a + max_b
    return c, max_a, max_b

"Translating numpy code to tf bit-by-bit"
def get_likelihoods_from_feature_expectations(feature_exp_matrix,
                                              true_reward_matrix,
                                              beta,
                                              prior,
                                              feature_exp_matrix_true=None,
                                              precision=tf.float64):
    # Get shape parameters
    num_true_rewards = true_reward_matrix.shape[0]


    # Build graph
    # TODO: Make sure every tensor starts with a dimension for the batch size (number of features)

    # Get proxy-true likelihood matrix
    feature_exp = tf.placeholder(precision, name='feature_exp')    # shape=(2,20)
    true_rewards = tf.placeholder(precision, name='true_rewards')  # shape=(10,20)
    avg_reward_matrix = tf.matmul(feature_exp, tf.transpose(true_rewards), name='avg_reward_matrix')
    # likelihoods = tf.exp(beta * avg_reward_matrix, name='likelihoods')
    log_likelihoods_new = beta * avg_reward_matrix

    # Get true-true avg rewards
    true_feature_exp = tf.constant(feature_exp_matrix_true, dtype=precision)
    true_reward_avg_reward_matrix = tf.matmul(true_feature_exp, tf.transpose(true_rewards))
    true_reward_avg_reward_vec = tf.diag_part(true_reward_avg_reward_matrix,name='true_reward_avg_reward_vec')

    # Calculate posterior
    log_Z_w = tf.reduce_logsumexp(log_likelihoods_new, axis=0, name='log_Z_q')
    log_P_q_z = log_likelihoods_new - log_Z_w

    # Not fully in log space:
    P_q_z = tf.exp(log_P_q_z, name='probs')
    sum_to_1 = tf.reduce_sum(P_q_z, axis=0, name='prob_sum_to_1')
    prior = tf.constant(prior, dtype=precision, name='prior')
    prior_expand = tf.expand_dims(prior,1)
    Z_q = tf.matmul(P_q_z, prior_expand, name='Z_q')
    # Z_q = tf.einsum('n,nm->m',prior,P_q_z)
    posterior = tf.multiply(P_q_z, prior, name='posterior')
    posterior = posterior / Z_q # reshape Z_q back to vector
    post_ent = tf.reduce_sum(- posterior * tf.log(posterior), axis=1, name='post_ent')   # check summing correct

    # In log space:
    log_Z_q, max_a, max_b = logdot(log_P_q_z, tf.log(prior))
    log_posterior = log_P_q_z + tf.log(prior) - tf.expand_dims(log_Z_q, 1) # check broadcasting correct
    post_sum_to_1 = tf.reduce_sum(tf.exp(log_posterior), axis=1, name='post_sum_to_1')
    log_post_ent = logdot(log_posterior, tf.log(log_posterior))





    # Run graph in session
    with tf.Session() as sess:
        # var_list = [likelihoods, log_likelihoods, log_likelihoods_new, true_reward_avg_reward_vec]
        var_list = [log_Z_w, log_P_q_z, P_q_z, sum_to_1, Z_q, posterior, log_Z_q, post_ent, post_sum_to_1, log_post_ent]
        return sess.run(var_list, feed_dict={feature_exp: feature_exp_matrix,
                                    true_rewards: true_reward_matrix, true_feature_exp: feature_exp_matrix_true})


if __name__=='__main__':
    n_proxy = 2; n_true = 5
    # feature_dim = 20
    fe = np.eye(n_proxy,20)
    fe_true = np.eye(n_true,20)
    r_space_true = np.eye(n_true, 20)
    prior = np.ones(n_true)
    avg_reward_matrix = get_likelihoods_from_feature_expectations(fe, r_space_true, 2, prior, feature_exp_matrix_true=fe_true)
    print avg_reward_matrix.round(1)