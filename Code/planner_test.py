import numpy as np
import random
import tensorflow as tf
import unittest
from itertools import product

from planner import GridworldModel, BanditsModel
from gridworld import Direction
from gridworld import GridworldMdp, GridworldMdpWithDistanceFeatures
from gridworld import NStateMdpGaussianFeatures
from agents import OptimalAgent, ImmediateRewardAgent


class TestPlanner(unittest.TestCase):


    def test_gridworld_planner(self):
        def check_model_equivalent(model, query, weights, mdp, num_iters):
            with tf.Session() as sess:
                sess.run(model.initialize_op)
                (qvals,) = model.compute(
                    ['q_values'], sess, mdp, query, weight_inits=weights)

            agent = OptimalAgent(gamma=model.gamma, num_iters=num_iters)
            for i, proxy in enumerate(model.proxy_reward_space):
                for idx, val in zip(query, proxy):
                    mdp.rewards[idx] = val
                agent.set_mdp(mdp)
                check_qvals_equivalent(qvals[i], agent, mdp)

        def check_qvals_equivalent(qvals, agent, mdp):
            for state in mdp.get_states():
                if mdp.is_terminal(state):
                    continue
                x, y = state
                for action in mdp.get_actions(state):
                    expected_q = agent.qvalue(state, action)
                    action_num = Direction.get_number_from_direction(action)
                    actual_q = qvals[y, x, action_num]
                    # Using softmax, not max, so expect limited accuracy
                    self.assertAlmostEqual(expected_q, actual_q, places=2)

        np.random.seed(1)
        random.seed(1)
        dim = 4
        grid = GridworldMdp.generate_random(8, 8, 0.1, dim)
        mdp = GridworldMdpWithDistanceFeatures(grid)
        mdp.rewards = np.random.randint(-9, 9, size=[dim])
        query = [0, 3]
        other_weights = mdp.rewards[1:3]
        # Use beta_planner = 1000 so that softmax is approximately max
        model = GridworldModel(
            dim, 0.9, len(query), 2, 1, None, 1, 1000, [], 0.1, False, True,
            8, 8, 25)
        check_model_equivalent(model, query, other_weights, mdp, 25)



    def test_bandits_planner(self):
        def check_model_equivalent(model, query, weights, mdp, num_iters):
            with tf.Session() as sess:
                sess.run(model.initialize_op)
                (qvals,) = model.compute(['q_values'], sess, mdp, query, weight_inits=weights)

            agent = ImmediateRewardAgent(gamma=model.gamma)
            for i, proxy in enumerate(model.proxy_reward_space):
                mdp.change_reward(proxy)

                # for idx, val in zip(query, proxy):
                #     mdp.rewards[idx] = val
                agent.set_mdp(mdp)
                check_qvals_equivalent(qvals[:,i], agent, mdp)

        def check_qvals_equivalent(qvals, agent, mdp):
            for state in mdp.get_states():
                if mdp.is_terminal(state):
                    return
                expected_q = agent.qvalue(state, state)
                actual_q = qvals[state]
                self.assertAlmostEqual(expected_q, actual_q, places=5)

        dim = 5
        weights = np.random.randint(-9, 9, size=[dim])
        mdp = NStateMdpGaussianFeatures(
            num_states=7, rewards=weights, start_state=0,
            preterminal_states=[], feature_dim=dim, num_states_reachable=7)
        # query = [0, 2, 3]
        query = [0, 1, 2, 3, 4]
        # other_weights = np.array([weights[1], weights[4]])
        other_weights = np.zeros(0)
        model = BanditsModel(
            dim, 0.9, len(query), 2, 1, None, 1, 1000, [], 0.1, False, True)
        check_model_equivalent(model, query, other_weights, mdp, 20)




if __name__ == '__main__':
    unittest.main()
