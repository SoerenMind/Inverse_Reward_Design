import numpy as np
import random
import tensorflow as tf
import unittest
from itertools import product

from planner import GridworldModel, GridworldModelUsingConvolutions
from gridworld import Direction
from gridworld import GridworldMdp, GridworldMdpWithDistanceFeatures
from agents import OptimalAgent

class TestPlanner(unittest.TestCase):
    def test_gridworld_planner(self):
        np.random.seed(1)
        random.seed(1)
        grid = GridworldMdp.generate_random(8, 8, 0.1, 0.1)
        mdp = GridworldMdpWithDistanceFeatures(grid)
        dim = 4
        while len(mdp.goals) != dim:
            grid = GridworldMdp.generate_random(8, 8, 0.1, 0.1)
            mdp = GridworldMdpWithDistanceFeatures(grid)

        mdp.rewards = np.random.randn(dim)
        mdp.feature_weights = mdp.rewards
        query = [0, 3]
        other_weights = mdp.rewards[1:3]
        proxy_space = list(product(range(-1, 2), range(-1, 2)))
        dummy_true_reward_matrix = np.random.rand(3, dim)
        model = GridworldModel(dim, 0.9, query, proxy_space, dummy_true_reward_matrix, mdp.rewards, 1, 'entropy', 8, 8, 10)

        with tf.Session() as sess:
            sess.run(model.initialize_op)
            (qvals,) = model.compute(['q_values'], sess, mdp, weight_inits=other_weights)

        agent = OptimalAgent(gamma=0.9, num_iters=10)
        for i, proxy in enumerate(proxy_space):
            for idx, val in zip(query, proxy):
                mdp.rewards[idx] = val
            agent.set_mdp(mdp)
            self.check_equivalent(qvals[i], agent, mdp)

    def check_equivalent(self, qvals, agent, mdp):
        for state in mdp.get_states():
            if mdp.is_terminal(state):
                continue
            x, y = state
            for action in mdp.get_actions(state):
                expected_q = agent.qvalue(state, action)
                action_num = Direction.get_number_from_direction(action)
                actual_q = qvals[y,x,action_num]
                # self.assertEqual(expected_q, actual_q)
                self.assertAlmostEqual(expected_q, actual_q, places=2)

if __name__ == '__main__':
    unittest.main()
