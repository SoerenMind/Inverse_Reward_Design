import numpy as np
import tensorflow as tf
import unittest

from planner import GridworldModel, GridworldModelUsingConvolutions
from gridworld import Direction
from gridworld import GridworldMdp, GridworldMdpWithDistanceFeatures
from agents import OptimalAgent

class TestPlanner(unittest.TestCase):
    def test_planner(self):
        grid = GridworldMdp.generate_random(8, 8, 0.1, 0.1)
        mdp = GridworldMdpWithDistanceFeatures(grid)
        feature_dim = len(mdp.goals)
        print 'MDP with ' + str(feature_dim) + ' features'
        mdp.rewards = np.random.randn(feature_dim)
        mdp.feature_weights = mdp.rewards
        agent = OptimalAgent(gamma=0.9, num_iters=10)
        agent.set_mdp(mdp)
        model = GridworldModelUsingConvolutions(feature_dim, 8, 8, 0.9, 10, [], None, None, None)

        with tf.Session() as sess:
            sess.run(model.initialize_op)
            (qvals,) = model.compute_from_reward_weights(['q_values'], sess, mdp, mdp.feature_weights)

        for state in mdp.get_states():
            if mdp.is_terminal(state):
                continue
            x, y = state
            for action in mdp.get_actions(state):
                expected_q = agent.qvalue(state, action)
                action_num = Direction.get_number_from_direction(action)
                actual_q = qvals[0,y,x,action_num]
                # self.assertEqual(expected_q, actual_q)
                self.assertAlmostEqual(expected_q, actual_q, places=2)

if __name__ == '__main__':
    unittest.main()
