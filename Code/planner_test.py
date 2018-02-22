import numpy as np
import tensorflow as tf
import unittest

from planner import Model
from gridworld import Direction
from gridworld import GridworldMdp, GridworldMdpWithDistanceFeatures
from agents import OptimalAgent

class TestPlanner(unittest.TestCase):
    def test_planner(self):
        grid = GridworldMdp.generate_random(8, 8, 0.1, 0.1)
        mdp = GridworldMdpWithDistanceFeatures(grid)
        # Hack because agents.py expects mdp.rewards to exist
        mdp.rewards = np.array([1, -1])
        agent = OptimalAgent(gamma=0.9, num_iters=20)
        agent.set_mdp(mdp)
        feature_dim = len(mdp.goals)
        model = Model(feature_dim, 8, 8, 0.9, 21)

        with tf.Session() as sess:
            sess.run(model.initialize_op)
            qvals = model.compute_qvals(sess, mdp)

        for state in mdp.get_states():
            if mdp.is_terminal(state):
                continue
            x, y = state
            for action in mdp.get_actions(state):
                expected_q = agent.qvalue(state, action)
                action_num = Direction.get_number_from_direction(action)
                actual_q = qvals[0,y,x,action_num]
                self.assertAlmostEqual(expected_q, actual_q, places=2)

if __name__ == '__main__':
    unittest.main()
