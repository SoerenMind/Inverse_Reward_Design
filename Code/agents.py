from agent_interface import Agent
from collections import defaultdict
from utils import Distribution
from gridworld import Direction
import numpy as np
import random

class DirectionalAgent(Agent):
    """An agent that goes in a specific direction or exits.

    This agent only plays grid worlds.
    """
    def __init__(self, direction, mdp, gamma=1.0):
        Agent.__init__(self, gamma)
        self.mdp = mdp
        self.default_action = direction

    def get_action(self, state):
        if self.default_action not in self.mdp.get_actions(state):
            return Direction.EXIT
        return self.default_action



class ImmediateRewardAgent(Agent):
    """An agent that picks the action that gets the highest reward on the next state.
    Called 'immediate reward' because the reward for the current state is assumed fixed (doesnt depend on action).
    """
    def __init__(self, gamma=1.0):
        Agent.__init__(self, gamma)

    # @profile
    def qvalue(self, state, action):
        """Estimates Q(state,a) by sampling the reward off the next state"""
        results = self.mdp.get_transition_states_and_probs(state, action)
        rand = random.random()
        sum = 0
        for next_state, prob in results:
            next_action = Direction.EXIT  # The action does not matter as the reward depends only on the state
            sum += prob
            if sum > 1.0:
                raise ValueError('Total transition probability more than one.')
            if rand < sum:
                next_reward = self.mdp.get_reward(next_state, next_action)
                action_value = next_reward
                return action_value
        raise ValueError("Total transition probability less than one")

    # @profile
    def get_action_distribution(self, state):
        """Returns uniform distribution over actions of highest q-value.
        Note that the q-value is approximated by sampling the next state and and next reward.
        """
        # (Could instead define reward(state,action) = reward(features(next_state)))
        actions = self.mdp.get_actions(state)
        best_value, best_actions = float("-inf"), []
        # qvalues = [(a, self.qvalue(state, a)) for a in actions]

        # If ending episode:
        if actions[0] == Direction.EXIT:
            assert len(actions) == 1
            return Distribution({Direction.EXIT: 1})

        for a in actions:
            action_value = self.qvalue(state, a)
            if action_value > best_value:
                best_value, best_actions = action_value, [a]
            elif action_value == best_value:
                best_actions.append(a)
        return Distribution({a: 1 for a in best_actions})

    def compute_policy(self):
        """Stores the optimal action for each state"""
        states = self.mdp.get_states()
        states.remove(self.mdp.terminal_state)
        rewards = np.empty(len(states))
        for i, state in enumerate(states):
            features = self.mdp.get_features(state)
            reward = self.mdp.get_reward_from_features(features)
            rewards[i] = reward
        self.best_action = int(np.argmax(rewards))

    def quick_get_action(self, state):
        return self.best_action



class ValueIterationLikeAgent(Agent):
    """An agent that chooses actions using something similar to value iteration.

    Instead of working directly on states from the mdp, we perform value
    iteration on generalized states (called mus), following the formalism in
    "Learning the Preferences of Bounded Agents" from a NIPS 2015 workshop.

    The algorithm in this class is simply standard value iteration, but
    subclasses can easily change the behavior while reusing most of the code by
    overriding hooks into the algorithm.
    """

    def __init__(self, gamma=1.0, beta=None, num_iters=50):
        """Initializes the agent, setting any relevant hyperparameters.

        gamma: Discount factor.
        beta: Noise parameter when choosing actions. beta=None implies that
        there is no noise, otherwise actions are chosen with probability
        proportional to exp(beta * value).
        num_iters: The maximum number of iterations of value iteration to run.
        """
        super(ValueIterationLikeAgent, self).__init__(gamma)
        self.beta = beta
        self.num_iters = num_iters

    def set_mdp(self, mdp):
        super(ValueIterationLikeAgent, self).set_mdp(mdp)
        return self.compute_values()

    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates a dictionary self.values, such that self.values[mu] is the
        value (a float) of the generalized state mu.
        """
        values = defaultdict(float)
        for iter in range(self.num_iters):
            new_values = defaultdict(float)
            for s in self.mdp.get_states():
                actions = self.mdp.get_actions(s)
                if not actions:
                    continue
                qvalues = [(self.qvalue(s, a, values), a) for a in actions]
                _, chosen_action = max(qvalues)
                new_values[s] = self.qvalue(s, chosen_action, values)

            if self.converged(values, new_values):
                self.values = new_values
                return iter

            values = new_values

        self.values = values
        return iter

    def converged(self, values, new_values):
        """Returns True if value iteration has converged.

        Value iteration has converged if no value has changed by more than 1e-3.

        values: The values from the previous iteration of value iteration.
        new_values: The new value computed during this iteration.
        """
        for s in new_values.keys():
            if abs(values[s] - new_values[s]) > 1e-3:
                return False
        return True

    def qvalue(self, s, a, values=None):
        """Computes Q(s, a) from the values table.

        s: State
        a: Action
        values: Dictionary such that values[s] is the value of
        state s. If None, then self.values is used instead.
        """
        if values is None:
            values = self.values
        r = self.mdp.get_reward(s, a)
        transitions = self.mdp.get_transition_states_and_probs(s, a)
        return r + self.gamma * sum([p * values[s2] for s2, p in transitions])

    def get_action_distribution(self, s):
        """Returns a Distribution over actions.

        Note that this is a normal state s, not a generalized state mu.
        """
        actions = self.mdp.get_actions(s)
        if self.beta is not None:
            q_vals = np.array([self.qvalue(s, a) for a in actions])
            q_vals = q_vals - np.mean(q_vals)  # To prevent overflow in exp
            action_dist = np.exp(self.beta * q_vals)
            return Distribution(dict(zip(actions, action_dist)))
        else:
            best_value, best_actions = float("-inf"), []
            for a in actions:
                action_value = self.qvalue(s, a)
                if action_value > best_value:
                    best_value, best_actions = action_value, [a]
                elif action_value == best_value:
                    best_actions.append(a)
            return Distribution({a : 1 for a in best_actions})


class OptimalAgent(ValueIterationLikeAgent):
    """An agent that implements regular value iteration."""
    pass
