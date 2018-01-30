from agent_interface import Agent
from collections import defaultdict
# from utils import Distribution
from gridworld import Direction
import numpy as np
import random

class DirectionalAgent(Agent):
    """An agent that goes in a specific direction or exits.

    This agent only plays grid worlds.
    """
    def __init__(self, direction, gamma=1.0):
        Agent.__init__(self, gamma)
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
            # return Distribution({actions[0]: 1})
            return [Direction.EXIT]

        for a in actions:
            action_value = self.qvalue(state, a)
            if action_value > best_value:
                best_value, best_actions = action_value, [a]
            elif action_value == best_value:
                best_actions.append(a)
        # return Distribution({a: 1 for a in best_actions})
        # return Distribution({best_actions[0]: 1})
        return best_actions


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
        self.compute_values()

    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates a dictionary self.values, such that self.values[mu] is the
        value (a float) of the generalized state mu.
        """
        values = defaultdict(float)
        for iter in range(self.num_iters):
            new_values = defaultdict(float)
            for mu in self.get_mus():
                actions = self.get_actions(mu)
                if not actions:
                    continue
                new_mu = self.get_mu_for_planning(mu)  # Typically new_mu == mu
                qvalues = [(self.qvalue(new_mu, a, values), a) for a in actions]
                _, chosen_action = max(qvalues)
                new_values[mu] = self.qvalue(mu, chosen_action, values)

            if self.converged(values, new_values):
                self.values = new_values
                return

            values = new_values

        self.values = values

    def converged(self, values, new_values):
        """Returns True if value iteration has converged.

        Value iteration has converged if no value has changed by more than 1e-3.

        values: The values from the previous iteration of value iteration.
        new_values: The new value computed during this iteration.
        """
        for mu in new_values.keys():
            if abs(values[mu] - new_values[mu]) > 1e-3:
                return False
        return True

    def qvalue(self, mu, a, values=None):
        """Computes Q(mu, a) from the values table.

        mu: Generalized state
        a: Action
        values: Dictionary such that values[mu] is the value of generalized
        state mu. If None, then self.values is used instead.
        """
        if values is None:
            values = self.values
        r = self.get_reward(mu, a)
        transitions = self.get_transition_mus_and_probs(mu, a)
        return r + self.gamma * sum([p * values[mu2] for mu2, p in transitions])

    def get_action_distribution(self, s):
        """Returns a Distribution over actions.

        Note that this is a normal state s, not a generalized state mu.
        """
        mu = self.extend_state_to_mu(s)
        actions = self.mdp.get_actions(s)
        if self.beta is not None:
            q_vals = np.array([self.qvalue(mu, a) for a in actions])
            q_vals = q_vals - np.mean(q_vals)  # To prevent overflow in exp
            action_dist = np.exp(self.beta * q_vals)
            return Distribution(dict(zip(actions, action_dist)))

        best_value, best_actions = float("-inf"), []
        for a in actions:
            action_value = self.qvalue(mu, a)
            if action_value > best_value:
                best_value, best_actions = action_value, [a]
            elif action_value == best_value:
                best_actions.append(a)
        # return Distribution({a : 1 for a in best_actions})
        return Distribution({best_actions[0] : 1})

    def get_mus(self):
        """Returns all possible generalized states the agent could be in.

        This is the equivalent of self.mdp.get_states() for generalized states.
        """
        return self.mdp.get_states()

    def get_actions(self, mu):
        """Returns all actions the agent could take from generalized state mu.

        This is the equivalent of self.mdp.get_actions() for generalized states.
        """
        s = self.extract_state_from_mu(mu)
        return self.mdp.get_actions(s)

    def get_reward(self, mu, a):
        """Returns the reward for taking action a from generalized state mu.

        This is the equivalent of self.mdp.get_reward() for generalized states.
        """
        s = self.extract_state_from_mu(mu)
        return self.mdp.get_reward(s, a)

    def get_transition_mus_and_probs(self, mu, a):
        """Gets information about possible transitions for the action.

        This is the equivalent of self.mdp.get_transition_states_and_probs() for
        generalized states. So, it returns a list of (next_mu, prob) pairs,
        where next_mu must be a generalized state.
        """
        s = self.extract_state_from_mu(mu)
        return self.mdp.get_transition_states_and_probs(s, a)

    def get_mu_for_planning(self, mu):
        """Returns the generalized state that an agent uses for planning.

        Specifically, the returned state is used when looking forward to find
        the expected value of a future state.
        """
        return mu

    def extend_state_to_mu(self, state):
        """Converts a normal state to a generalized state."""
        return state

    def extract_state_from_mu(self, mu):
        """Converts a generalized state to a normal state."""
        return mu

class OptimalAgent(ValueIterationLikeAgent):
    """An agent that implements regular value iteration."""
    pass

class DelayDependentAgent(ValueIterationLikeAgent):
    """An agent that plans differently as it looks further in the future.

    Delay dependent agents calculate values differently as they look further
    into the future. They extend the state with the delay d. Intuitively, the
    generalized state (s, d) stands for "the agent is looking d steps into the
    future at which point is in state s".

    This class is not a full agent. It simply overrides the necessary methods in
    order to support generalized states containing the delay. Subclasses must
    override other methods in order to actually use the delay to change the
    value iteration algorithm in some way.
    """

    def __init__(self, max_delay=None, gamma=1.0, beta=None, num_iters=50):
        """Initializes the agent, setting any relevant hyperparameters.

        max_delay: Integer specifying the maximum value of d to consider during
        planning. If None, then max_delay is set equal to num_iters, which will
        ensure that the values for generalized states of the form (s, 0) are not
        affected by the max_delay. Note that large values of max_delay can cause
        a significant performance overhead.
        """
        super(DelayDependentAgent, self).__init__(gamma, beta, num_iters)
        self.max_delay = max_delay if max_delay is not None else num_iters

    def get_mus(self):
        """Override to handle states with delays."""
        states = self.mdp.get_states()
        return [(s, d) for s in states for d in range(self.max_delay + 1)]

    def get_transition_mus_and_probs(self, mu, a):
        """Override to handle states with delays."""
        s, d = mu
        transitions = self.mdp.get_transition_states_and_probs(s, a)
        newd = min(d + 1, self.max_delay)
        return [((s2, newd), p) for s2, p in transitions]

    def extend_state_to_mu(self, state):
        """Override to handle states with delays."""
        return (state, 0)

    def extract_state_from_mu(self, mu):
        """Override to handle states with delays."""
        return mu[0]

class TimeDiscountingAgent(DelayDependentAgent):
    """A hyperbolic time discounting agent.

    Such an agent discounts future rewards in a time inconsistent way. If they
    would get future reward R, they instead plan as though they would get future
    reward R/(1 + kd). See the paper "Learning the Preferences of Ignorant,
    Inconsistent Agents" for more details.
    """

    def __init__(self, max_delay, discount_constant,
                 gamma=1.0, beta=None, num_iters=50):
        """Initializes the agent, setting any relevant hyperparameters.

        discount_constant: Float. The parameter k in R/(1 + kd) (see above).
        """
        super(TimeDiscountingAgent, self).__init__(
            max_delay, gamma, beta, num_iters)
        self.discount_constant = discount_constant

    def get_reward(self, mu, a):
        """Override to apply hyperbolic time discounting."""
        s, d = mu
        discount = (1.0 / (1.0 + self.discount_constant * d))
        return discount * self.mdp.get_reward(s, a)

class NaiveTimeDiscountingAgent(TimeDiscountingAgent):
    """The naive time discounting agent.

    See the paper "Learning the Preferences of Ignorant, Inconsistent Agents"
    for more details.
    """
    pass

class SophisticatedTimeDiscountingAgent(TimeDiscountingAgent):
    """The sophisticated time discounting agent.

    See the paper "Learning the Preferences of Ignorant, Inconsistent Agents"
    for more details.
    """
    def get_mu_for_planning(self, mu):
        """Override to implement sophisticated time-inconsistent behavior."""
        s, d = mu
        return (s, 0)

class MyopicAgent(DelayDependentAgent):
    """An agent that only looks forward for a fixed horizon."""

    def __init__(self, horizon, gamma=1.0, beta=None, num_iters=50):
        """Initializes the agent, setting any relevant hyperparameters.

        horizon: Integer, the number of steps forward that the agent looks while
        planning. This must also be used as the max_delay -- if the max_delay
        was lower, the agent would no longer have a finite horizon, and if the
        max_delay was higher, we would do extra computation that is never used.
        """
        # The maximum delay should be the horizon.
        super(MyopicAgent, self).__init__(horizon, gamma, beta, num_iters)
        self.horizon = horizon

    def get_reward(self, mu, a):
        """Override to ignore rewards after the horizon."""
        s, d = mu
        if d >= self.horizon:
            return 0
        return super(MyopicAgent, self).get_reward(mu, a)
