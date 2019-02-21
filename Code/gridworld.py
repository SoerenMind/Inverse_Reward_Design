from collections import defaultdict
from disjoint_sets import DisjointSets
from functools import reduce
from operator import mul
import numpy as np
import random
import itertools
from scipy.stats import invwishart, multivariate_normal


class Mdp(object):
    def __init__(self):
        pass

    def get_start_state(self):
        """Returns the start state."""
        raise NotImplementedError


    def get_states(self):
        """Returns a list of all possible states the agent can be in.

        Note it is not guaranteed that the agent can reach all of these states.
        """
        raise NotImplementedError

    def get_actions(self, state):
        """Returns the list of valid actions for 'state'.

        Note that you can request moves into walls. The order in which actions
        are returned is guaranteed to be deterministic, in order to allow agents
        to implement deterministic behavior.
        """
        raise NotImplementedError

    def get_features(self,state):
        """Returns feature vector for state"""
        return self.features[state]

    def get_reward(self, state, action):
        """Get reward for state, action transition.

        This is the living reward, except when we take EXIT, in which case we
        return the reward for the current state.
        """
        features = self.get_features(state)
        return self.get_reward_from_features(features)

    def get_reward_from_features(self, features):
        """Returns dot product of features with reward weights. Uses self.rewards unless extra argument is given."""
        reward = np.dot(features, self.rewards)  # Minimize lines in this function by returning this directly.
        return reward

    def is_terminal(self, state):
        """Returns True if the current state is terminal, False otherwise.

        A state is terminal if there are no actions available from it (which
        means that the episode is over).
        """
        return state == self.terminal_state

    def get_transition_states_and_probs(self, state, action):
        """Gets information about possible transitions for the action.

        Returns list of (next_state, prob) pairs representing the states
        reachable from 'state' by taking 'action' along with their transition
        probabilities.
        """
        raise NotImplemented

    def change_reward(self, rewards):
        '''Sets new reward function (for reward design).'''
        self.rewards = rewards

    def convert_to_numpy_input(self):
        """Encodes this MDP in a format well-suited for deep models."""
        raise NotImplemented



class NStateMdp(Mdp):
    '''An MDP with N=num_states states and N actions which are always possible.
    Action i leads to state i.
    preterminal_states transition to a generic terminal state via a terminal action.
    '''
    def __init__(self,num_states, rewards, start_state, preterminal_states):
        super(NStateMdp,self).__init__()
        self.num_states = num_states    # Or make a grid and add n actions
        self.terminal_state = 'Terminal State'
        self.preterminal_states = preterminal_states    # Preterminal states should include states with no available actions. Otherwise get_actions==>[]==>
        # self.populate_rewards_and_start_state(rewards)
        self.start_state = start_state
        self.rewards = np.array(rewards)

    def get_states(self):
        all_states = range(self.num_states)
        # return all_states + [self.terminal_state]
        return all_states

    def get_start_state(self):
        """Returns the start state."""
        return self.start_state

    def get_actions(self, state):
        """Returns the list of valid actions for 'state'.
        Note that all actions are valid unless the state is terminal (then none are valid).
        """
        if self.is_terminal(state):
            return []
        if state in self.preterminal_states:
            return [Direction.EXIT]
        act = range(self.num_states)
        return act

    def get_reward(self,state,action):
        """Get reward for state, action transition."""
        features = self.get_features(state)
        return self.get_reward_from_features(features)

    def change_reward(self, rewards):
        '''Sets new reward function (for reward design).'''
        try: assert self.rewards.shape == rewards.shape
        except:
            pass
        self.rewards = rewards

    # @profile
    def get_reward_from_features(self, features):
        """Returns dot product of features with reward weights."""
        return np.dot(features, self.rewards)

    def get_state_list(self, trajectories):
        """Returns list of states in 'trajectories'."""
        state_list = np.array([tup[0] for trajectory in trajectories for tup in trajectory])
        return state_list

    def is_terminal(self, state):
        """Returns True if the current state is terminal, False otherwise."""
        return state == self.terminal_state

    def get_transition_states_and_probs(self, state, action):
        """Gets information about possible transitions for the action.

        Returns [(next_state, 1.0)] if dynamics are deterministic.
        """
        if action not in self.get_actions(state):
            raise ValueError("Illegal action %s in state %s" % (action, state))

        if action == Direction.EXIT:
            # TODO: Terminal state integer returns corresponding reward or reward for 'Terminal state'?
            return [(self.terminal_state, 1.0)]

        # TODO: Really unsure about terminal state situation
        next_state = self.attempt_to_move_in_direction(state, action)
        return [(next_state, 1.0)]

    def attempt_to_move_in_direction(self, state, action):
        """Return the new state an agent would be in if it took the action."""
        assert type(action) == int
        new_state = action
        return new_state   # new state is action





########################################################
class NStateMdpHardcodedFeatures(NStateMdp):
    def get_features(self,state):
        """Outputs np array of features - a one-hot encoding of the state.
        """
        # features = np.zeros(self.num_states)
        # features[state] = 1
        features = np.zeros(2)
        if state == 0:
            features[0] = 1
        elif state == 1 and self.num_states == 3:   # if three states (test env), cut out state 1
            pass
        elif state == 1 and self.num_states == 2:
            features[1] = 1; features[0] = 1
        elif state == 2:
            features[1] = 1
        else:
            raise ValueError('should only have three states for this featurization')
        return features





########################################################
class NStateMdpGaussianFeatures(NStateMdp):
    """
    Features for each state are drawn from the same Gaussian for all states. The map: state \mapsto features is deterministic.

    Additional variables:
    -num_states_reachable: Integer k <= N which we may change between training and test MDP.
    -SEED
    """
    def __init__(self, num_states, rewards, start_state, preterminal_states, feature_dim, num_states_reachable, SEED=1):
        self.SEED = SEED
        super(NStateMdpGaussianFeatures, self).__init__(num_states, rewards, start_state, preterminal_states)
        self.feature_dim = feature_dim
        self.num_states_reachable = num_states_reachable
        self.populate_features()
        self.feature_matrix_nonlin = populate_nonlin_features(self.feature_matrix)
        self.type = 'bandits'
        # self.populate_reward

    def populate_features(self):
        """Draws each state's features from a Gaussian and stores them in a dictionary and matrix."""
        self.features = {}
        np.random.seed(self.SEED)
        self.SEED += 1  # Ensures different features for each new MDP
        mean = np.zeros(self.feature_dim)
        # cov = invwishart.rvs(df=self.feature_dim, scale=np.ones(self.feature_dim), size=1)
        cov = np.eye(self.feature_dim)
        self.feature_matrix = np.zeros([self.num_states, self.feature_dim])
        for state in self.get_states():
            features = multivariate_normal.rvs(mean=mean,cov=cov,size=1)
            self.features[state] = features
            self.feature_matrix[state,:] = np.array(features)

    def get_features(self, state):
        return self.features[state]

    def get_actions(self, state):
        """Returns available actions except ones that lead to unreachable states"""
        actions = super(NStateMdpGaussianFeatures, self).get_actions(state)
        if Direction.EXIT in actions:
            return actions
        else:   # TODO: Why only two actions in debugger when there could be 3?
            return actions[:self.num_states_reachable]

    def add_feature_map(self, feature_dict):
        """Adds a feature map that overwrites the one from self.populate_features.
        This makes sure that a test MDP can have the same feature map as the training MDP.
        """
        self.feature_dim = feature_dict.copy()

    def convert_to_numpy_input(self, use_nonlinear_features=False):
        """Encodes this MDP in a format well-suited for deep models. Gives a matrix from states to features."""
        if use_nonlinear_features:
           return self.feature_matrix_nonlin
        else:
            return self.feature_matrix

class NStateMdpRandomGaussianFeatures(NStateMdp):
    """
    Features for each state are drawn from a different Gaussian for each state. The map: state \mapsto features is stochastic.

    Additional variables:
    -num_states_reachable: Integer k <= N which we may change between training and test MDP.
    -SEED
    """
    def __init__(self, num_states, rewards, start_state, preterminal_states, feature_dim, num_states_reachable, SEED=1):
        # Superclass init:
        # super(NStateMdp, self).__init__(num_states, rewards, start_state, preterminal_states)
        self.num_states = num_states
        self.terminal_state = 'Terminal State'
        self.preterminal_states = preterminal_states    # Preterminal states should include states with no available actions. Otherwise get_actions==>[]==>
        self.start_state = start_state
        self.rewards = np.array(rewards)
        # Additional for this class
        self.SEED = SEED
        self.feature_dim = feature_dim
        self.num_states_reachable = num_states_reachable
        self.populate_features()
        self.type = 'bandits'

    def populate_features(self):
        """Draws each state's feature DISTRIBUTION PARAMETERS from an Inv Wishard and stores them in a
        dictionary self.feature_params: state -> feature parameters."""
        self.feature_params = {}
        np.random.seed(self.SEED)
        self.SEED += 1  # Ensures different features for each new MDP
        mean_hyperprior = np.zeros(self.feature_dim)
        cov_hyperprior = np.eye(self.feature_dim)

        self.feature_matrix_mean = np.zeros([self.num_states, self.feature_dim])
        for state in self.get_states():
            mean = multivariate_normal.rvs(mean=mean_hyperprior, cov=cov_hyperprior)
            cov = invwishart.rvs(df=self.feature_dim, scale=np.ones(self.feature_dim), size=1)
            self.feature_params[state] = (mean, cov)
            self.feature_matrix_mean[state,:] = np.array(mean)



    def get_features(self, state):
        """Draws features(state) from the Gaussian corresponding to the state."""
        (mean, cov) = self.feature_params[state]
        features = multivariate_normal.rvs(mean, cov)
        return features

    def add_feature_map(self, feature_dict):
        """Adds a feature map that overwrites the one from self.populate_features.
        This makes sure that a test MDP can have the same feature map as the training MDP.
        """
        raise NotImplementedError

    def convert_to_numpy_input(self):
        """Encodes this MDP in a format well-suited for deep models."""
        return self.feature_matrix_mean



########################################################
class GridworldMdp(Mdp):
    """A grid world where the objective is to navigate to one of many rewards.

    Specifies all of the static information that an agent has access to when
    playing in the given grid world, including the state space, action space,
    transition probabilities, rewards, start space, etc.

    Once an agent arrives at a state with a reward, the agent must take the EXIT
    action which will give it the reward. In any other state, the agent can take
    any of the four cardinal directions as an action, getting a living reward
    (typically negative in order to incentivize shorter paths).
    """

    def __init__(self, grid, args, living_reward=-0.01, noise=0):
        """Initializes the MDP.

        grid: A sequence of sequences of spaces, representing a grid of a
        certain height and width. See assert_valid_grid for details on the grid
        format.
        living_reward: The reward obtained when taking any action besides EXIT.
        noise: Probability that when the agent takes a non-EXIT action (that is,
        a cardinal direction), it instead moves in one of the two adjacent
        cardinal directions.

        Raises: AssertionError if the grid is invalid.
        """
        self.assert_valid_grid(grid)
        self.height = len(grid)
        self.width = len(grid[0])
        self.living_reward = living_reward
        self.noise = noise
        self.terminal_state = 'Terminal State'

        self.walls = [[space == 'X' for space in row] for row in grid]
        self.type = 'gridworld'
        self.args = args
        # self.populate_rewards_and_start_state(grid)


    def assert_valid_grid(self, grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a wall), ' ' (representing
        an empty space), 'A' (representing the start state), or a value v so
        that float(v) succeeds (representing a reward).

        Often, grid will be a list of strings, in which case the rewards must be
        single digit positive rewards.
        """
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must all be walls
        for y in range(height):
            assert grid[y][0] == 'X', 'Left border must be a wall'
            assert grid[y][-1] == 'X', 'Right border must be a wall'
        for x in range(width):
            assert grid[0][x] == 'X', 'Top border must be a wall'
            assert grid[-1][x] == 'X', 'Bottom border must be a wall'

        def is_float(element):
            try:
                return float(element) or True
            except ValueError:
                return False

        # An element can be 'X' (a wall), ' ' (empty element), 'A' (the agent),
        # or a value v such that float(v) succeeds and returns a float.
        def is_valid_element(element):
            return element in ['X', ' ', 'A'] or is_float(element)

        all_elements = [element for row in grid for element in row]
        assert all(is_valid_element(element) for element in all_elements), 'Invalid element: must be X, A, blank space, or a number'
        assert all_elements.count('A') == 1, "'A' must be present exactly once"
        floats = [element for element in all_elements if is_float(element)]
        assert len(floats) >= 1, 'There must at least one reward square'

    def populate_rewards_and_start_state(self, grid):
        """Sets self.rewards and self.start_state based on grid.

        Assumes that grid is a valid grid.

        grid: A sequence of sequences of spaces, representing a grid of a
        certain height and width. See assert_valid_grid for details on the grid
        format.
        """
        self.rewards = {}
        self.start_state = None
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] not in ['X', ' ', 'A']:
                    self.rewards[(x, y)] = float(grid[y][x])
                elif grid[y][x] == 'A':
                    self.start_state = (x, y)

    def get_random_start_state(self):
        """Returns a state that would be a legal start state for an agent.

        Avoids walls and reward/exit states.

        Returns: Randomly chosen state (x, y).
        """
        y = random.randint(1, self.height - 2)
        x = random.randint(1, self.width - 2)
        while self.walls[y][x] or (x, y) in self.rewards:
            y = random.randint(1, self.height - 2)
            x = random.randint(1, self.width - 2)
        return (x, y)

    def convert_to_numpy_input(self):
        """Encodes this MDP in a format well-suited for deep models.

        Returns three things -- a grid of indicators for whether or not a wall
        is present, a Numpy array of features, and the start state (a tuple in
        the format x, y).
        """
        walls = np.array(self.walls, dtype=int)
        return walls, self.feature_matrix, self.start_state

    @staticmethod
    def generate_random(args, height, width, pr_wall, feature_dim, goals=None, living_reward=0, noise=0, print_grid=False, decorrelate=False):
        """Generates a random instance of a Gridworld."""

        # Does each goal\object type appear multiple times?
        if args.repeated_obj:
            num_goals = args.num_obj_if_repeated
        else:
            num_goals = feature_dim

        def generate_goals(states):
            goal_pos = np.random.choice(len(states), num_goals, replace=False)
            # Place a 'proxy' goal/object which is in the same position as another goal everywher except once
            if args.repeated_obj:
                placed_proxy = False
                placed_isolated_proxy = False
                goals = []
                for idx in goal_pos:
                    objects = []
                    x, y = states[idx]
                    if not decorrelate:
                        # For object number feature_dim-2, correlate it with object feature_dim with probability p_corr
                        # feature_dim-2 is the proxy
                        obj_type_1 = np.random.choice(feature_dim - 1)
                        if obj_type_1 == feature_dim - 2:
                            placed_proxy = True
                            if placed_isolated_proxy:
                                objects.append(feature_dim-1)
                            else:
                                placed_isolated_proxy = True
                    else:
                        obj_type_1 = np.random.choice(feature_dim)
                        placed_proxy = True
                    objects.append(obj_type_1)
                    goal = (x, y, objects)
                    goals.append(goal)
                if not placed_proxy:
                    # Repeat if proxy not placed (low probability event)
                    try:
                        return generate_goals(states)
                    except RuntimeError:
                        raise ValueError('Number of goals lower than feature dim?')
                return goals
            # Generate goals the normal way
            else:
                goals = [states[i] for i in goal_pos]
                for j, goal in enumerate(goals):
                    goal = (goal[0], goal[1], [j])
                    goals[j] = goal
                return goals

        def generate_start_and_goals(goals):
            start_state = (width // 2, height // 2)
            states = [(x, y) for x in range(1, width-1) for y in range(1, height-1)]
            states.remove(start_state)
            if goals is None:
                goals = generate_goals(states)
            return start_state, goals

        (start_x, start_y), goals = generate_start_and_goals(goals)
        goals_wo_type = [(x, y) for x, y, _ in list(goals)]
        required_nonwalls = list(goals_wo_type)
        required_nonwalls.append((start_x, start_y))

        directions = [
            Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        grid = [['X'] * width for _ in range(height)]
        walls = [(x, y) for x in range(1, width-1) for y in range(1, height-1)]
        dsets = DisjointSets([])
        first_state = required_nonwalls[0]
        for x, y, in required_nonwalls:
            grid[y][x] = ' '
            walls.remove((x, y))
            dsets.add_singleton((x, y))
            dsets.union((x, y), first_state)

        min_free_spots = (1 - pr_wall) * len(walls)
        random.shuffle(walls)
        # Ensure walls don't cage the agent
        while dsets.get_num_elements() < min_free_spots or not dsets.is_connected():
            x, y = walls.pop()
            grid[y][x] = ' '
            dsets.add_singleton((x, y))
            for direction in directions:
                newx, newy = Direction.move_in_direction((x, y), direction)
                if dsets.contains((newx, newy)):
                    dsets.union((x, y), (newx, newy))

        grid[height // 2][width // 2] = 'A'
        for x, y in goals_wo_type:
            grid[y][x] = random.randint(-9, 10)

        # Print grid
        if print_grid:
            for row in grid:
                row_new = []
                for place in row:
                    place = str(place)
                    row_new.append(place)
                print(str(row_new))
        return grid, goals

    def get_start_state(self):
        """Returns the start state."""
        return self.start_state

    def get_states(self):
        """Returns a list of all possible states the agent can be in.

        Note it is not guaranteed that the agent can reach all of these states.
        """
        coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        all_states = [(x, y) for x, y in coords if not self.walls[y][x]]
        all_states.append(self.terminal_state)
        return all_states

    def get_actions(self, state):
        """Returns the list of valid actions for 'state'.

        Note that you cannot request moves into walls. The order in which
        actions are returned is guaranteed to be deterministic, in order to
        allow agents to implement deterministic behavior.
        """
        if self.is_terminal(state):
            return []
        x, y = state
        if self.walls[y][x]:
            return []
        # TODO (soerenmind): Decide when to end episodes if it saves time
        # if state in self.rewards:
        #     return [Direction.EXIT]
        result = []
        for act in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
            next_state = self.attempt_to_move_in_direction(state, act)
            if next_state != state:
                result.append(act)
        return result

    def get_reward(self, state, action):
        """Get reward for state, action transition.

        This is the living reward, except when we take EXIT, in which case we
        return the reward for the current state.
        """
        if state in self.rewards and action == Direction.EXIT:
            return self.rewards[state]
        return self.living_reward


    def is_terminal(self, state):
        """Returns True if the current state is terminal, False otherwise.

        A state is terminal if there are no actions available from it (which
        means that the episode is over).
        """
        return state == self.terminal_state

    def get_transition_states_and_probs(self, state, action):
        """Gets information about possible transitions for the action.

        Returns list of (next_state, prob) pairs representing the states
        reachable from 'state' by taking 'action' along with their transition
        probabilities.
        """
        if action not in self.get_actions(state):
            raise ValueError("Illegal action %s in state %s" % (action, state))

        if action == Direction.EXIT:
            return [(self.terminal_state, 1.0)]

        next_state = self.attempt_to_move_in_direction(state, action)
        if self.noise == 0.0:
            return [(next_state, 1.0)]

        successors = defaultdict(float)
        successors[next_state] += 1.0 - self.noise
        for direction in Direction.get_adjacent_directions(action):
            next_state = self.attempt_to_move_in_direction(state, direction)
            successors[next_state] += (self.noise / 2.0)

        return successors.items()

    def attempt_to_move_in_direction(self, state, action):
        """Return the new state an agent would be in if it took the action.

        Requires: action is in self.get_actions(state).
        """
        x, y = state
        newx, newy = Direction.move_in_direction(state, action)
        return state if self.walls[newy][newx] else (newx, newy)

    def __str__(self):
        """Returns a string representation of this grid world.

        The returned string has a line for every row, and each space is exactly
        one character. These are encoded in the same way as the grid input to
        the constructor -- walls are 'X', empty spaces are ' ', the start state
        is 'A', and rewards are their own values. However, rewards like 3.5 or
        -9 cannot be represented with a single character. Such rewards are
        encoded as 'R' (if positive) or 'N' (if negative).
        """
        def get_char(x, y):
            if self.walls[y][x]:
                return 'X'
            elif type(self.rewards) == type({}) and (x, y) in self.rewards:
                reward = self.rewards[(x, y)]
                # Convert to an int if it would not lose information
                reward = int(reward) if int(reward) == reward else reward
                posneg_char = 'R' if reward >= 0 else 'N'
                reward_str = str(reward)
                return reward_str if len(reward_str) == 1 else posneg_char
            elif (x, y) == self.get_start_state():
                return 'A'
            else:
                return ' '

        def get_row_str(y):
            return ''.join([get_char(x, y) for x in range(self.width)])

        return '\n'.join([get_row_str(y) for y in range(self.height)])



class GridworldMdpWithFeatures(GridworldMdp):
    """
    Same as GridWorldMdp, but there is a feature map and the reward is a linear function of the features.
    """
    def __init__(self, grid, args, living_reward=-0.01, noise=0):
        super(GridworldMdpWithFeatures, self).__init__(grid, args, living_reward, noise)
        self.grid = grid
        self.feature_dim = args.feature_dim
        self.populate_features()
        if args.nonlinear:
            self.feature_matrix_nonlin = populate_nonlin_features(self.feature_matrix)

    def get_features(self,state):
        """Returns feature vector for state"""
        x, y = state
        return self.feature_matrix[y,x,:]

    def populate_features(self):
        raise NotImplementedError

    def get_reward(self, state, action):
        features = self.get_features(state)
        return np.dot(features, self.rewards)

class GridworldMdpWithDistanceFeatures(GridworldMdpWithFeatures):
    """Features are based on distance to places with reward."""
    def __init__(self, grid, goals, args, dist_scale=0.5, living_reward=-0.01, noise=0, rewards=None):
        self.dist_scale = dist_scale
        self.goals = goals
        # self.feature_weights = None
        self.euclid_features = args.euclid_features
        super(GridworldMdpWithDistanceFeatures, self).__init__(
            grid, args, living_reward=-0.01, noise=0)

    def populate_features(self):
        self.populate_features_and_start_state()

    def populate_features_and_start_state(self):
        """Sets self.feature_matrix and self.start_state based on grid.

        Features are saved in self.feature_matrix, which is a 3D Numpy array. They represent the distances**distance_exponent to the
        fixed number of goal locations. Distances are euclidean and the exponent is -1 by default.

        Assumes that grid is a valid grid.

        grid: A sequence of sequences of spaces, representing a grid of a
        certain height and width. See assert_valid_grid for details on the grid
        format.
        """
        # self.feature_weights = []
        self.start_state = None

        # Save start state
        for y in range(len(self.grid)):
            for x in range(len(self.grid[0])):
                # if self.grid[y][x] not in ['X', ' ', 'A']:
                #     # self.feature_weights.append(float(self.grid[y][x]))
                #     self.goals.append((x,y))
                if self.grid[y][x] == 'A':
                    self.start_state = (x, y)


        height, width = len(self.grid), len(self.grid[0])
        self.feature_matrix = np.zeros([height, width, self.feature_dim])
        # Save features for each state based on distance to goals
        for y in range(1,height-1):
            for x in range(1,width-1):
                features = np.zeros(self.feature_dim)
                for i,j,obj_nums in self.goals:
                    distance = np.linalg.norm(np.array((x,y)) - np.array((i,j)))
                    rbf_nearness = np.exp(- self.dist_scale * distance)
                    feat_value = -distance / 5. if self.euclid_features and not self.args.repeated_obj \
                        else rbf_nearness

                    'Featurization for different object types:'
                    for obj_num in obj_nums:
                        features[obj_num] += feat_value

                self.feature_matrix[y,x,:] = features

    def convert_to_numpy_input(self, use_nonlinear_features=False):
        """Encodes this MDP in a format well-suited for deep models.

        Returns three things -- a grid of indicators for whether or not a wall
        is present, a Numpy array of features, and the start state (a tuple in
        the format x, y).
        -use_features_true: the returned feature matrix contains high nonlinear features too.
        """
        feature_matrix_returned = self.feature_matrix_nonlin if use_nonlinear_features else self.feature_matrix
        walls = np.array(self.walls, dtype=int)
        return walls, feature_matrix_returned, self.start_state


class GridworldEnvironment(object):
    """An environment containing a single agent that can take actions.

    The environment keeps track of the current state of the agent, and updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.reset()

    def get_current_state(self):
        return self.state

    def get_actions(self, state):
        return self.gridworld.get_actions(state)

    def perform_action(self, action):
        """Performs the action, updating the state and providing a reward."""
        state = self.get_current_state()
        next_state, reward = self.get_random_next_state(state, action)
        self.state = next_state
        return (next_state, reward)

    def get_random_next_state(self, state, action):
        """Chooses the next state according to T(state, action)."""
        rand = random.random()
        sum = 0.0
        results = self.gridworld.get_transition_states_and_probs(state, action)
        for next_state, prob in results:
            sum += prob
            if sum > 1.0:
                raise ValueError('Total transition probability more than one.')
            if rand < sum:
                reward = self.gridworld.get_reward(state, action)
                return (next_state, reward)
        raise ValueError('Total transition probability less than one.')

    def reset(self):
        """Resets the environment. Does NOT reset the agent."""
        self.state = self.gridworld.get_start_state()

    def is_done(self):
        """Returns True if the episode is over and the agent cannot act."""
        return self.gridworld.is_terminal(self.get_current_state())



class Direction(object):
    """A class that contains the five actions available in Gridworlds.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)
    # This is hacky, but we do want to ensure that EXIT is distinct from the
    # other actions, and so we define it here instead of in an Action class.
    EXIT = 'exit'
    INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST, EXIT]
    DIRECTION_TO_INDEX = { a:i for i, a in enumerate(INDEX_TO_DIRECTION) }
    ALL_DIRECTIONS = INDEX_TO_DIRECTION

    @staticmethod
    def move_in_direction(point, direction):
        """Takes a step in the given direction and returns the new point.

        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions, except not Direction.EXIT.
        """
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def get_adjacent_directions(direction):
        """Returns the directions within 90 degrees of the given direction.

        direction: One of the Directions, except not Direction.EXIT.
        """
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return [Direction.EAST, Direction.WEST]
        elif direction in [Direction.EAST, Direction.WEST]:
            return [Direction.NORTH, Direction.SOUTH]
        raise ValueError('Invalid direction: %s' % direction)

    @staticmethod
    def get_number_from_direction(direction):
        return Direction.DIRECTION_TO_INDEX[direction]

    @staticmethod
    def get_direction_from_number(number):
        return Direction.INDEX_TO_DIRECTION[number]


def populate_nonlin_features(feature_matrix):
    """Given a feature map, computes all its polynomials up to degree 2 and stores them in a matrix."""
    states_shape, dim = feature_matrix.shape[:-1], feature_matrix.shape[-1]
    num_states = reduce(mul, states_shape, 1)
    quadratic_dim = ((dim + 1) * (dim + 2)) // 2 - 1
    upper_triangle_indices = np.triu_indices(dim + 1)

    feature_matrix = np.reshape(feature_matrix, [num_states, dim])
    feature_matrix_nonlin = np.zeros([num_states, quadratic_dim])
    for state in range(num_states):
        features = np.concatenate([[1],feature_matrix[state,:]])
        # Compute all quadratic features
        quadratic_features = np.outer(features,features)
        # Keep only the upper triangle and flatten, in order to drop duplicates
        quadratic_features = quadratic_features[upper_triangle_indices]
        # Drop the first (constant) feature
        feature_matrix_nonlin[state,:] = quadratic_features[1:]
    return np.reshape(feature_matrix_nonlin, list(states_shape) + [quadratic_dim])
