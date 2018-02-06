import numpy as np
from copy import copy, deepcopy

# class Environment_No_State:
#     def __init__(self,s_terminal=np.array([]),horizon=10,s_start=0):
#         '''
#         Environment without a self.state variable. State is handled by agent.
#
#         Inputs:
#         -rfunc: 1D array of size |S| or dim(feature space)
#         -s_terminal: list of terminal states, ending episode
#         -horizon: int (after which episode terminates)
#         -steps_taken: int
#         -s_start: starting state array or int
#         '''
#         self.s_terminal = s_terminal    # terminal states array
#         self.horizon = horizon
#         self.steps_taken = 0
#         self.s_start = s_start
#     def add_rfunc(self,rfunc):
#         self.rfunc = rfunc
#     def get_next_state(self,s,a):
#         pass
#     def get_reward(self,s,a):
#         pass
#     def get_next_state_reward(self,s,a):
#         return self.get_next_state(s,a), self.get_reward(s,a)
#     def get_next_state_reward_update(self,s,a):
#         '''Outpus next r and s and updates steps taken'''
#         self.steps_taken += 1
#         return self.get_next_state_reward(s,a)
#     def is_terminal(self,s):
#         assert type(self.s_terminal) == list
#         if type(s) == int:
#             terminal = s in self.s_terminal
#         else:   # s assumed to be array
#             terminal = any((s == x).all() for x in self.s_terminal)
#         return terminal or self.steps_taken >= self.horizon - 1
#     def get_num_states(self):
#         return self.rfunc.shape[0]
#     def get_rfunc(self):
#         assert isinstance(self.rfunc,np.ndarray)
#         # if not self.rfunc:
#         #     EnvironmentError('env missing reward function')
#         return self.rfunc
#
#
#
#
# class N_State_Env(Environment_No_State):
#     '''Environment with n states and n actions, which choose the state directly.'''
#     def get_next_state(self,s,a):
#         if not isinstance(a,int):
#             print(Warning('a should be int'))
#         return a
#     def get_reward(self,s,a):
#         return self.rfunc[s]
#     def get_avg_reward(self,feature_expectations):
#         return np.dot(feature_expectations,self.rfunc)



# class Environment:
#     def __init__(self,s_terminal=np.array([]),horizon=10,s_start=0):
#         '''Inputs:
#         -s_terminal: 1D array of terminal states, ending episode
#         -horizon: int (after which episode terminates)
#         -steps_taken: int
#         -s_start: ?D starting state
#         '''
#         self.s_terminal = s_terminal    # terminal states array
#         self.horizon = horizon
#         self.steps_taken = 0
#         self.s_start = s_start
#     def get_current_state(self):
#         return self.state
#     def get_possible_actions(self):
#         pass
#     def add_rfunc(self,rfunc):
#         self.rfunc = rfunc
#     def get_next_state(self,a):
#         pass
#     def reset(self):
#         pass
#     def is_terminal(self,s):
#         return s in self.s_terminal or self.steps_taken >= self.horizon - 1
#     def get_reward(self,s,a):
#         pass
#     def get_next_state_reward(self,s,a):
#         return self.get_next_state(s,a), self.get_reward(s,a)
#     def get_next_state_reward_update(self,s,a):
#         self.steps_taken += 1
#         return self.get_next_state_reward(s,a)
#     def get_num_states(self):
#         return self.rfunc.shape[0]
#     def get_rfunc(self):
#         assert isinstance(self.rfunc,np.ndarray)
#         # if not self.rfunc:
#         #     EnvironmentError('env missing reward function')
#         return self.rfunc

# class Basic_Grid_World(Environment_No_State):
#     def __init__(self,height, width, s_terminal=np.array([]),horizon=10,s_start=0):
#         '''Inputs:
#         -s_terminal: 1D array of terminal states, ending episode
#         -horizon: int (after which episode terminates)
#         -steps_taken: int
#         -s_start: ?D starting state
#         '''
#         self.s_terminal = s_terminal    # terminal states array
#         self.horizon = horizon
#         self.steps_taken = 0
#         self.s_start = s_start
#         self.height = height
#         self.width = width
#     # def get_possible_actions(self):
#     #     return (1,2,3,4)
#     def get_reward(self,s,a):
#         (x,y) = s
#         return self.rfunc[x][y]
#     def get_next_state(self,state,a):
#         # state = self.get_current_state()
#         if a == 1:
#             if state[1] == self.height:
#                 return state.copy()
#             else:
#                 return state + np.array([0,1])
#         if a == 2:
#             if state[0] == self.width:
#                 return state.copy()
#             else:
#                 return state + np.array([1,0])
#         if a == 3:
#             if state[1] == 0:
#                 return state
#             else:
#                 return state + np.array([0,-1])
#         if a == 4:
#             if state[0] == 0:
#                 return state
#             else:
#                 return state + np.array([-1,0])
#         else: raise ValueError('action should be 1,2,3 or 4')
#     def reset(self):
#         self.state = np.array([0,0])
#     #TODO: How to compute reward?
