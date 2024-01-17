from typing import Tuple
import numpy as np
from env import EnvWithModel
from policy import Policy

def value_prediction(env:EnvWithModel, pi:Policy, initV:np.array, theta:float) -> Tuple[np.array,np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """
    # action probability matrix
    action_prob_mat = np.array([[pi.action_prob(s, a) for s in range(env.spec.nS)] for a in range(env.spec.nA)])

    # copy initial V
    V = initV.copy()

    # run until error is small enough
    while True:
        error = 0
        # iterate through the states in the environment
        for s in range(env.spec.nS):
            # value for state s
            v = V[s]
            # action probabilities for state s
            pi_vec = action_prob_mat[:, s]
            # trans matrix for state s
            trans_mat = env.TD[s, :, :]
            # reward matrix for state s
            reward_mat = env.R[s, :, :]
            # expected returns for all state transitions
            return_mat = pi_vec[:, np.newaxis] * trans_mat * (reward_mat + env.spec.gamma * V[np.newaxis, :])
            # save the new value for s
            V[s] = return_mat.sum()
            # updating error
            error = max(error, np.abs(V[s] - v))

        # when the error is smaller than theta, save the state and calc action-val
        if error < theta:
            break

    # estimate action-value from value prediction above
    Q = (env.TD * (env.R + env.spec.gamma * V[np.newaxis, np.newaxis, :])).sum(axis=-1)

    return V, Q



def value_iteration(env: EnvWithModel, initV: np.array, theta: float) -> Tuple[np.array, Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """
    # copy initial V
    V = initV.copy()

    # run until error is small enough
    while True:
        error = 0
        # iterate through the states in the environment
        for s in range(env.spec.nS):
            # value for state s
            v = V[s]
            # trans matrix for state s
            trans_mat = env.TD[s, :, :]
            # rewards matrix for state s
            reward_mat = env.R[s, :, :]
            # expected returns for all state transitions
            return_mat = (trans_mat * (reward_mat + env.spec.gamma * V[np.newaxis, :])).sum(axis=1).max()
            # save the new value for s
            V[s] = return_mat.sum()
            # updating error
            error = max(error, np.abs(V[s] - v))

        # when the error is smaller than theta, save the state and calc action-val
        if error < theta:
            break

    # instance of the my policy class
    class MyPolicy(Policy):
        def __init__(self, pi_mat):
            self.pi_mat = pi_mat

        def action_prob(self, state: int, action: int) -> float:
            return self.pi_mat[state, action]

        def action(self, state: int) -> int:
            return self.pi_mat[state, :].argmax()

    # create policy matrix based on this
    pi_mat = (env.TD * (env.R + env.spec.gamma * V[np.newaxis, np.newaxis, :])).sum(axis=2)
    pi = MyPolicy(pi_mat)

    return V, pi