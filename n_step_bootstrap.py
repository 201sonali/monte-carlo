import itertools
from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    # copy initial V
    V = initV.copy()

    for t in trajs:
        # upper limit for timestpes
        timesteps = 2 ** 30
        # return, rewards, and states
        G_acc = 0
        reward_arr = []
        state_arr = []

        for count, (s, a, rn, sn) in enumerate(t):
            # reward
            reward_arr.append(rn)
            # initial state
            if count == 0:
                state_arr.append(s)
            # state transition
            state_arr.append(sn)
            # break when you reach high number of timesteps
            if count >= timesteps - 1:
                break

            # starting time step for n-step update
            tau = count - n + 1

            # updating reward (G_acc) and state-value (V) based on tau
            if tau > 0:
                G_acc -= reward_arr[tau - 1]
            G_acc /= env_spec.gamma
            if tau + n <= timesteps:
                G_acc += np.power(env_spec.gamma, n - 1) * reward_arr[tau + n - 1]
            if tau + n < timesteps:
                G = G_acc + np.power(env_spec.gamma, n) * V[state_arr[tau + n]]
            else:
                G = G_acc
            if tau >= 0:
                V[state_arr[tau]] += alpha * (G - V[state_arr[tau]])
            if tau >= timesteps - 1:
                break

            # increment count
            count += 1

    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    Q = initQ.copy()
    pi_mat = np.ones((env_spec.nS, env_spec.nA)) / env_spec.nA

    for t in trajs:
        # upper limit for timestpes
        timesteps = 2**30
        # return, importance sampling ratio, rewards, states, actions
        G_acc = 0
        rho_acc = 1
        reward_arr = []
        state_arr = []
        action_arr = []

        for count in range(timesteps):
            if count < timesteps:
                try:
                    s, a, rn, sn = t[count]
                    reward_arr.append(rn)
                    action_arr.append(a)
                    if count == 0:
                        state_arr.append(s)
                    state_arr.append(sn)
                except IndexError:
                    timesteps = count
                    break


        # empty list to store the action probability ratios
        pi2bpi_arr = []
        # initial state and action
        initial_state = state_arr[0]
        initial_action = action_arr[0]
        # action probability ratio for the initial state-action pair
        initial_ratio = pi_mat[initial_state, initial_action] / bpi.action_prob(initial_state, initial_action)
        # add initial ratio
        pi2bpi_arr.append(initial_ratio)

        for count in itertools.count():
            tau = count - n + 1
            if tau > 0:
                G_acc -= reward_arr[tau - 1]
            # updating rho
            if tau >= 0:
                # if tau is small
                if abs(pi2bpi_arr[tau]) < 1e-05:
                    rho_acc = np.prod(pi2bpi_arr[tau + 1:min(tau + n, timesteps - 1) + 1])
                # otherwise
                else:
                    rho_acc /= pi2bpi_arr[tau]
            # discount return
            G_acc /= env_spec.gamma
            if count < timesteps:
                G_acc += np.power(env_spec.gamma, n - 1) * reward_arr[tau + n - 1]
            if count + 1 <= timesteps - 1:
                pi2bpi_arr.append(
                    pi_mat[state_arr[count + 1], action_arr[count + 1]] / bpi.action_prob(state_arr[count + 1], action_arr[count + 1]))
                rho_acc *= pi2bpi_arr[count + 1]

            if tau + n >= timesteps:
                G = G_acc
            else:
                G = G_acc + np.power(env_spec.gamma, n) * Q[state_arr[tau + n], action_arr[tau + n]]

            if tau >= 0:
                Q[state_arr[tau], action_arr[tau]] += alpha * rho_acc * (G - Q[state_arr[tau], action_arr[tau]])

                # greedy policy update
                s = state_arr[count]
                best_action = Q[s, :].argmax()
                best_action = best_action if np.isscalar(best_action) else best_action[0]
                pi_vec = np.zeros(env_spec.nA)
                pi_vec[best_action] = 1
                pi_mat[s, :] = pi_vec

                if tau >= timesteps - 1:
                    break

    class MyPolicy(Policy):
        def __init__(self, pi_mat):
            self.pi_mat = pi_mat

        def action_prob(self, state: int, action: int) -> float:
            return self.pi_mat[state, action]

        def action(self, state: int) -> int:
            best_action = self.pi_mat[state, :].argmax()
            if np.isscalar(best_action):
                return best_action
            else:
                return best_action[0]

    pi = MyPolicy(pi_mat)

    return Q, pi
