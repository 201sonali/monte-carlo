from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################

    # copy Q for action-val aka q vals function
    Q = initQ.copy()
    # 0s in the length of Q for the counts of state-action pairs
    C = np.zeros(Q.shape)

    # loop over trajectories/SARSA tuples
    for t in trajs:
        # counter for cumulative reward aka return
        G = 0
        # iterate in reverse order per MC
        for s, a, rn, sn in list(reversed(t)):
            # updated return = old return * discount + immediate reward
            G = G * env_spec.gamma + rn
            # increment counts (C) and q vals (Q)
            C[s, a] += 1
            Q[s, a] += 1 / C[s, a] * (G - Q[s, a])

    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################

    # copy Q for action-val aka q vals function
    Q = initQ.copy()
    # 0s in the length of Q for the counts of state-action pairs
    C = np.zeros(Q.shape)

    # loop over trajectories/SARSA tuples
    for t in trajs:
        # counter for cumulative reward aka return
        G = 0
        # starting weight
        W = 1
        # iterate in reverse order per MC
        for s, a, rn, sn in list(reversed(t)):
            # updated return = old return * discount + immediate reward
            G = G * env_spec.gamma + rn
            # increment counts (C) and q vals (Q), same as above but scaled by weight
            C[s, a] += W
            Q[s, a] += W / C[s, a] * (G - Q[s, a])
            # updated weight = estimated policy / behavior policy
            W *= pi.action_prob(s, a) / bpi.action_prob(s, a)

            # break if the weight is small, ie 1e-05
            if abs(W) < 1e-05:
                break

    return Q
