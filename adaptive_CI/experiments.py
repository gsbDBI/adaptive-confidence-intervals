import numpy as np
from adaptive_CI.compute import apply_floor, draw, expand

""" Exploration """


def ts_mab_probs(sum, sum2, neff, prev_t, floor_start=0.005, floor_decay=0.0, num_mc=20):
    # ================
    # Thompson sampling agent with prior N(0, 1) and update the posterior mean
    # and var based on data
    # ================

    # agent prior N(0,1)
    K = len(sum)
    Z = np.random.normal(size=(num_mc, K))

    # estimate empirical mean and variance
    mu = sum / np.maximum(neff, 1)
    var = sum2 / np.maximum(neff, 1) - (sum / np.maximum(neff, 1)) ** 2
    # var = 1

    # calculate posterior
    # 1/sigma(n)^2 = 1/sigma(0)^2 + n / sigma^2
    # sigma(n)^2 = 1 / (n / sigma^2 + 1/sigma(0)^2)
    posterior_var = 1 / (neff / var + 1 / 1.0)
    # mu(n) = sigma^2 / (n * sigma(0)^2 + sigma^2) * mu(0) + n*sigma(0)^2 /
    # (n*sigma(0)^2+sigma^2) * mu
    posterior_mean = neff / (var + neff) * mu

    idx = np.argmax(Z * np.sqrt(posterior_var) + posterior_mean, axis=1)
    w_mc = np.array([np.sum(idx == k) for k in range(K)])
    p_mc = w_mc / num_mc

    # assignment probability floor 1/(t)
    probs = apply_floor(p_mc, amin=floor_start / (prev_t + 1) ** floor_decay)

    return probs


def random_mab_probs(K):
    return np.ones(K) / K


def epsgreedy_mab_probs(sum, neff, epsilon=0.1):
    K = len(sum)
    mean = sum / neff
    amax = np.amax(mean)

    argmax = np.where(mean == amax)[0]
    probs = np.zeros(K)
    for i in range(K):
        if i in argmax:
            probs[i] = (1 - epsilon) / len(argmax) + epsilon / K
        else:
            probs[i] = epsilon / K

    return probs


""" Experiments """


def generate_y(truth, dgp, T, K):
    s, b = dgp.split("_")
    b = float(b)
    if s == 'uniform':
        # uniform noise in  [-b,b]
        return np.array(truth) + np.random.uniform(-1, 1, size=(T, K)) * b
    elif s == 'normal':
        # Gaussian noise N(0, b^2)
        return np.array(truth) + np.random.normal(0, 1, size=(T, K)) * b
    elif s == "exp":
        # centered exponential noise with scale parameter b
        return np.array(truth) + np.random.exponential(scale=b, size=(T, K)) - b
    elif s == "lognormal":
        # centered lognormal noise with underlying normal distribution N(0, b^2)")
        return np.array(truth) + np.random.lognormal(mean=0.0, sigma=b, size=(T, K)) - np.exp(b**2 / 2)
    else:
        raise NotImplementedError(
            "Only implemented centered normal/uniform/exponential/lognormal noises.")


def run_mab_experiment(ys,
                       initial=0,
                       floor_start=0.005,
                       floor_decay=0.0,
                       exploration='TS',
                       init_sum=None, init_sum2=None,
                       init_neff=None):  # These last three used for simulating future propensity scores

    T, K = ys.shape
    T0 = initial * K
    arms = np.empty(T, dtype=np.int_)
    rewards = np.empty(T)
    probs = np.empty((T, K))

    # Initialize if at the middle of an experiment
    sum = np.zeros(K) if init_sum is None else init_sum
    sum2 = np.zeros(K) if init_sum2 is None else init_sum2
    neff = np.zeros(K) if init_neff is None else init_neff
    ndraws = np.zeros((T, K))

    for c, t in enumerate(range(T)):

        if t < T0:
            # Run first "batch": deterministically select each arm `initial`
            # times
            p = np.full(K, 1 / K)
            w = t % K
        else:
            if exploration in {'TS', 'TS_exploration'}:
                p = ts_mab_probs(sum, sum2, neff, t, floor_start=floor_start, floor_decay=floor_decay)
                if exploration=='TS_exploration':
                    # exploration sampling
                    p = p * (1 - p)
                    # normalized to 1
                    p = p / np.sum(p)
            elif exploration.startswith('EG'):
                _, epsilon = exploration.split('_')
                p = epsgreedy_mab_probs(sum, neff, epsilon=float(epsilon))
            elif exploration=='RAN':
                p = random_mab_probs(K)
            else:
                raise NotImplementedError('Only implement TS(thompson)/TS_exploration(p(1-p)) / EG(epsilon greedy)/ RAN(random) exploration!')
            w = np.random.choice(K, p=p)

        # TS with Gaussian prior
        sum[w] += ys[t, w]
        sum2[w] += ys[t, w] ** 2
        neff[w] += 1

        arms[t] = w
        rewards[t] = ys[t, w]
        probs[t] = p
        ndraws[t] = neff

    data = {"arms": arms,
            "rewards": rewards,
            "ndraws": ndraws,
            "rewards": rewards,
            "probs": probs}
    return data
