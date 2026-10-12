import random
import math
import numpy as np
from functools import partial

# Toy example of compositional MCMC sampling
# p has nonzero probability of generating error token E which leads to "off the rails repetition" of all 1's: ...E111111
# q has very low probability of generating off the rails repeated 1's
# p and q typically sample from uniform random sequences otherwise
# p * q should avoid sampling E and 1
# Naively adding the logits does not avoid the error token, but MCMC sampling from p * q does.

def error_token_p(prefix, T):
    # If prefix contains error token 'E', next token is always '1'.
    # tokens: {1, 2, ... ,T, E}
    if 'E' in prefix:
        output_probs = {"E": 0.0}
        for i in range(1, T+1):
            output_probs[str(i)] = 0.0
        output_probs["1"] = 1.0
        return output_probs
    else:
        # No 'E' in prefix, so we have 1/T for E, half the remainder for 0 and 1.
        p_e = 1.0 / (T+1)
        output_probs = {'E': p_e}
        for i in range(1, T+1):
            output_probs[str(i)] = p_e
        return output_probs

def error_token_q(prefix, T):
    # Exponentially low probability of generating '1'
    alpha = math.exp(-T)  # Probability of '1'
    p_others = 1.0 - alpha
    output_probs = {"E": p_others/T}
    for i in range(1, T+1):
        output_probs[str(i)] = p_others/T
    output_probs["1"] = alpha
    return output_probs

def normalize(dist):
    norm_dist = {}
    norm_Z = sum([dist[key] for key in dist.keys()])
    for key in dist.keys():
        norm_dist[key] = dist[key]*1/norm_Z
    return norm_dist

def set_product(output_p, output_q):
    output_pq = {}
    for key in output_p.keys():
        output_pq[key] = output_p[key]*output_q[key]
    return output_pq

def sample_autoregressive(next_token_dist, T, seq_len=None):
    if seq_len is None:
        seq_len = 2 * T

    prefix = []
    for _ in range(seq_len):
        dist = next_token_dist(prefix, T)  # distribution for the next token
        tokens = list(dist.keys())
        probs  = list(dist.values())
        # sample from the distribution
        chosen_token = random.choices(tokens, weights=probs, k=1)[0]
        prefix.append(chosen_token)
    return prefix



def naive_composition(p, q, T, context=None, seq_len=None):
    if seq_len is None:
        seq_len = 2 * T

    if context == None:
        context = []
        prefix = context.copy()
    else:
        prefix = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []
    for _ in range(seq_len - len(context)):
        dist_unnorm = set_product(p(prefix, T), q(prefix, T)) # naive product
        dist = normalize(dist_unnorm)
        tokens = list(dist.keys())
        probs  = list(dist.values())
        # sample from the distribution
        chosen_token = random.choices(tokens, weights=probs, k=1)[0]
        prefix.append(chosen_token)
        log_probs_norm.append(np.log(dist[chosen_token]))
        log_probs_unnorm.append(np.log(dist_unnorm[chosen_token]))

    return prefix, log_probs_norm, log_probs_unnorm






def compositional_sampler(p, q, mcmc_steps, T, context=[], seq_len=None):
    if seq_len is None:
        seq_len = 2 * T

    c = len(context)

    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    for _ in range(seq_len - c):
        dist_unnorm = set_product(p(gen, T), q(gen, T)) # naive product
        dist = normalize(dist_unnorm)
        tokens = list(dist.keys())
        probs  = list(dist.values())
        # sample from the distribution
        chosen_token = random.choices(tokens, weights=probs, k=1)[0]
        gen.append(chosen_token)
        log_probs_norm.append(np.log(dist[chosen_token]))
        log_probs_unnorm.append(np.log(dist_unnorm[chosen_token]))


        for _ in range(mcmc_steps):
            t = len(gen)
            idx = random.randint(c, t-1)

            prop, log_prob_prop, target_log_prob_prop = naive_composition(p, q, T, gen[:idx], seq_len=t)
            log_prob_cur = log_probs_norm.copy()[idx-c:]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:]

            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            assert(len(gen) == len(prop) == t)
            if np.random.rand() < np.exp(log_r):
                assert(len(gen) == len(prop))
                gen = prop.copy()
                assert(len(log_probs_norm[idx-c:]) == len(log_prob_prop))
                assert(len(log_probs_unnorm[idx-c:]) == len(target_log_prob_cur))
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

    return gen, log_probs_norm, log_probs_unnorm



            

N = 20

for _ in range(N):
    naive, _, _ = naive_composition(error_token_p, error_token_q, T=20)
    print("|".join(naive))

print("-----------------")


for _ in range(N):
    gen, _, _ = compositional_sampler(error_token_p, error_token_q, mcmc_steps=20, T=20, seq_len=None)
    print("|".join(gen))


