import numpy as np
from numpy.lib.function_base import select
from scipy.stats import truncnorm
from sympy import multipledispatch
from sympy.ntheory.generate import primepi


"""Experimental, not quite there yet
Idea: instead of just factoring all the numbers at runtime, we can generate composite #s from a list of primes
Doing that is implemented here, but ugly and doesn't really beat sympy on speed for #s even up to 2**32
So not working on it for now...
"""

primes = np.load('data/primes_2B.npy', mmap_mode='r')
primes_less_than_2973 = primes[:429]


def log(num, base=np.e):
    return np.log(num)/np.log(base)

def compute_number_distinct_factors(max_factors, mean, std):
    result = truncnorm((1-mean)/std, (max_factors-mean)/std, loc=mean, scale=std).rvs(1)
    result = np.rint(result)
    return int(result[0])

def one_pm_point_two_over_log_squared(x, sign=1):
    return 1 + sign*.2/(log(x)**2)
def p_minus_one_over_p_product(arr):
    return np.prod((arr - 1)/arr)
def p_over_p_minus_one_product(arr):
    return np.prod(arr/(arr-1))


def compute_lower_product_bound(lower, upper):
    first_term = np.exp(-np.euler_gamma)/log(upper)*one_pm_point_two_over_log_squared(upper, sign=-1)
    if lower < 2973:
        terms_to_multiply = primes_less_than_2973[primes_less_than_2973 <= lower]
        second_term = p_over_p_minus_one_product(terms_to_multiply)
    else:
        second_term = np.exp(np.euler_gamma)*log(lower)*one_pm_point_two_over_log_squared(lower, sign=-1)
    return first_term * second_term


def compute_upper_product_bound(lower, upper):
    first_term = np.exp(-np.euler_gamma)/log(upper)*one_pm_point_two_over_log_squared(upper)
    if lower < 2973:
        terms_to_multiply = primes_less_than_2973[primes_less_than_2973 <= lower]
        second_term = p_over_p_minus_one_product(terms_to_multiply)
    else:
        second_term = np.exp(np.euler_gamma)*log(lower)*(one_pm_point_two_over_log_squared(lower))
    return first_term * second_term
    

def compute_product_bounds(lower, upper):
    if upper < 2973:
        terms_to_multiply = primes_less_than_2973[(primes_less_than_2973 >= lower) & (primes_less_than_2973 <= upper)]
        product = p_minus_one_over_p_product(terms_to_multiply)
        return product, product
    
    lb = compute_lower_product_bound(lower, upper)
    ub = compute_upper_product_bound(lower, upper)
    
    return lb, ub


def compute_interval_probability(lower, upper):
    lb, ub = compute_product_bounds(lower, upper)
    ub = min(ub, 1)
    return 1-(lb + ub)/2

def select_prime(neighborhood, exact_primepi):
    # since we have primes less than 2973 already and it's small, when we have small values get them "exactly"
    # by computing the closest prime to the neightborhood
    if neighborhood < 2973:
        index = np.argmin(np.abs(primes_less_than_2973 - neighborhood))
    elif neighborhood < exact_primepi:
        index = int(primepi(neighborhood))
    else:
        x = neighborhood
        index = np.random.randint(x//(log(x)-1), x//(log(x)-1.1)+1)
    return primes[index]

def select_small_primes_small_special_case(lb, ub):
    primes_in_interval = primes_less_than_2973[(primes_less_than_2973>=lb) & (primes_less_than_2973<=ub)]
    probs = 1/primes_in_interval
    probs = probs / np.sum(probs)
    return np.random.choice(primes_in_interval, p=probs)

def sample_prime(lb, ub, binary_convergence_thresh, larger_factor_scaling, exact_primepi):
    while (ub - lb) > binary_convergence_thresh:
        # print(lb, ub)
        midpoint = (lb + ub) // 2
        lower_prob = compute_interval_probability(lb, midpoint)
        upper_prob = larger_factor_scaling*compute_interval_probability(midpoint, ub)
        lower_plus_upper = lower_prob + upper_prob
        lower_norm_prob = lower_prob/lower_plus_upper

        # print(lower_norm_prob)
        # print('='*100)
        if np.random.rand() < lower_norm_prob:
            ub = midpoint
        else:
            lb = midpoint
    lb, ub = int(lb), int(ub)
    if ub < 2973:
        return select_small_primes_small_special_case(lb, ub)
    else:
        position = np.random.choice(lb + np.arange(ub-lb))
    # position = np.random.choice(lb + np.arange(ub-lb))
    prime = select_prime(position, exact_primepi)
    return prime



"""Algorithm for generating composites that are reasonably distributed given a list of primes

Let the composite that we produce be X

1. Determine how many distinct factors the number has
    https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93Kac_theorem
    Number of distinct prime factors follows a normal distribution

    To quote the article: It is very difficult if not impossible to discover the Erdös-Kac theorem empirically, 
    as the Gaussian only shows up when n starts getting to be around 10^100

    Therefore, don't use the theorem correctly because it's not useful... Instead just sample from a normal distribution
    with prespecifried mean and standard deviation. 
    Also truncated beacuse all numbers have at least one prime factor

2. Once we have the number of prime factors, we need to sample each of them.
    To sample them, I use binary search.
    Given the upper bound and lower bound of numbers that we're looking to compute, split the interval in half
    For each half of the interval, compute the probability that X has a prime factor in that interval:
        Assume that each of the prime factors are independent
        For each p in the interval the probability that p divides X is 1/p since X should be a random number
            So the probability that p doesn't divide x is (p-1)/p=1-1/p
        So the probability that X divides any number in the interval is the probability that it divides at least one
        Which is one minus the probability that no numbers divide X
        Which is equal to 1 - product over primes in interval of (1-1/p)
        We can use Theorem 6.12 in https://arxiv.org/pdf/1002.0442.pdf to estimate lower & upper bounds on those intervals
            Note that the paper specifies for bounds for all primes less than some upper bound not primes in an interval
            But the paper also gives bounds on the reciporcal
            So the bounds for the interval is equal to (reciporcal bounds for 0...lb)*bounds for (0...ub)
            So just multiply both lower bounds to get a lower bound on the interval and same with upper
            And then normalize so we can sample from them
        also to control the difficulty of the problem, there's a paramater, larger_factor_scaling that will scale the upper probability
            so it's more likely to generate larger numbers. e.g. if lower/upper is 50/50, and scale = 1.5 then the probabilities
            will be .5/1.25 for lower and .75/1.25 for upper
    Once we have that estimated probability, decide whether or not to take the upper or lower interval by sampling with probs
    Repeat until we've reached a small enough interval
    Once we have an interval, we need to choose a prime approximately from that interval
    First reduce the interval to a number by picking somewhere in it uniformly at random. Call that point N (N = neighborhood)
    Cases for picking the exact value N:
        1. N < 2973:
            We already have those numbers stored, so it's easy to work with and things are most sensitive when N is small
            We're going to pick the closest prime number to N and use that as our factor
        2. Otherwise we're going to use the prime counting function of N and take the prime at that index. Two cases:
            1. N < 2**16: In this case we're going to compute the prime counting function of N exactly
            2. Otherwise, we're going to use the upper bound here in theorem 5.1: https://www.sci-hub.se/10.1007/s11139-016-9839-4
                It's an upper bound, not a lower + upper bound, but emperically it's pretty close, and I didn't notice any nice tight lower bounds
                This'll make the primes a bit bigger on average, but that's ok

    Once we have each of the numbers, we need to compute the multiplicty
    Since X is a random number, the we can compute the probability of p being a factor with multiplicity K as 
        (p-1)/(p^(k+1))
        Or at least I think it is?
        Overall though it seems like we're producing numbers with lower multiplicities than we lshould
        So scale the probabilities of being multiplicitiy >=2 by a certain factor
    So we just sample according to this probability to generate all the multiplicities of primes

    Now, if we end up creating a prime number, then there is the option to resample it randomly (true by defualt)
        we'd want to do this to avoid generating the same small primes repeatedly
        Since we're much more likely to sample from the smallest numbers
        And multplicity of 1 is pretty likely, we'll end up hitting those primes a lot
        So we manually resample the primes so that they're more interesting if we want
    
"""

def compute_multiplicty(num, upper_bound, multiplicity_multiplier):
    max_factors = max(int(log(upper_bound, num)), 1)
    probs = num*np.ones(max_factors)
    valid_multiplicities = 1 + np.arange(max_factors)
    probs = (probs-1)/np.power(probs, valid_multiplicities)
    probs[1:] = probs[1:]*multiplicity_multiplier
    probs = probs / probs.sum()
    multiplicity = np.random.choice(valid_multiplicities, p=probs)
    return multiplicity

def generate_number(smallest_prime_idx = 0, upper_bound=2**32, n_factors_mean=3, n_factors_std=1.5, binary_convergence_thresh=100,
                    larger_factor_scaling=1, exact_primepi=2**16, uniform_prime=True, multiplicity_multiplier = 10):
    if binary_convergence_thresh > 2973:
        raise ValueError("Binary convergence thresh is assumed to be less than or lequal to 2973. Got %d"%binary_convergence_thresh)
    max_factors = log(upper_bound, primes[smallest_prime_idx])
    num_distinct_factors = compute_number_distinct_factors(max_factors, n_factors_mean, n_factors_std)
    # num_distinct_factors = 1
    # np.random.seed(8)
    values = [sample_prime(primes[smallest_prime_idx], 
                        upper_bound, 
                        binary_convergence_thresh, 
                        larger_factor_scaling, 
                        exact_primepi) for i in range(num_distinct_factors)]
    multiplicities = [compute_multiplicty(n, upper_bound, multiplicity_multiplier) for n in values]

    if uniform_prime and len(values)==1 and multiplicities[0]==1:
        values = [np.random.choice(primes)]
        while values[0] > upper_bound:
            values = [np.random.choice(primes)]
    # print(values, multiplicities)
    return values, multiplicities



if __name__ == "__main__":
    # generate_number()
    # sys.exit()
    from tqdm.auto import tqdm
    qty = 1000
    pbar = tqdm(total = qty)
    res = []
    for i in range(qty):
        v, m = generate_number()
        # res.append(max(m) > 1)
        res.append(m[0]==1 and len(m)==1)
        pbar.update(1)
    print(np.mean(res))