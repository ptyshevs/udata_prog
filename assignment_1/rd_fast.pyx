from libc.math cimport sqrt, ceil, log, sin, cos, M_PI
from libc.stdlib cimport srand
from cpython cimport array
import array
from libc.time cimport time, time_t, clock
cimport cython

cdef extern from "limits.h":
    double RAND_MAX

cdef double two_pi = M_PI * 2;
cdef double z1_cache[1];

cdef:
    double x_n = 0.0;
    double a = 15485863;
    double c = 373587883;
    double m = 573268139;

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef double rand(double seed=0.0):
    """
    Generate sample from Standard Uniform distribution ~ U(0, 1)
    range: [0, 1)
    """
    global x_n, a, b, c;
    if x_n == 0.0 and seed == 0.0:
        x_n = <double>clock();
    x_n = (a * x_n + c) % m;
    return x_n / m;

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef int randint(int low, high=None, double seed=0.0):
    """
    Generate random integer sample from Uniform distribution ~ U(low, high)

    Slightly biased towards smaller values (because of modulo)
    """
    global x_n, a, b, c;
    cdef:
        int out, n;
    if high is None:
        high = low
        low = 0
    n = high - low
    if x_n == 0.0 and seed == 0.0:
        x_n = <double>clock();
    elif seed != 0.0:
        x_n = seed
    x_n = (a * x_n + c) % m;
    return low + <int>x_n % n

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef double gauss(double mu=0.0, double sigma=1.0):
    """
    Generate sample from Gaussian distribution ~ N(μ, σ^2), using
    Box-Muller transformation
    """
    global z1_cache
    cdef:
        double u1, u2, z0, r, k;
    
    if z1_cache[0] != 0:
        z1 = z1_cache[0];
        z1_cache[0] = 0.0;
        return z1 * sigma + mu;
    u1 = rand()
    u2 = rand();
    r = sqrt(-2 * log(u1));
    k = two_pi * u2;
    z0 = r * cos(k);
    z1_cache[0] = r * sin(k);
    # X = Z * σ + μ
    return z0 * sigma + mu

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef array.array exponential(double scale=1.0, size=None):
    """
    Draw samples from an exponential distribution.

    For more information see an example here:
    https://en.wikipedia.org/wiki/Quantile_function
    @param scale: inverse of the rate parameter
    @param size: if None, one sample is produced. Otherwise must be iterable.
                 For example (n, k, m) will produce n * k * m samples
    """
    cdef:
        unsigned int n_samples, i;
    n_samples = 1
    if size is not None:
        for i in size:
            n_samples *= i
    return array.array('d', [scale * -log(rand()) for _ in range(n_samples)])

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cpdef double beta(double alpha=1.0, double shape=1.0):
    """
    Draw sample from Beta distribution

    Note:
        Method is not practically applicable for large alpha and/or beta. More
        general version involves computing Gamma(1, alpha) and Gamma(1, beta)
    Method used: Convolution
    See more:
    https://bit.ly/2CYKTls
    """
    cdef:
        unsigned int n_samples, a, b;
    a = <unsigned int>ceil(alpha);
    b = <unsigned int>ceil(shape);
    n_samples = a + b - 1
    if n_samples == 1:
        return rand()
    else:
        # return i-th order statistic
        samples = sorted([rand() for _ in range(n_samples)])
        return samples[a - 1]

def test():
    print("test is called")

cpdef void set_seed(unsigned int seed=0):
    """
    Set seed for libc.stdlib pseudo-random generator
    """
    if seed == 0:
        srand(time(NULL));
    else:
        srand(seed);