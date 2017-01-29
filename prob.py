from __future__ import print_function
import math
fact = math.factorial
from decimal import *
getcontext().prec = 8

def n_choose_k(n, k):
    a = Decimal(fact(n))
    b = Decimal(fact(k))
    c = Decimal(fact(n-k))
    return a / b / c

# Probability to guess k items from n binary items,
# where p is the probability to guess one binary value
def fit_prob(p, n, k):
    p = Decimal(p)
    n = Decimal(n)
    k = Decimal(k)
    s = Decimal(0)
    for i in range(k,n):
        s = s + n_choose_k(n,i) * p**i * (1-p)**(n-i)
    print(s)

def test1():
    print("5 choose 2 =", n_choose_k(5, 2))
    print("500 choose 200 =", n_choose_k(500, 200))

if __name__ == '__main__':
    #fit_prob(0.5, 5000, 3500)
    fit_prob(0.5, 168, 134)

# TeX Formula:
# probability = \sum_{i=k}^{n} \binom{n}{i} \cdot p^{i} \cdot (1-p)^{n-i}


