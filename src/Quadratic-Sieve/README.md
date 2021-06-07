# Quadratic-Sieve

https://en.wikipedia.org/wiki/Quadratic_sieve

An implementation of the quadratic sieve algorithm for integer factorization. Gathers congruences using multiple processes, each using multiple polynomial quadratic sieve, and then solves the system of equations to find a factor.

Usage:  

To use a single process for factorizing n:  

` pypy mpqs.py n `  

To use p processes:  

` pypy mpqs.py n p `  

To print progress of each process and keep printing estimated time remaining:  

` pypy mpqs.py n p 1 `  

Example: 

``` 
$ time pypy mpqs.py 1000000000100000000000000000074000000005700000000000000000969 4
Factorizing  61 digit number
Size of factor base ->  5459
Collecting data in parallel using 4 processes
Time taken to gather data ->  47.0951719284 seconds
Solving equations ...
Time taken to solve equations ->  17.254873991 seconds
1000000000100000000000000000074000000005700000000000000000969 = 1000000000000000000000000000057^1 * 1000000000100000000000000000017^1

real	1m4.559s
user	3m16.606s
sys	0m0.365s
```
