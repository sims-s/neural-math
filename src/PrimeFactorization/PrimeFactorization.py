'''
  Reduces the integer n into a product of prime factors
  and returns a mapping from the prime factor to its multiplicity.
  Example: 40 = 2^3 * 5 would return { 2: 3, 5: 1 }
'''
def primeFactorize(n):
  primeFactors = {}
  primeFactor = 0
  i = 2

  while i <= n / i:
    if n % i == 0:
      primeFactor = i
      primeFactors[primeFactor] = primeFactors.get(primeFactor, 0) + 1
      n /= i
    else:
      i += 1

  if primeFactor < n: primeFactor = int(n)
  primeFactors[primeFactor] = primeFactors.get(primeFactor, 0) + 1

  return primeFactors
