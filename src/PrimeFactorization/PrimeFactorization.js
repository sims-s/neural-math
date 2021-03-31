/**
 * Reduces the integer n into a product of prime factors
 * and returns a mapping from the prime factor to its multiplicity.
 * Example: 40 = 2^3 * 5 would return { 2: 3, 5: 1 }
 */
function primeFactorize(n) {
  const primeFactors = {};
  let primeFactor = 0;

  function pushPrimeFactor() {
    primeFactors[primeFactor] = (primeFactors[primeFactor] || 0) + 1;
  }

  let i = 2;
  while (i <= n / i) {
    if (n % i === 0) {
      primeFactor = i;
      pushPrimeFactor();
      n /= i;
    } else {
      i++;
    }
  }

  if (primeFactor < n) primeFactor = n;
  pushPrimeFactor();

  return primeFactors;
}
