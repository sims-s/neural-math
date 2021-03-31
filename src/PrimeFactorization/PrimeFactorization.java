import java.util.HashMap;
import java.math.BigInteger;

/**
  * This class is used to calculate the prime factorization of any positive integer.
  *
  * @author Michael Yaworski of http://mikeyaworski.com
  * @version April 13, 2020
  */
public class PrimeFactorization {
  /**
    * Helper method to add a prime factor to a mapping of factors => multiplicities.
    */
  private static void setPrimeFactor(HashMap<Long, Long> primeFactors, Long primeFactor) {
    Long multiplicity = primeFactors.containsKey(primeFactor) ? primeFactors.get(primeFactor) : 0;
    primeFactors.put(primeFactor, multiplicity + 1);
  }

  /**
    * Helper method to add a prime factor to a mapping of factors => multiplicities.
    */
  private static void setBigPrimeFactor(HashMap<BigInteger, BigInteger> primeFactors, BigInteger primeFactor) {
    BigInteger multiplicity = primeFactors.containsKey(primeFactor) ? primeFactors.get(primeFactor) : BigInteger.ZERO;
    primeFactors.put(primeFactor, multiplicity.add(BigInteger.ONE));
  }

  /**
    * Reduces the integer n into a product of prime factors
    * and returns a mapping from the prime factor to its multiplicity.
    *
    * @param n The BigInteger number to prime factorize.
    * @return A HashMap where each key is a prime factor and each value is its multiplicty.
    */
  public static HashMap<Long, Long> primeFactorize(long n) {
    HashMap<Long, Long> primeFactors = new HashMap<>();
    Long multiplicity;
    long primeFactor = 0L;
    long i = 2L;

    while (i <= n / i) { // smallest prime factor to the square root of n (largest possible factor of n)
      if (n % i == 0) { // the prime number i is a factor of n (i will never go into n if it's composite since the prime factor of that compositite number would have already been tested)
        primeFactor = i; // therefore, this is a prime factor of n
        setPrimeFactor(primeFactors, primeFactor);
        n /= i; // divide out that prime factor from n to get the rest of the prime factors
        // don't increment i: test if this same prime factor goes into n multiple times (e.g. 18 = 2*3*3)
      } else {
        i++; // i is not a prime factor of n, so increment
      }
    }

    // n had no more prime factors, so n is a prime factor
    // else, n was divided down to 1, meaning that the last prime factor divided itself out. therefore, it is the last prime factor
    if (primeFactor < n) primeFactor = n;
    setPrimeFactor(primeFactors, primeFactor);
    
    return primeFactors;
  }

  /**
    * Reduces the integer n into a product of prime factors
    * and returns a mapping from the prime factor to its multiplicity.
    *
    * @param n The BigInteger number to prime factorize
    * @return A HashMap where each key is a prime factor and each value is its multiplicty.
    */
  public static HashMap<BigInteger, BigInteger> primeBigFactorize(BigInteger n) {
    HashMap<BigInteger, BigInteger> primeFactors = new HashMap<>();
    BigInteger primeFactor = BigInteger.ZERO;
    BigInteger i = new BigInteger("2");

    while (i.compareTo(n.divide(i)) <= 0) {
      if (n.mod(i).longValue() == 0) {
        primeFactor = i;
        setBigPrimeFactor(primeFactors, primeFactor);
        n = n.divide(i);
      } else {
        i = i.add(BigInteger.ONE);
      }
    }
    
    if (primeFactor.compareTo(n) < 0) primeFactor = n;
    setBigPrimeFactor(primeFactors, primeFactor);
    
    return primeFactors;
  }
}
