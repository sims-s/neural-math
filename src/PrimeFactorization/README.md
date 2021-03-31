# Taken from: https://github.com/mikeyaworski/PrimeFactorization

# PrimeFactorization

## Overview

This calculates the prime factorization of numbers in multiple languages (Java, JavaSript, Python). This is well optimized code and should calculate results of large numbers very quickly.

## Methods

### Java


- `HashMap<Long, Long> primeFactorize(long n)` Reduces the parameter n into a product of only prime numbers and returns a mapping of the prime factor to its multiplicity. Uses the long type to do all of the calculations, which limits the maximum value to prime factorize to a little over 9 quintillion (2^63 - 1).
- `HashMap<BigInteger, BigInteger> primeBigFactorize(BigInteger n)` Reduces the parameter n into a product of only prime numbers and returns a mapping of the prime factor to its multiplicity. Uses the BigInteger class to do all of the calculations, which has no theoretical limit for the value to prime factorize.

### JavaScript

- `primeFactorize(n)` Reduces the parameter n into a product of only prime numbers and returns a mapping of the prime factor to its multiplicity

### Python

- `primeFactorize(n)` Reduces the parameter n into a product of only prime numbers and returns a mapping of the prime factor to its multiplicity

## Usage

Just download these file(s) and paste it/them directly into your project(s).

## Examples

### Java

For the long method:

```java
PrimeFactorization.primeFactorize(40) // { 2: 3, 5: 1 }
```

This will fail when trying to prime factorize a number larger than 2 billion (2^31 - 1) because it is outside the range of an int literal. So, to combat against this, the use of a BigInteger can be used, or just represent it as a long by using L at the end of the number:

```java
// add this import statement
import java.math.BigInteger;
        
PrimeFactorization.primeFactorize(new BigInteger("600851475143").longValue()) // { 71: 1, 839: 1, 1471: 1, 6857: 1 }
PrimeFactorization.primeFactorize(600851475143L) // { 71: 1, 839: 1, 1471: 1, 6857: 1 }
```

That will allow you to use the method on values larger than 2 billion (outside the int literal range), but still has to be less than 9 quintillion (maximum value of long) because even though we are using BigInteger, which has no theoretical maximum or minimum value, we are using `longValue()` on the BigInteger.

For the BigInteger method (notice how there is never any conversion to long or int):

```java
PrimeFactorization.primeBigFactorize(new BigInteger("600851475143")) // { 71: 1, 839: 1, 1471: 1, 6857: 1 }
```

### JavaScript
```javascript
console.log(primeFactorize(40)); // { 2: 3, 5: 1 }
```

### Python
```python
print(primeFactorize(40)) # { 2: 3, 5: 1 }
```
