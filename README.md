# Factoring numbers using transformer encoder-decoder model

## Run Training & Evaluation

`python scripts/train_factorization_model.py --config [YOUR_CONFIG.YAML]`  
For an example config, see [this file](config.yaml)


## Observations
Model training can occur in many different bases. E.g. the number 12 can be represented as "[1][1][0][0]" in base 2, "[2][2]" in base 5 or "[12]" in bases larger than 12. (brackets indiciate a single token. i.e. in bases larger than 12, there is a single token for the number 12)
* Model performance is significantly affected by the base of training. When training in prime bases, the model's degrades significantly.    
* Different positional encodings affect the model's performance a lot. 
* Initialization in the attention layers, should use `gain=.5`. Seems to improve performance a little bit.


### Reading metrics.json
(out of date, be wary)
* correct
  * **correct_product : Percent of numbers whose predicted factors have a product equal to the original number.**
  * **correct_factorization: Percent of numbers whose prediction is a correct prime factorization (i.e. have the same product & all predicted factors are prime)**
* beam_accuracy:
  * correct product/factorization as defined above for each beam
* by_prob:
  * correct product/factorization for all beams grouped by the log probability decile of each prediction
  * **percent_prime_factors_pred: For each prediction (i.e. num_beams predictions per number), what is the percent of factors predicted that are prime?**
* by_n_target_factors: What is the correct product/factorization mean grouping by the number of factors in the number that was being factored?. Additionally, have sizes of each bucket.
* by_target_number: Metrics now grouped by the size of the number (decile)
* pred_same_as_target_beam_0:
  * Just looking at the top beam, how often is the model regurgitating the input number when the number is not prime?
  * pred_same_as_target: Whether or not the model outputed the tokens that were inputted
  * target_is_prime: whether the number being factored is prime or not
  * There are 4 entries associated with this (may be fewer if some have a count of 0). The 4 entries are the combinations of the above two features (i.e. T/T, T/F, F/T, F/F)
* by_min_target_factor:
  * Given the prime factorization of the target, take the minimum prime factor, and group it by decile (may be fewer due to abundance of small prime factors)
  * What is the models accuracy based on that? i.e. is the model able to factor numbers that are the product of larger numbers? or is it just good at pulling out factors of 2,3,...

  

