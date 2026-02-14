/*
 * IPL Ball-by-Ball Bayesian Categorical Regression Model
 * ======================================================
 * 
 * Framework:
 *   outcome[n] ~ Categorical(theta[n])
 *   theta[n] = softmax(X[n] * beta + alpha)
 *   beta ~ Normal(0, 1)        (weakly informative prior)
 *   alpha ~ Normal(0, 2)       (weakly informative intercept prior)
 *
 * Outcomes (K=7): W, 0, 1, 2, 3, 4, 6
 * Features (P): match context, player form, venue stats, career skills
 *
 * Reference category: outcome index K (last category)
 * We estimate K-1 = 6 sets of coefficients.
 */

data {
  int<lower=1> N;                    // Number of observations (balls)
  int<lower=2> K;                    // Number of outcome categories (7)
  int<lower=1> P;                    // Number of predictor features
  matrix[N, P] X;                    // Feature matrix (standardized)
  array[N] int<lower=1, upper=K> y;  // Observed outcomes (1..K)
}

parameters {
  matrix[P, K-1] beta;              // Regression coefficients (P x 6)
  vector[K-1] alpha;                // Intercepts for K-1 categories
}

model {
  // Priors
  to_vector(beta) ~ normal(0, 1);   // Weakly informative
  alpha ~ normal(0, 2);             // Weakly informative intercepts

  // Likelihood (vectorized for speed)
  for (n in 1:N) {
    vector[K] logits;
    
    // K-1 categories get linear predictor
    logits[1:(K-1)] = (X[n] * beta)' + alpha;
    
    // Reference category (K) gets logit = 0
    logits[K] = 0;

    // Categorical likelihood via softmax
    y[n] ~ categorical_logit(logits);
  }
}

generated quantities {
  // Posterior predictive: predicted probabilities for each observation
  // (Optional: comment out this block to speed up sampling)
  array[N] simplex[K] theta;
  
  for (n in 1:N) {
    vector[K] logits;
    logits[1:(K-1)] = (X[n] * beta)' + alpha;
    logits[K] = 0;
    theta[n] = softmax(logits);
  }
}
