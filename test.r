library("MCMCpack")

#' Computes the ELBO for the linear regression example
#'
#' param y univariate outcome variable
#' param x univariate predictor variable
#' param beta_mu mean of the variational density for \beta
#' param beta_sd standard deviation of the variational density for \beta
#' param nu parameter of the variational density for \sigma^2
#' param nr_samples number of samples for the Monte carlo integration
#' returns ELBO
compute_elbo <- function(y, x, beta_mu, beta_sd, nu, tau2, nr_samples = 1e4) {
  n <- length(y)
  sum_y2 <- sum(y^2)
  sum_x2 <- sum(x^2)
  sum_yx <- sum(x * y)

  # Takes a function and computes its expectation with respect to q(\beta)
  E_q_beta <- function(fn) {
    integrate(function(beta) {
      dnorm(beta, beta_mu, beta_sd) * fn(beta)
    }, -Inf, Inf)$value
  }

  # Takes a function and computes its expectation with respect to q(\sigma^2)
  E_q_sigma2 <- function(fn) {
    integrate(function(sigma) {
      dinvgamma(sigma^2, (n + 1) / 2, nu) * fn(sigma)
    }, 0, Inf)$value
  }


  # Compute expectations of log p(\sigma^2)
  E_log_p_sigma2 <- E_q_sigma2(function(sigma) log(1 / sigma^2))

  # Compute expectations of log p(\beta \mid \sigma^2)
  E_log_p_beta <- (
    log(tau2 / beta_sd^2) * E_q_sigma2(function(sigma) log(sigma^2)) +
      (beta_sd^2 + tau2) / (tau2) * E_q_sigma2(function(sigma) 1 / sigma^2)
  )

  # Compute expectations of the log variational densities q(\beta)
  E_log_q_beta <- E_q_beta(function(beta) dnorm(beta, beta_mu, beta_sd, log = TRUE))
  # E_log_q_sigma2 <- E_q_sigma2(function(x) log(dinvgamma(x, (n + 1)/2, nu))) # fails

  # Compute expectations of the log variational densities q(\sigma^2)
  sigma2 <- rinvgamma(nr_samples, (n + 1) / 2, nu)
  E_log_q_sigma2 <- mean(log(dinvgamma(sigma2, (n + 1) / 2, nu)))


  # Compute the expected log likelihood
  E_log_y_b <- sum_y2 - 2 * sum_yx * beta_mu + (beta_sd^2 + beta_mu^2) * sum_x2
  E_log_y_sigma2 <- E_q_sigma2(function(sigma) log(sigma^2) * 1 / sigma^2)
  E_log_y <- n / 4 * log(2 * pi) * E_log_y_b * E_log_y_sigma2


  # Compute and return the ELBO
  ELBO <- E_log_y + E_log_p_beta + E_log_p_sigma2 - E_log_q_beta - E_log_q_sigma2
  ELBO
}

#' Implements CAVI for the linear regression example
#'
#' param y univariate outcome variable
#' param x univariate predictor variable
#' param tau2 prior variance for the standardized effect size
#' returns parameters for the variational densities and ELBO
lmcavi <- function(y, x, tau2, nr_samples = 1e5, epsilon = 1e-2) {
  n <- length(y)
  sum_y2 <- sum(y^2)
  sum_x2 <- sum(x^2)
  sum_yx <- sum(x * y)

  # is not being updated through variational inference!
  beta_mu <- sum_yx / (sum_x2 + 1 / tau2)

  res <- list()
  res[["nu"]] <- 5
  res[["beta_mu"]] <- beta_mu
  res[["beta_sd"]] <- 1
  res[["ELBO"]] <- 0

  j <- 1
  has_converged <- function(x, y) abs(x - y) < epsilon
  ELBO <- compute_elbo(y, x, beta_mu, 1, 5, tau2, nr_samples = nr_samples)

  # while the ELBO has not converged
  while (!has_converged(res[["ELBO"]][j], ELBO)) {

    nu_prev <- res[["nu"]][j]
    beta_sd_prev <- res[["beta_sd"]][j]

    # used in the update of beta_sd and nu
    E_qA <- sum_y2 - 2 * sum_yx * beta_mu + (beta_sd_prev^2 + beta_mu^2) *
      (sum_x2 + 1 / tau2)

    # update the variational parameters for sigma2 and beta
    nu <- 1 / 2 * E_qA
    beta_sd <- sqrt(((n + 1) / E_qA) / (sum_x2 + 1 / tau2))

    # update results object
    res[["nu"]] <- c(res[["nu"]], nu)
    res[["beta_sd"]] <- c(res[["beta_sd"]], beta_sd)
    res[["ELBO"]] <- c(res[["ELBO"]], ELBO)

    # compute new ELBO
    j <- j + 1
    ELBO <- compute_elbo(y, x, beta_mu, beta_sd, nu, tau2, nr_samples = nr_samples)
  }

  res
}

gen_dat <- function(n, beta, sigma) {
  x <- rnorm(n)
  y <- 0 + beta * x + rnorm(n, 0, sigma)
  data.frame(x = x, y = y)
}

set.seed(1)
dat <- gen_dat(100, 0.30, 1)

mc <- lmcavi(dat$y, dat$x, tau2 = 0.50^2)
mc