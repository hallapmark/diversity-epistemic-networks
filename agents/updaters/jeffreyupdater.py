from agents.abstractagents.doxasticagent import DoxasticAgent
from agents.experimenters.binomialexperimenter import BinomialExperimenter

# Following Zollman (2007) and Weatherall, O'Connor and Bruner (2018), this
# implementation assumes that there are only two competing hypotheses/possible worlds:
# H := p = 0.5 + epsilon and not-H := p = 0.5 - epsilon where p is the probability that
# a binary event happens. There is alternative literature that
# works with a full distribution of hypotheses (see also beta distribution, and Zollman 2010.)

""" A Jeffrey updater who knows how to update on binomial distributions."""
class JeffreyUpdater(DoxasticAgent):
    def __init__(self, epsilon: float, m: float, **kw):
        super().__init__(**kw)
        self.epsilon = epsilon # How much better theory B is. p = 0.5 + epsilon
        # Influencers can include self
        self.m = m
        self.jeffrey_influencers: list[BinomialExperimenter] = []
        
    def add_jeffrey_influencer(self, influencer: BinomialExperimenter):
        self.jeffrey_influencers.append(influencer)

    # Public interface
    # Superclass mandatory method implementation
    def jeffrey_update_credence(self):
        for influencer in self.jeffrey_influencers:
            self._jeffrey_update_credence_on_influencer(influencer)

    def dm(self, influencer: BinomialExperimenter) -> float:
        d = abs(self.credence - influencer.credence)
        return d * self.m

    # Private methods
    def _jeffrey_update_credence_on_influencer(self, influencer: BinomialExperimenter): 
        exp = influencer.get_experiment_data()
        if exp:
            k = exp.k
            n = exp.n
            p = 0.5 + self.epsilon
            p_E_H = self._truncated_likelihood(k, n, p)
            p_E_nH = self.truncated_p_E_nH(k, n, p)
            p_E = self._marginal_likelihood(self.credence, p_E_H, p_E_nH)
            p_H_E = self.credence * p_E_H / p_E
            p_H_nE = self.credence * (1 - p_E_H) / (1 - p_E)
            dm = self.dm(influencer)
            # No anti-updating, simply ignore evidence past certain point
            posterior_p_E = 1 - min(1, dm) * (1 - p_E)
            self.credence = self._jeffrey_calculate_posterior(self.credence, p_H_E, posterior_p_E, p_H_nE)

    def _jeffrey_calculate_posterior(self, 
                                     prior: float, 
                                     p_H_E: float, 
                                     posterior_p_E,
                                     p_H_nE: float) -> float:
        """ TIt is assumed that there 
        are only two possible parameter values (two possible worlds): p and 1-p. This parameter gives
        the probability of a "success" event occurring on a given try. """
        # This formula looks bonkers but it is derived from the full Bayes' formula (as applied to binomials).
        # This speeds up the simulations.
        if prior > 0:
            return p_H_E * posterior_p_E + p_H_nE * (1 - posterior_p_E) 
        else:
            return 0
    
    # P(E)
    def _marginal_likelihood(self,
                             prior: float,
                             p_E_H: float,
                             p_E_nH: float) -> float:
        return prior * p_E_H + (1 - prior) * p_E_nH


    # P(E|H)
    def _likelihood(self, k, n, p, factorial_func) -> float:
        """ Use the full likelihood formula if you are either ignoring the denominator
        in Bayes' theorem or if you need to calculate P(E|H) independently for some
        reason. Otherwise, use the truncated likelihood function."""
        f = factorial_func
        return f(n) / (f(k) * f(n-k)) * p ** k * (1 - p) ** (n - k)

    # P(E|H) when some terms cancel out in the denominator and numerator of Bayes' theorem
    def _truncated_likelihood(self, k: int, n: int, p: float) -> float:
        """ Use this only if you are calculating the posterior, i.e. you are
        not ignoring the denominator. You can use the fact that some terms cancel
        out from the denominator and numerator to simplify the likelihood formula."""
        # Note: This simplifies even further in Bayes' theorem. But we can use this function
        # in Jeffrey updating.
        return p ** k * (1 - p) ** (n - k)

    def truncated_p_E_nH(self, k: int, n: int, p: float) -> float:
        """ Calculate P(E|~H) for the binomial distribution when there are only two possible
        parameter values/two possible worlds: p and 1-p."""
        return (1-p) ** k * p ** (n - k)
