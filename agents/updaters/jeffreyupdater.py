from typing import Optional
from agents.abstractagents.doxasticagent import DoxasticAgent
from agents.experimenters.binomialexperimenter import BinomialExperimenter

# Following Zollman (2007) and Weatherall, O'Connor and Bruner (2018), this
# implementation assumes that there are only two competing hypotheses/possible worlds:
# H := p = 0.5 + epsilon and not-H := p = 0.5 - epsilon where p is the probability that
# a binary event happens. There is alternative literature that
# works with a full distribution of hypotheses (see also beta distribution, and Zollman 2010.)

""" A Jeffrey updater who knows how to update on binomial distributions."""
class JeffreyUpdater(DoxasticAgent):
    # If m == 0, this is the same as a Bayesian updater
    def __init__(self, epsilon: float, m: float, **kw):
        super().__init__(**kw)
        self.epsilon = epsilon # How much better theory B is. p = 0.5 + epsilon
        # Influencers can include self
        self.m = m
        self.influencers: list[BinomialExperimenter] = []

        ## SKEPTIC
        self.skeptic_influencers: list[BinomialExperimenter] = []

        ## INFLUENCE OF SKEPTIC TRACKED
        self.below_min_and_skeptic_attempted_influence = False
        self.skeptic_credence_influence: float = 0
        self.skeptic_credence_directional_influence: float = 0
        # Skeptic's influence this round. A positive value
        # means the skeptic influenced toward the truth, 
        # a negative value that they influenced towards falsehood.
        # E.g. skeptic moved one agent's credence from .3 to .4, and another
        # agent's credence from .5 to .6. Skeptic moved +.2 credence in round.

        self.skeptic_brier_influence: float = 0
        self.skeptic_brier_directional_influence: float = 0
        # Skeptic's influence on brier score. A positive value means
        # skeptic made things worse.

        self.skeptic_lifetime_brier_directional_influence: Optional[float] = None
        # Skeptic's influence through journey from cr <.5 to cr=.5

        
    def add_jeffrey_influencer(self, influencer: BinomialExperimenter):
        self.influencers.append(influencer)

    # Public interface
    # Superclass mandatory method implementation
    def jeffrey_update_credence(self):
        # reset round stats
        self.below_min_and_skeptic_attempted_influence = False
        self.skeptic_credence_influence = 0
        self.skeptic_credence_directional_influence = 0
        self.skeptic_brier_influence = 0
        self.skeptic_brier_directional_influence = 0
        for influencer in self.influencers + self.skeptic_influencers:
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
            old_credence = self.credence
            self.credence = self._jeffrey_calculate_posterior(self.credence, p_H_E, posterior_p_E, p_H_nE)
            ## CALCULATE SKEPTIC'S EFFECT 
            ## on those who take take uninformative action
            if influencer in self.skeptic_influencers and old_credence < .5:
                self.below_min_and_skeptic_attempted_influence = True
                cr_diff = self.credence-old_credence
                self.skeptic_credence_influence += abs(cr_diff)
                self.skeptic_credence_directional_influence += cr_diff

                br_diff = self.brier(self.credence) - self.brier(old_credence)
                self.skeptic_brier_influence += abs(br_diff)
                self.skeptic_brier_directional_influence += br_diff
                if not self.skeptic_lifetime_brier_directional_influence:
                    self.skeptic_lifetime_brier_directional_influence = br_diff
                else:
                    self.skeptic_lifetime_brier_directional_influence += br_diff
    
    def brier(self, credence: float) -> float:
        return (credence - 1)**2
            
    def _jeffrey_calculate_posterior(self, 
                                     prior: float, 
                                     p_H_E: float, 
                                     posterior_p_E,
                                     p_H_nE: float) -> float:
        """ It is assumed that there
        are only two possible parameter values (two possible worlds): p and 1-p. This parameter gives
        the probability of a "success" event occurring on a given try. """
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
