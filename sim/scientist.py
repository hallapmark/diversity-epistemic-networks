from __future__ import annotations
import numpy as np
from sim.experimentgen import BinomialExperiment, ExperimentGen
from typing import Optional

from sim.sim_models import ENParams

LOW_STOP = .5

""" A scientist who runs experiments on a binomial distribution, and who stops 
experimenting when credence is below a certain threshold."""
class Scientist(): 
    def __init__(self,
                 prior: float,
                 params: ENParams,
                 rng: np.random.Generator,
                 is_skeptic: bool,
                 is_propagandist: bool = False
                 ):
        if is_skeptic and is_propagandist:
            raise ValueError("A scientist cannot be both a skeptic and a propagandist")
        self.credence = prior
        self.params = params
        self.is_skeptic = is_skeptic
        self.is_propagandist = is_propagandist 
        # Only publishes results that favor the theory 'A is better'
        # Does not actually conduct fraudulent experiments

        self.binomial_experiment_gen = ExperimentGen(rng)
        self.round_binomial_experiment: Optional[BinomialExperiment] = None
        self.rounds_of_experience = 0
        
        self.influencers: list[Scientist] = []
        # Influencers can include self
    
    def __str__(self):
        k = self.round_binomial_experiment.k if self.round_binomial_experiment else 'N/A'
        n = self.round_binomial_experiment.n if self.round_binomial_experiment else 'N/A'
        return f"credence = {round(self.credence, 3)}, k = {k}, n = {n}"
    
    # Public interface
    def report_experiment_data(self) -> Optional[BinomialExperiment]:
        rbe = self.round_binomial_experiment
        if self.is_propagandist:
            return rbe if rbe and rbe.k / rbe.n < .5 else None 
            # only report if it looks bad for theory B
        return rbe
    
    def decide_round_research_action(self):
        if self.is_propagandist:
            self._experiment(self.params.trials, self.params.epsilon)
            return # Early exit if is propagandist. Propagandist always experiments
        
        # All other agents decide based on their credence, with the constraints listed below
        if self.credence < LOW_STOP:
            self.round_binomial_experiment = None
        elif self.params.alternator and self.credence == .5:
            try_B: bool = np.random.choice([True, False])
            if try_B:
                self._experiment(self.params.trials, self.params.epsilon)
            else:
                self.round_binomial_experiment = None
        else:
            self._experiment(self.params.trials, self.params.epsilon)

    def add_jeffrey_influencer(self, influencer: Scientist):
        self.influencers.append(influencer)

    def jeffrey_update_credence(self):
        if self.is_skeptic or self.is_propagandist:
            return
        for influencer in self.influencers:
            self._jeffrey_update_credence_on_influencer(influencer)

    def dm(self, influencer: Scientist) -> float:
        d = abs(self.credence - influencer.credence)
        return d * self.params.m

    # Private methods   
    def _experiment(self, n: int, epsilon):
        self.round_binomial_experiment = self.binomial_experiment_gen.experiment(n, epsilon)
    
    def _jeffrey_update_credence_on_influencer(self, influencer: Scientist): 
        exp = influencer.report_experiment_data()
        if exp:
            k = exp.k
            n = exp.n
            p = 0.5 + self.params.epsilon
            p_E_H = self._truncated_likelihood(k, n, p)
            p_E_nH = self._truncated_p_E_nH(k, n, p)
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

    # P(E|H) when some terms cancel out in the denominator and numerator of Bayes' theorem
    def _truncated_likelihood(self, k: int, n: int, p: float) -> float:
        """ Calculate likelihood using a simplified formula (some terms cancel out from the 
        denominator and numerator)."""
        return p ** k * (1 - p) ** (n - k)

    def _truncated_p_E_nH(self, k: int, n: int, p: float) -> float:
        """ Calculate P(E|~H) for the binomial distribution when there are only two possible
        parameter values/two possible worlds: p and 1-p."""
        return (1-p) ** k * p ** (n - k)
    