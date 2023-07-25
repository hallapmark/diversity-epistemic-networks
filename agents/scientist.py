import numpy as np
from agents.crsupervisor import CredenceBasedSupervisor
from agents.experimenters.binomialexperimenter import BinomialExperimenter
from agents.updaters.jeffreyupdater import JeffreyUpdater
from capabilities.experimentgen import BinomialExperiment, ExperimentGen
from typing import Optional

""" A scientist who runs experiments on a binomial distribution, and who stops 
experimenting when credence is below a certain threshold."""
class Scientist(JeffreyUpdater, CredenceBasedSupervisor, BinomialExperimenter): 
    def __init__(self, 
                 rng: np.random.Generator, 
                 n_per_round: int, 
                 epsilon: float, 
                 low_stop: float,
                 prior: float,
                 m: float):
        super().__init__(epsilon = epsilon,
                         m = m,
                         prior = prior,
                         low_stop = low_stop)
        self.n_per_round = n_per_round
        self.binomial_experiment_gen = ExperimentGen(rng)
        self.round_binomial_experiment: Optional[BinomialExperiment] = None
    
    def __str__(self):
        k = self.round_binomial_experiment.k if self.round_binomial_experiment else 'N/A'
        n = self.round_binomial_experiment.n if self.round_binomial_experiment else 'N/A'
        return f"credence = {round(self.credence, 2)}, k = {k}, n = {n}"

    # BinomialExperimenter implementation
    def get_experiment_data(self) -> Optional[BinomialExperiment]:
        return self.round_binomial_experiment
    
    # CredenceBasedSupervisor mandatory method implementations
    def _stop_action(self):
        self.round_binomial_experiment = None
    
    def _continue_action(self):
        self._experiment(self.n_per_round, self.epsilon)
        
    def _experiment(self, n: int, epsilon):
        self.round_binomial_experiment = self.binomial_experiment_gen.experiment(n, epsilon)
    