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
                 stop_threshold: float, 
                 prior: float,
                 m: float):
        super().__init__(epsilon = epsilon,
                         stop_threshold = stop_threshold,
                         prior = prior,
                         m = m)
        self.n_per_round = n_per_round
        self.binomial_experiment_gen = ExperimentGen(rng)
        self.round_binomial_experiment: Optional[BinomialExperiment] = None
    
    # BinomialExperimenter implementation
    def get_experiment_data(self) -> Optional[BinomialExperiment]:
        return self.round_binomial_experiment
    
    # CredenceBasedSupervisor mandatory method implementations
    def _stop_action(self):
        self.round_binomial_experiment = None
        self._finally()
    
    def _continue_action(self):
        self._experiment(self.n_per_round, self.epsilon)
        self._finally()
        
    def _finally(self):
        self.jeffrey_update_credence()
        
    def _experiment(self, n: int, epsilon):
        self.round_binomial_experiment = self.binomial_experiment_gen.experiment(n, epsilon)
    