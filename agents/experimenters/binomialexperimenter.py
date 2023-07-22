from typing import Protocol, Optional
from capabilities.experimentgen import BinomialExperiment

class BinomialExperimenter(Protocol):
    """ Protocol. def get_experiment_data(self) -> Optional[BinomialExperiment]:"""
    def get_experiment_data(self) -> Optional[BinomialExperiment]:  
        """ Get the data from the latest experiment, k: int and n: int."""
    
    def get_credence(self) -> float:  # type: ignore
        """ Get the agent's credence. """
    