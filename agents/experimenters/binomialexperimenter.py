from typing import Protocol, Optional
from capabilities.experimentgen import BinomialExperiment

class BinomialExperimenter(Protocol):
    credence: float
    def get_experiment_data(self) -> Optional[BinomialExperiment]:  
        """ Get the data from the latest experiment, k: int and n: int."""
