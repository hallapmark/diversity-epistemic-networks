from abc import ABC

class DoxasticAgent(ABC):
    def __init__(self, prior: float):
        self.credence: float = prior