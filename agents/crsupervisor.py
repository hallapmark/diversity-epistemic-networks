from agents.abstractagents.doxasticagent import DoxasticAgent
from abc import abstractmethod

class CredenceBasedSupervisor(DoxasticAgent):
    def __init__(self, low_stop: float, **kw):
        super().__init__(**kw)
        self.low_stop = low_stop

    def decide_round_research_action(self):
        if self.credence < self.low_stop:
            self._stop_action()
        else:
            self._continue_action()
    
    @abstractmethod
    def _stop_action(self):
        raise NotImplementedError 
    
    @abstractmethod
    def _continue_action(self):
        raise NotImplementedError
