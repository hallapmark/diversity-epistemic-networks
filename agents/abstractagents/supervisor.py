from typing import Protocol

class ResearchSupervisor(Protocol):
    def decide_research_action(self):
        """ Decide whether to continue or stop research."""
    
    def _stop_action(self):
        """ Stop research."""
    
    def _continue_action(self):
        """ Continue research."""