from sim.simsetup import *
from multiprocessing import freeze_support

def main():
    sim_type = ENSimType.ZOLLMAN_COMPLETE
    # sim_count is standardly 10000 in the Zollman (2007) literature.
    # It is 1000 in O'Connor and Weatherall 2018, and in Weatherall, O'Connor and Bruner (2020).
    sim_count = 10000
    simsetup = ENSimSetup(sim_count, sim_type)
    simsetup.quick_setup()

    sim_type = ENSimType.ZOLLMAN_CYCLE
    sim_count = 10000
    simsetup = ENSimSetup(sim_count, sim_type)
    simsetup.quick_setup()

if __name__ == "__main__":
    freeze_support()
    # https://docs.python-guide.org/shipping/freezing/
    main()