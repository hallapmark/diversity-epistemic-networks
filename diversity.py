from sim.simsetup import *
from multiprocessing import freeze_support

def main():
    # Run a comparison of (a) a lifecycle sim with no skeptic; (b) a lifecycle sim
    # with a skeptic.

    sim_count = 1000
    # sim_count is standardly 10000 in the Zollman (2007) literature.
    # It is 1000 in O'Connor and Weatherall 2018, and in Weatherall, O'Connor and Bruner (2020).

    sim_type = ENSimType.LIFECYCLE
    simsetup = ENSimSetup(sim_count, sim_type)
    simsetup.quick_setup()
    print("\nBaseline lifecycle sims done\n")

    sim_type = ENSimType.LIFECYCLE_W_SKEPTICS
    simsetup = ENSimSetup(sim_count, sim_type)
    simsetup.quick_setup()
    print("\nLifecycle with skeptics sims done\n")

    sim_type = ENSimType.LIFECYCLE_W_ALTERNATOR_SKEPTICS
    simsetup = ENSimSetup(sim_count, sim_type)
    simsetup.quick_setup()
    print("\nLifecycle with alternator skeptics sims done\n")

if __name__ == "__main__":
    freeze_support()
    # https://docs.python-guide.org/shipping/freezing/
    main()
