import random
import numpy as np
from main import SIRsimulation, PersonState
from kernels import wall, multi_negative_gaussian
from matplotlib.animation import PillowWriter

def vaccinate_rollout(sim):
    """
    Each time this rule is called, randomly vaccinate a small fraction of susceptibles.
    Vaccination = set to recovered in the *current* frame (t), so it carries into t+1 via step().
    """
    t = sim.step_count
    frame = sim.history[t]

    susceptible_coords = np.argwhere(frame == PersonState.susceptible.value)
    if len(susceptible_coords) == 0:
        return

    k = min(120, len(susceptible_coords))  # vaccinate up to 120 per intervention step
    chosen_idx = np.random.choice(len(susceptible_coords), size=k, replace=False)
    chosen = susceptible_coords[chosen_idx]

    for x, y in chosen:
        frame[x, y] = PersonState.recovered.value
        if sim.waning_recovery:
            sim.recovery_timers[x, y] = random.gauss(sim.average_recovered_time, sim.recovered_time_variance)

    sim.history[t] = frame

def main():
    sim = SIRsimulation(
        gridsize=(100, 100),
        infection_radius=2,
        step_threshold=250,
        average_infection_time=5,
        infection_probability=0.12,
        infection_time_variance=2,
        average_recovered_time=40,     # vaccine immunity duration (if waning_recovery=True)
        recovered_time_variance=8,
        travel_prob=0.10,
        travel_infection_prob=0.10,
        waning_recovery=False,         # set True if you want vaccine immunity to wane
        travel_TF=True,
        rules={
            40: vaccinate_rollout,
            60: vaccinate_rollout,
            80: vaccinate_rollout,
            100: vaccinate_rollout,
            120: vaccinate_rollout
        }
    )

    sim.initialize_density(wall(sim.gridsize, 5, 40, 10, 20),n_points=50)
    sim.initialize_density(multi_negative_gaussian(sim.gridsize, [(10,10), (30,30), (25,25)], 10), n_points=2000)
    sim.add_infected(1)

    sim.run()
    a = sim.animate()
    a.save("vaccination_rollout.gif", writer=PillowWriter(fps=10))

if __name__ == "__main__":
    main()
