import main as simulation
import kernels as kernel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from enum import Enum
import random
import itertools
from dataclasses import dataclass
from typing import Callable, Dict

if  __name__ == '__main__':
  print("Running test simulation...")
  sim = simulation.SIRsimulation(
    gridsize=(200,200),
    infection_radius=1,
    step_threshold=400,
    average_infection_time=5,
    infection_probability=0.15, #up
    infection_time_variance=2,
    average_recovered_time = 20, 
    recovered_time_variance = 4,
    travel_prob=0.1, 
    travel_infection_prob=0.1,
    waning_recovery=True,
    travel_TF = True,
    rules={100:simulation.dec_infection_probability}   
    )
  
  sim.initialize_density(kernel.multi_negative_gaussian(sim.gridsize, [(20,20), (80,80)], 20), proc_points=0.4)
  sim.add_infected(5)
  sim.run()

  a = sim.animate()
  a.save("movie.gif", writer=PillowWriter(fps=10))
  

