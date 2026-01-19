import main as simulation
import kernels as kernel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.animation import FFMpegWriter
from enum import Enum
import random
import itertools
from dataclasses import dataclass
from typing import Callable, Dict

if  __name__ == '__main__':
  print("Running test simulation...")
  
  reduce_travel = simulation.SIRsimulation(
    travel_prob=0.1,
    waning_recovery=True,
    travel_TF = True,
    rules={400:simulation.reduce_travel}   
    )
  
  reduce_travel.initialize_density(kernel.multi_negative_gaussian(reduce_travel.gridsize, [(40,40), (160,160)], 30), proc_points=0.4)
  reduce_travel.add_infected(5)
  reduce_travel.run()

  a = reduce_travel.animate()
  # a.save("reduce_travel_multi_gaussian.gif", writer=PillowWriter(fps=10))
  a.save("reduce_travel_multi_gaussian.mp4", writer=FFMpegWriter(fps=10))
  

  increase_infection_prob = simulation.SIRsimulation(
    waning_recovery=True,
    travel_TF = True,
    rules={400:simulation.increase_infection_prob}   
    )
  
  increase_infection_prob.initialize_density(kernel.multi_negative_gaussian(increase_infection_prob.gridsize, [(40,40), (160,160)], 30), proc_points=0.4)
  increase_infection_prob.add_infected(5)
  increase_infection_prob.run()

  b = increase_infection_prob.animate()
  b.save("increase_infection_prob_multi_gaussian.mp4", writer=FFMpegWriter(fps=10))
  

