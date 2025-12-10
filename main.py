import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from enum import Enum
import random
import itertools
from dataclasses import dataclass
from typing import Callable, Dict


class DensityInitType(Enum):
  POSITIVE = 0 
  NEGATIVE = 1

@dataclass
class DesityConstructor:
  typ:DensityInitType
  fn:Callable # f.ex np.random.uniform or sth like that
  args:Dict # args for above func

class PersonState(Enum):
  susceptible = 0
  infected = 1
  recovered = 2
  empty = 3

class SIRsimulation:
    
    def __init__(self, gridsize, infection_radius, step_threshold, average_infection_time, infection_probability, infection_time_variance, average_recovered_time, recovered_time_variance, timestep=1):
      assert isinstance(gridsize, tuple)
      self.gridsize = gridsize  #size of grid
      self.grid = np.full(gridsize, PersonState.susceptible.value) #initialize grid based off of gridzie and fill with susceptible
      self.infection_timers = np.full(gridsize, 0) #keep track of time for time steps
      self.recovery_timers = np.full(gridsize, 0)
      self.infection_probability = infection_probability # beta
      self.infection_radius = infection_radius #how far an indi can infect
      self.step_threshold = step_threshold #max # of time steps to go through
      self.step_count = 0 #number of time steps
      
      self.history = np.zeros((step_threshold, *gridsize)) #encodes history of each cell(individual) for each time step. x,y,timestep; think 2d w individuals and the z is time steps
      self.average_infection_time = average_infection_time
      self.infection_time_variance = infection_time_variance
      self.average_recovered_time = average_recovered_time
      self.recovered_time_variance = recovered_time_variance
      self.timestep = timestep
    
    def is_not_finished(self): #keep going if havent reach threshold and still have infected
      infected_left_over = np.any(self.grid == PersonState.infected.value)
      return (self.step_count < self.step_threshold) and infected_left_over
    
    def initialize_density(self, kernel, n_points=30):
      t = self.step_count
      cur_matrix = self.history[t]
      assert kernel.shape == cur_matrix.shape
  
      if not hasattr(self, "rng"):
          self.rng = np.random.default_rng()
  
      flat = kernel.astype(float).ravel()
      total = flat.sum()
      if total <= 0:
          raise ValueError("Kernel must have a positive sum to define a distribution.")
      probs = flat / total
  
      idx = self.rng.choice(flat.size, p=probs, size=n_points)
  
      rows, cols = np.unravel_index(idx, kernel.shape)
  
      cur_matrix[rows, cols] = PersonState.empty.value
      self.history[t] = cur_matrix
      print(cur_matrix)
      return np.column_stack((rows, cols))
      
    def add_infected(self, number):
      for i in range(number):
        x = random.randint(0, self.gridsize[0]-1)
        y = random.randint(0, self.gridsize[1]-1)
        self.history[self.step_count][x,y] = PersonState.infected.value
        self.grid[x, y] = PersonState.infected.value
        self.infection_timers[x, y] = random.gauss(self.average_infection_time, self.infection_time_variance)
    
    def run(self):
      while self.step_count < self.step_threshold-2:
        print('step', self.step_count)
        self.step()
        self.save_frame()
    
    
    def get_neighbors(self, x, y): 
      m, n = self.gridsize
      rad = self.infection_radius
      comb_x = [i % m for i in range(x-rad, x+rad+1)] # "%"" operator wraps around the ends of matrix for both x and y
      comb_y = [i % n for i in range(y-rad, y+rad+1)]
      return itertools.product(comb_x, comb_y)

    def step(self):
      t = self.step_count 
      #infection_mask = (self.history[t]==PersonState.infected.value)
      self.history[t+1] = self.history[t]     
      
      # update the ifection and recovery timers by substracting one timestep from each entry, then clip the values, 
      # so they are between 0 and 100
      self.infection_timers = self.infection_timers - self.timestep
      self.infection_timers = self.infection_timers.clip(min=0, max=100)
      self.recovery_timers = self.recovery_timers - self.timestep
      self.recovery_timers = self.recovery_timers.clip(min=0, max=1000)

      # get coordinates of the individuals that are currently infected
      infected_coordinates = np.argwhere(self.history[t] == PersonState.infected.value)
      # loop through all the infected individuals
      for x,y in infected_coordinates:
          # get the neighbors for the infected individual
          neighbors = self.get_neighbors(x,y)
          # loop through the neighbors
          for (nx, ny) in neighbors:
            # if the neighbor is susceptible
            if self.history[t][nx,ny] == PersonState.susceptible.value:
              # draw a random number in (0,1)
              rand_float = random.random()
              # if this number is smaller than the infection probability
              if rand_float <= self.infection_probability:
                # update the neighbors state to infected in the next state and add a timer
                self.history[t+1][nx,ny] = PersonState.infected.value
                self.infection_timers[nx, ny] = random.gauss(self.average_infection_time, self.infection_time_variance)
        
          # if the timer of the infected individual has reached 0
          if self.infection_timers[x,y] == 0:
            # update it's state to be recovered
            self.history[t+1][x,y] = PersonState.recovered.value
            self.recovery_timers[x,y] = random.gauss(self.average_recovered_time, self.recovered_time_variance)
      
      # get coordinates of the individuals that are currently recovered
      recovered_coordinates = np.argwhere(self.history[t] == PersonState.recovered.value)
      # loop through all the recovered individuals
      for x,y in recovered_coordinates:
        # if the individuals recovery timer has reached 0
        if self.recovery_timers[x,y] == 0:
          # update it's state in the next timestep to susceptible
          self.history[t+1][x,y] = PersonState.susceptible.value
          
      self.step_count += 1 

    #A Different method
    #  for (x,y), item in np.ndenumerate(self.grid):
        #Susceptible
     #   m, n = self.gridsize
     #   if item is PersonState.susceptible.value:
     #     infected_neighbors = 0

     #     for dx in range(self.infection_radius + 1):
     #       for dy in range(self.infection_radius + 1):
     #         if dx and dy == 0:
     #           continue
              
     #         nx,ny = x + dx, y + dy
     #         if 0 <= nx < m and 0 <= ny < n:
     #           infected_neighbors += 1
     #
     #     If infected_neigbors > 0:
     #      p = 1 - (1 - self.infection_rate)
     #      if np.random.rand() < p:      
     #        self.grid[x, y] = PersonState.infected
     #        self_timer[x, y] = 0

     #     elif infected_neigbors > 0:
     #      self_timer[x, y] += 1
     #      if self_timer[x, y] >= self.recovery_time:      
     #        self.grid[x, y] = PersonState.recovered


    def animate(self, colormap=None):
      'Make animation of the whole history'
      T,n,m = self.history.shape
      if colormap is None:
        colormap = ListedColormap([
          "lightblue",   # susceptible (0)
          "red",         # infected (1)
          "green"        # recovered (2)
          "gray"         # empty / wall 
          ])

      fig, ax = plt.subplots()
      im = ax.imshow(self.history[0], cmap=colormap, interpolation='nearest', vmin=0, vmax=2)
      ax.set_title('Cellular automata')
      ax.axis('off')

      def init():
        im.set_data(self.history[0])
        return (im,)
      
      def update(frame):
        im.set_data(self.history[frame])
        ax.set_title(f't={frame}')
        return (im,)
      
      ani = FuncAnimation(fig,update, frames=T, init_func=init, interval = 100, blit=True)
      plt.show()
      return ani
      

def plot_SIR(simulation):
      history = simulation.history
      T, m, n = history.shape

      S_counts = []
      I_counts = []
      R_counts = []

      for t in range(simulation.step_count + 1):
        frame = history[t]
        S_counts.append(np.sum(frame == PersonState.susceptible.value))
        I_counts.append(np.sum(frame == PersonState.infected.value))
        R_counts.append(np.sum(frame == PersonState.recovered.value))

        timesteps = np.arange(len(S_counts))
        plt.figure()
        plt.plot(timesteps, S_counts, label='Susceptible')
        plt.plot(timesteps, I_counts, label='Infected')
        plt.plot(timesteps, R_counts, label='Recovered')
        plt.xlabel('Time step')
        plt.ylabel('Number of people')
        plt.title('SIR counts over time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
           
            
if  __name__ == '__main__':
  print("Running test simulation...")
  sim = SIRsimulation(
    gridsize=(50,50),
    infection_radius=1,
    step_threshold=100,
    average_infection_time=5,
    infection_probability=0.1,
    infection_time_variance=2,
    average_recovered_time = 10, 
    recovered_time_variance = 2    
    )
  from utils import gaussian_kernel
  sim.initialize_density(gaussian_kernel(sim.gridsize, 5))
  sim.add_infected(25)
  sim.run()
  plot_SIR(sim)

  for x in range(9):
    print('difference\n', sim.history[x+1] - sim.history[x])
  a = sim.animate()
  a.save('movie.mp4')

#plot number of people in S, I, R pop over time - HF
#beta fit to SIR -MB
#random travellers and infections that happen w a certain prob - HF
#diff matrices - empty, etc. - PP
#timer matrix for recovery time - also include waning immunity - AS
#github - AI -> done



