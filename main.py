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
random.seed(92)

colormap = ListedColormap([
  "lightblue",   # susceptible (0)
  "red",         # infected (1)
  "green",        # recovered (2)
  "gray"         # empty / wall 
  ])

class PersonState(Enum):
  susceptible = 0
  infected = 1
  recovered = 2
  empty = 3


def change_stuff(self):
  print('changing stuff')
  self.infection_radius = 4

def dec_infection_probability(self):
  print('changing stuff')
  self.infection_probability = 0.075

def reduce_travel(self):
  print('reduce traveling')
  self.travel_prob = self.travel_prob * 0.01

def increase_infection_prob(self):
  print('increase infection probability')
  self.infection_probability = self.infection_probability * 2

class SIRsimulation:
    
    def __init__(self, gridsize=(200,200), infection_radius=1, step_threshold=800, average_infection_time=10, infection_probability=0.05, infection_time_variance=4, average_recovered_time=60, recovered_time_variance=15, travel_prob=0.01, waning_recovery=False, travel_TF=True, timestep=1, rules=None):
      assert isinstance(gridsize, tuple)
      self.gridsize = gridsize  #size of grid
      self.grid = np.full(gridsize, PersonState.susceptible.value) #initialize grid based off of gridzie and fill with susceptible
      self.infection_timers = np.full(gridsize, 0) #keep track of time for time steps
      self.recovery_timers = np.full(gridsize, 0)
      self.infection_probability = infection_probability # beta
      self.infection_radius = infection_radius #how far an indi can infect
      self.step_threshold = step_threshold #max # of time steps to go through
      self.step_count = 0 #number of time steps

      self.rules = rules
      # rules: { timestep -> func }

      self.travel_prob = travel_prob
      self.travel_infection_prob = self.infection_probability
      self.history = np.zeros((step_threshold, *gridsize)) #encodes history of each cell(individual) for each time step. x,y,timestep; think 2d w individuals and the z is time steps
      self.average_infection_time = average_infection_time
      self.infection_time_variance = infection_time_variance
      self.waning_recovery = waning_recovery
      self.average_recovered_time = average_recovered_time
      self.recovered_time_variance = recovered_time_variance
      self.timestep = timestep
      self.travel_TF = travel_TF
    
    def is_not_finished(self): #keep going if havent reach threshold and still have infected
      infected_left_over = np.any(self.grid == PersonState.infected.value)
      return (self.step_count < self.step_threshold) and infected_left_over
    
    def initialize_density(self, kernel, n_points=None, proc_points=None):
      assert proc_points is not None or n_points is not None
      if proc_points is not None:
        n_points = int(self.gridsize[0]*self.gridsize[1]*proc_points)

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
      return np.column_stack((rows, cols))
      
    def add_infected(self, number):
      for i in range(number):
        x = random.randint(0, self.gridsize[0]-1)
        y = random.randint(0, self.gridsize[1]-1)
        self.history[self.step_count][x,y] = PersonState.infected.value
        self.grid[x, y] = PersonState.infected.value
        self.infection_timers[x, y] = random.gauss(self.average_infection_time, self.infection_time_variance)
    
    def apply_rules(self):
      if self.rules and self.step_count in self.rules.keys():
        self.rules[self.step_count](self)
      
    def run(self):
      while self.step_count < self.step_threshold-2:
        #print('step', self.step_count)
        self.apply_rules()
        self.step()
        if self.travel_TF == True:
          self.random_travel_and_infection()
        #self.save_frame()
    
    
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
      
      # update the infection and recovery timers by substracting one timestep from each entry, then clip the values, 
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
            if self.waning_recovery == True:
              self.recovery_timers[x,y] = random.gauss(self.average_recovered_time, self.recovered_time_variance)
      
      if self.waning_recovery == True:
        # get coordinates of the individuals that are currently recovered
        recovered_coordinates = np.argwhere(self.history[t] == PersonState.recovered.value)
        # loop through all the recovered individuals
        for x,y in recovered_coordinates:
          # if the individuals recovery timer has reached 0
          if self.recovery_timers[x,y] == 0:
            # update it's state in the next timestep to susceptible
            self.history[t+1][x,y] = PersonState.susceptible.value
          
      self.step_count += 1 

    
    def print_frame(self):
      fig, ax = plt.subplots()
      im = ax.imshow(self.history[self.step_count], cmap=colormap, interpolation='nearest', vmin=0, vmax=3)
      plt.show()

    def random_travel_and_infection(self):
      t = self.step_count
      current = self.history[t]
      m, n = self.gridsize

      # Work on a copy so we don't interfere with iteration
      new_frame = current.copy()

      for x in range(m):
        for y in range(n):
            state = current[x, y]

            # Skip 'empty' cells if you decide to use them later
            if state == PersonState.empty.value:
                continue

            # Decide if this person travels
            if random.random() < self.travel_prob:
                # Pick a random destination
                tx = random.randint(0, m - 1)
                ty = random.randint(0, n - 1)
                dest_state = current[tx, ty]

                # If an infected person travels onto a susceptible one,
                # there is a chance of infection at the destination.
                if (state == PersonState.infected.value and
                    dest_state == PersonState.susceptible.value and
                    random.random() < self.travel_infection_prob):

                    new_frame[tx, ty] = PersonState.infected.value
                    # Give the new infected a timer
                    self.infection_timers[tx, ty] = random.gauss(
                        self.average_infection_time,
                        self.infection_time_variance
                    )

      self.history[t] = new_frame


    def animate(self, colormap=None):
      'Make animation of the whole history'
      T,n,m = self.history.shape
      if colormap is None:
        colormap = ListedColormap([
          "lightblue",   # susceptible (0)
          "red",         # infected (1)
          "green",        # recovered (2)
          "gray"         # empty / wall 
          ])

      fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17,5))
      ax0 = ax[0]
      im = ax0.imshow(self.history[0], cmap=colormap, interpolation='nearest', vmin=0, vmax=3)
      #ax0.set_title('')
      #ax0.axis('off')

      history = self.history
      S_counts = []
      I_counts = []
      R_counts = []
      I_prev1 = [0]
      I_prev2 = [0,0]

      for t in range(self.step_count + 1):
        frame = history[t]
        S_counts.append(np.sum(frame == PersonState.susceptible.value))
        I_counts.append(np.sum(frame == PersonState.infected.value))
        R_counts.append(np.sum(frame == PersonState.recovered.value))
      I_prev1[1:len(I_counts[0:-1])] = I_counts[0:-1]
      I_prev2[2:len(I_counts[0:-2])] = I_counts[0:-2]
      Rt_counts1 = [i/(j+0.01) for i,j in zip(I_counts, I_prev1)]
      Rt_counts2 = [i/(j+0.01) for i,j in zip(I_counts, I_prev2)]

      timesteps = np.arange(len(S_counts))

      ax1 = ax[1]
      ax2 = ax[2]
      line_S, = ax1.plot([], [], label='Susceptible')
      line_I, = ax1.plot([], [], label='Infected')
      line_R, = ax1.plot([], [], label='Recovered')
      line_Rt1, = ax2.plot([], [], label='delta(t) = 1')
      line_Rt2, = ax2.plot([], [], label='delta(t) = 2')

      ax1.set_xlim(0, len(S_counts))
      ax1.set_ylim(0, max(S_counts[0], I_counts[0], R_counts[0], max(S_counts + I_counts + R_counts)))
      ax1.set_xlabel('Time step')
      ax1.set_ylabel('Number of people')
      ax1.set_title(f'SIR counts over time')
      ax1.legend()
      ax1.grid(True)

      ax2.set_xlim(0, len(S_counts))
      ax2.set_ylim(0, 5)
      ax2.set_xlabel('Time')
      ax2.set_ylabel('R(t)')
      ax2.set_title('Basic reproduction number')
      ax2.legend()
      ax2.grid(True)


      def init():
        im.set_data(self.history[0])
        line_S.set_data([], [])
        line_I.set_data([], [])
        line_R.set_data([], [])
        line_Rt1.set_data([], [])
        line_Rt2.set_data([], [])
        return im, line_S, line_I, line_R, line_Rt1, line_Rt2
      
      def update(frame):
        im.set_data(self.history[frame])
        ax0.set_title(frame)
        x = timesteps[:frame+1]
        line_S.set_data(x, S_counts[:frame+1])
        line_I.set_data(x, I_counts[:frame+1])
        line_R.set_data(x, R_counts[:frame+1])
        line_Rt1.set_data(x, Rt_counts1[:frame+1])
        line_Rt2.set_data(x, Rt_counts2[:frame+1])
        return im, line_S, line_I, line_R, line_Rt1, line_Rt2
      
      ani = FuncAnimation(fig,update, frames=T, init_func=init, interval = 10, blit=True)
      #plt.show()
      plt.close(fig)
      return ani
      


               
            
if  __name__ == '__main__':
  print("Running test simulation...")
  sim = SIRsimulation(waning_recovery=True)
  from kernels import gaussian_kernel, wall, negative_gaussian, multi_negative_gaussian
  sim.initialize_density(wall(sim.gridsize, 5,40, 10, 20), n_points=50)
  sim.initialize_density(multi_negative_gaussian(sim.gridsize, [(40,40), (160,160)], 40), proc_points=0.3)
  sim.add_infected(10)
  sim.run()


  a = sim.animate()
  a.save("movie.gif", writer=PillowWriter(fps=10))
  





#plot number of people in S, I, R pop over time - HF -> done
#beta fit to SIR -MB -> done
#random travellers and infections that happen w a certain prob - HF -> done
#diff matrices - empty, etc. - P
#timer matrix for recovery time - also include waning immunity - AS -> done
#github - AI -> done
#T/F for waning /recovery -> done



### new things to implement
# ~ Calculate R0 and R(t) then plot R(t) over time -> Rt plot is done
# ~ Plot infected over time for different values of beta/infection_probability and alpha/average_infection_time 
#     and infection_radius in simulation
# ~ Think about how we could reflect different disease containement measures in our simulation (f.e. wearing masks, 
#     social distancing, maybe even vaccinations)
#testing conditions for presentation: diff scenarios -> city area vs village??; etc


