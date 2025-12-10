import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from enum import Enum
import random
import itertools


class PersonState(Enum):
  susceptible = 0
  infected = 1
  recovered = 2
  empty = 3

class SIRsimulation:
    
    def __init__(self, gridsize, infection_rate, infection_radius, step_threshold, average_infection_time, infection_probability, infection_time_variance, recovery_time=10, timestep=1):
      assert isinstance(gridsize, tuple)
      self.gridsize = gridsize  #size of grid
      self.grid = np.full(gridsize, PersonState.susceptible.value) #initialize grid based off of gridzie and fill with susceptible
      self.timers = np.full(gridsize, 0) #keep track of time for time steps
      self.infection_rate = infection_rate #beta
      self.infection_probability = infection_probability
      self.infection_radius = infection_radius #how far an indi can infect
      self.step_threshold = step_threshold #max # of time steps to go through
      self.step_count = 0 #number of time steps
      
      self.history = np.zeros((step_threshold, *gridsize)) #encodes history of each cell(individual) for each time step. x,y,timestep; think 2d w individuals and the z is time steps
      self.recovery_time = recovery_time ###Change when simulating different times #num of time steps b4 infected recovers
      self.average_infection_time = average_infection_time
      self.infection_time_variance = infection_time_variance
      self.timestep = timestep
    
    def is_not_finished(self): #keep going if havent reach threshold and still have infected
      infected_left_over = np.any(self.grid == PersonState.infected.value)
      return (self.step_count < self.step_threshold) and infected_left_over
      
    def add_infected(self, number):
      for i in range(number):
        x = random.randint(0, self.gridsize[0]-1)
        y = random.randint(0, self.gridsize[1]-1)
        self.history[self.step_count][x,y] = PersonState.infected.value
        self.grid[x, y] = PersonState.infected.value
        self.timers[x, y] = random.gauss(self.average_infection_time, self.infection_time_variance)
    
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
      infection_mask = (self.history[t]==PersonState.infected.value)
      print(f'this is', infection_mask)
      self.history[t+1] = self.history[t]     
      
      # update the timers by substracting one timestep from each entry, then clip the values, so they are between 0 and 100
      self.timers = self.timers - self.timestep
      self.timers = self.timers.clip(min=0, max=100)

      infected_coordinates = np.argwhere(self.history[t] == PersonState.infected.value)
      # loop through all the infected individuals
      for x,y in infected_coordinates:
    #   for (x,y), value in np.ndenumerate(self.history[t][infection_mask]):
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
                self.timers[nx, ny] = random.gauss(self.average_infection_time, self.infection_time_variance)
        
          # if the timer of the infected individual has reached 0
          if self.timers[x,y] == 0:
            # update it's state to be recovered
            self.history[t+1][x,y] = PersonState.recovered.value
      print(sum(sum(infection_mask)))
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


    def plot_frame(self):
      ''''For saving the frame, add live plotting later'''
      self.save_frame()
    
    def save_frame(self):
      '''
      Call it per timestep to save state of the map at a given timepoint
      '''
      self.grid = self.history[self.step_count].copy()

    def animate(self, colormap=None):
      'Make animation of the whole history'
      T,n,m = self.history.shape
      if colormap is None:
        colormap = ListedColormap([
          "lightblue",   # susceptible (0)
          "red",         # infected (1)
          "green"        # recovered (2)
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
      
if  __name__ == '__main__':
  print("Running test simulation...")
  sim = SIRsimulation(
    gridsize=(50,50),
    infection_rate=5,
    infection_radius=1,
    step_threshold=100,
    average_infection_time=5,
    infection_probability=0.1,
    infection_time_variance=2,
    recovery_time=6
    )
   
  sim.add_infected(25)
  sim.run()
  for x in range(9):
    print('difference\n', sim.history[x+1] - sim.history[x])
  a = sim.animate()
  a.save('movie.mp4')

#plot number of people in S, I, R pop over time - HF / PP
#beta fit to SIR -MB
#random travellers and infections that happen w a certain prob - HF
#diff matrices - empty, etc. - PP
#timer matrix for recovery time - also include waning immunity - AS
#github - AI


