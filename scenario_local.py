from main import SIRsimulation
from kernels import cake_kernel
import main as simulation
from matplotlib.animation import FFMpegWriter

def inc_inf(self):
  self.infection_probability *= 2

if __name__ == '__main__':
  kernel = cake_kernel(
      200,                  # number of columns
      (60, (0.0, 0.2)),    # top 3 rows ~ Uniform[0.0, 0.2]
      (80, (0.4, 0.6)),    # next 4 rows ~ Uniform[0.4, 0.6]
      (60, (0.8, 1.0)),    # bottom 2 rows ~ Uniform[0.8, 1.0]
  )
  sim = SIRsimulation(waning_recovery=True, rules={400: simulation.increase_infection_prob})
  sim.initialize_density(kernel,proc_points=0.4)
  sim.add_infected(10)
  sim.run()
  anim = sim.animate()
  anim.save('cake_inc_inf.mp4', writer=FFMpegWriter(fps=10))
  print('Running cake - scenario 2')

  kernel = cake_kernel(
      200,                  # number of columns
      (60, (0.0, 0.2)),    # top 3 rows ~ Uniform[0.0, 0.2]
      (80, (0.4, 0.6)),    # next 4 rows ~ Uniform[0.4, 0.6]
      (60, (0.8, 1.0)),    # bottom 2 rows ~ Uniform[0.8, 1.0]
  )
  sim = SIRsimulation(waning_recovery=True, rules={400: simulation.reduce_travel})
  sim.initialize_density(kernel,proc_points=0.4)
  sim.add_infected(10)
  sim.run()
  anim = sim.animate()
  anim.save('cake_reduce_travel.mp4' , writer=FFMpegWriter(fps=10))


