from main import SIRsimulation
from kernels import cake_kernel

def inc(self):
  self.infection_probability *= 2

if __name__ == '__main__':
  kernel = cake_kernel(
      200,                  # number of columns
      (60, (0.0, 0.2)),    # top 3 rows ~ Uniform[0.0, 0.2]
      (80, (0.4, 0.6)),    # next 4 rows ~ Uniform[0.4, 0.6]
      (60, (0.8, 1.0)),    # bottom 2 rows ~ Uniform[0.8, 1.0]
  )
  sim = SIRsimulation(waning_recovery=True, rules={400: inc})
  sim.initialize_density(kernel,proc_points=0.3)
  sim.add_infected(5)
  sim.run()
  sim.animate()
