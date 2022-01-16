from pyakmaxsat import AKMaxSATSolver

solver = AKMaxSATSolver()
sampleset = solver.sample_wcnf('samp.wcnf')
print(sampleset)
