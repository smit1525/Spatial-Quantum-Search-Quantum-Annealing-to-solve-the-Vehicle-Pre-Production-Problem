from __future__ import print_function

from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler

from job_shop_scheduler import get_jss_bqm, is_auxiliary_variable
import numpy as np
# Construct a BQM for the jobs
jobs = {"Vehicle1": [("Test1", 2)],
        "Vehicle2": [("Test3", 1)],
        "Vehicle3": [("Test2", 2)]}
jobs=jobs.dtypes.to_dict()
max_time = 4	  # Upperbound on how long the schedule can be; 4 is arbitrary
bqm = get_jss_bqm(jobs, max_time)

# Submit BQM
# Note: may need to tweak the chain strength and the number of reads
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(bqm,
                           chain_strength=2,
                           num_reads=1000,
                           label='Scheduling')

# Grab solution
solution = sampleset.first.sample


# Grab selected nodes
selected_nodes = [k for k, v in solution.items() if v == 1]

# Parse node information
task_times = {k: [-1]*len(v) for k, v in jobs.items()}
for node in selected_nodes:
    if is_auxiliary_variable(node):
        continue
    job_name, task_time = node.rsplit("_", 1)
    task_index, start_time = map(int, task_time.split(","))

    task_times[job_name][task_index] = start_time

# Print problem and restructured solution
print("Jobs and their machine-specific tasks:")
for job, task_list in jobs.items():
    print("{0:9}: {1}".format(job, task_list))

print("\nJobs and the start times of each task:")
for job, times in task_times.items():
    print("{0:9}: {1}".format(job, times))
