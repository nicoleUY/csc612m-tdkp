# CSC612M-TDKP Integrating Project

## Time Dependent Knapsack Problem with Irregular Availability
Time-dependent knapsack problem (TDKP) with irregular availability is a variant of the classic 0/1 knapsack problem. 

**INPUTS**:
- knapsack capacity
- time horizon
- list of items
  - price (profit or value)
  - demand (weight or cost)
  - availability interval (time window)
 
**GOAL**: maximize profit while staying within the knapsack's capacity at every time step.

The classical knapsack problem is NP-hard, meaning there are no known polynomial-time algorithms for guaranteed optimal solutions. Since TDKP is a variation of the 0/1 knapsack problem, it is also NP-hard. A pseudo-polynomial dynamic programming (DP) solution exists for the standard knapsack problem, but its DP formulation does not work for our instance, where items have to be removed from the knapsack. 

## Files
- `knapsack.cu`: Simulated Annealing implementation (both CPU and GPU)
- `Testcase Generator.ipynb`: Testcase generator
- `test-cases/`: Directory containing the test cases that were used
- `CSC612M Integrating Project Video.mp4`: Video presentation
- `CSC612M Integrating Project Slides.pdf`: Video presentation slides

## Discussion of Algorithms implemented

### Brute force
Our brute-force implementation serves as the baseline for correctness. It enumerates all 2^𝑛
 subsets of items, making it feasible only for small 𝑛. However, it is essential for verifying the correctness of heuristic and optimized methods.
 
### SA on CPU
Simulated annealing (SA) is the first heuristic method we implemented. It uses randomization to transition between knapsack states and runs multiple independent instances to improve the chances of finding good solutions. To our knowledge, this is the first use of simulated annealing applied to the TDKP. We used generic hyperparameters recommended by CP-Algorithms.

### SA on GPU
We parallelized simulated annealing by running 400 independent SA chains simultaneously. Each chain uses 32 worker threads to parallelize selected computations.

The main difference from the CPU version is that the CPU implementation runs only one SA chain at a time (or a few chains via multithreading), while the CUDA implementation runs hundreds concurrently.

Some optimizations we implemented include:
- Page creation and prefetching
- Memory coalescing, allowing one cache line to serve multiple instances
- Worker threads writing demand-change events to shared memory in parallel
- Single-threaded demand accumulation using a prefix-array style computation
- Kernel synchronization to alternate between single-threaded and multithreaded phases
- Allowing knapsack demand to temporarily exceed capacity using a penalty term (configurable), reducing branch divergence
- Final correction of violations on the CPU (using the CPU baseline) due to excessive branching on the GPU

### Linear Programming (LP) as a Benchmark
This implementation uses OR-Tools, Google’s open-source optimization library. It acts as a ground-truth solver, allowing us to measure how close the heuristic solutions are to those from a mature linear programming engine.

### Other attempts
Exact dynamic programming maximizes the total price of items in the knapsack at each time step, given the set of items currently chosen. However, this still becomes exponential in time because it requires exploring all feasible subsets at each time point to guarantee correctness. Unless a non-exponential DP formulation exists, it behaves like brute force in the worst case.

It performs better when fewer items overlap in time or when time intervals are fixed, but these constraints are not guaranteed in our dataset. Because of this risk, exact DP was not practical or scalable for our project.


## Execution time comparison
<img width="472" height="728" alt="image" src="https://github.com/user-attachments/assets/8e92e137-21cc-4e54-b67b-69f5b35eaf72" />


**Execution Time (Average over 5 runs)**
| n    | SA (CPU)  | SA (GPU)  | SPEEDUP |
|------|-----------|-----------|---------|
| 10   | 2008.344  | 15.83418  | 126x    |
| 25   | 3246.888  | 18.44622  | 176x    |
| 50   | 6652.54   | 25.00182  | 266x    |
| 100  | 9588.222  | 35.5514   | 270x    |
| 200  | 16804.56  | 57.66832  | 291x    |
| 250  | 23425.46  | 70.99128  | 330x    |
| 500  | 37419.44  | 115.0966  | 325x    |
| 1000 | 69236.3   | 195.242   | 355x    |
| 2000 | 123274.2  | 313.7148  | 393x    |

As n increases, speedup improves significantly. The GPU implementation consistently achieves at least 100× speedup over the CPU version.

**Output Comparison**
| n    | LP    | SA (CPU) | SA (GPU) |
|------|-------|----------|----------|
| 10   | 104   | 104      | 104      |
| 25   | 281   | 281      | 281      |
| 50   | 613   | 613      | 613      |
| 100  | 1188  | 1188     | 1188     |
| 200  | 2574  | 2553     | 2553     |
| 250  | 3100  | 2990     | 2990     |
| 500  | 6332  | 5601     | 5601     |
| 1000 | 12462 | 9077     | 9077     |
| 2000 | 24903 | 13662    | 13662    |

This table compares the solution quality of SA against the LP baseline.

For smaller 𝑛, SA matches the LP values exactly, indicating correctness and consistency. As 𝑛 increases, SA values deviate slightly—as expected for a heuristic method—yet remain reasonably close. The deviation may also be influenced by the non-tuned hyperparameters, as our focus was primarily on evaluating execution time.

What’s important is that both the CPU and GPU implementations of SA produce the same outputs, ensuring correctness of the parallelization.
