# CSC612M-TDKP Integrating Project

## Time Dependent Knapsack Problem with Irregular Availability
Time-dependent knapsack problem (TDKP) with irregular availability is a variant of the classic 0/1 knapsack problem. 
INPUTS:
- knapsack capacity
- time horizon
- list of items
  - price (profit or value)
  - demand (weight or cost)
  - availability interval (time window)
 
GOAL: maximize profit while staying within the knapsack's capacity at every time step.

## Discussion of Algorithms implemented in your program

### Brute force

### SA on CPU

### SA on GPU
highlight which part of the sequential part is converted to parallel algo

### Other attemps
talk about DP

## Execution time comparison
<insert screenshot>

**Execution Time**
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



**Output**
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


