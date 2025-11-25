# CSC612M-TDKP Integrating Project

## Time Dependent Knapsack Problem with Irregular Availability

## Discussion of Algorithms implemented in your program

### Brute force

### DP on CPU

### DP on GPU

### SA on CPU

### SA on GPU
highlight which part of the sequential part is converted to parallel algo

## Execution time comparison
<insert screenshot>
  
**Execution Time of Different Kernels**
| **n**   | **BF** | **DP CPU** | **SA CPU** | **DP GPU** | **SA GPU** |
|---------|--------|------------|------------|------------|------------|
| 10      |        |            |            |            |            |
| 25      |        |            |            |            |            |
| 50      |        |            |            |            |            |
| 100     |        |            |            |            |            |
| 200     |        |            |            |            |            |
| 250     |        |            |            |            |            |
| 500     |        |            |            |            |            |
| 1000    |        |            |            |            |            |
| 2000    |        |            |            |            |            |


**Pairwise Speed (n = 10)**
| Method  | BF  | DP CPU | SA CPU | DP GPU | SA GPU |
|---------|-----|--------|--------|--------|--------|
| BF      | 1x  |        |        |        |        |
| DP CPU  |     | 1x     |        |        |        |
| SA CPU  |     |        | 1x     |        |        |
| DP GPU  |     |        |        | 1x     |        |
| SA GPU  |     |        |        |        | 1x     |

