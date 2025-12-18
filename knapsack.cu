#include <bits/stdc++.h>
using namespace std;

__host__ __device__ uint64_t splitmix64(uint64_t x) {
    // http://xorshift.di.unimi.it/splitmix64.c
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
}

__host__ __device__ bool random_bool(uint64_t seed) {
    return splitmix64(seed) & 1;
}

__host__ __device__ uint64_t random_index(uint64_t seed, uint64_t n) {
    return splitmix64(seed) % n;
}

__host__ __device__ float random_float(uint64_t seed) {
    return (float)splitmix64(seed) / UINT64_MAX;
}

size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

/*
Input format:
n - number of items
m - knapsack capacity
p_i - price (or profit) of i-th item
d_i - resource demand (or amount of space in the knapsack) of i-th item
l_i, r_i - time interval the item must be in the knapsack (if selected)


n m
d_1 p_1 l_1 r_1
...
d_n p_n l_n r_n

7 10
5 6 1 3
9 11 1 3
5 5 2 4
7 6 4 6
3 4 5 7
7 6 7 9
5 5 8 10

Expected output (sum of prices): 22


26 50
7 14 3 18
12 19 1 22
4 7 10 27
15 24 5 29
6 9 14 33
9 12 20 40
11 21 25 45
8 13 30 48
14 26 2 17
5 8 35 60
10 18 40 62
7 11 45 70
12 22 50 73
9 15 55 82
6 10 60 85
13 23 12 30
4 6 65 90
11 20 70 92
5 9 75 96
9 14 80 99
8 12 18 37
10 17 22 50
6 11 33 57
7 13 44 78
12 25 52 88
9 16 10 100

Expected output: 105
*/

struct item {
    int price;
    int demand;
    int l, r;
};

/*
    Simulated annealing parameters from https://cp-algorithms.com/num_methods/simulated_annealing.html
*/

#define SIMULATED_ANNEALING_ITERATIONS 16000 // cp-algs suggests -log_u(T) but this can be changed
#define SIMULATED_ANNEALING_INITIAL_TEMPERATURE 100000
#define SIMULATED_ANNEALING_COOLING_RATE 0.995
#define SIMULATED_ANNEALING_EXCESS_ITEM_PENALTY 10 // amount of price to subtract per excess unit demand
#define SIMULATED_ANNEALING_INSTANCES 256
#define WORKERS_PER_INSTANCE 128

int violation_cost = 0;
int check_profit_CPP(int max_time, int knapsack_capacity, vector<int> &demand_change, bool disallow_violations) {
    int violation_amt = 0;
    int cur_demand = 0;
    for(int t = 0; t <= max_time; t++) {
        cur_demand += demand_change[t];
        if(disallow_violations && cur_demand > knapsack_capacity) {
            return -1;
        }
        violation_amt = max(violation_amt, cur_demand - knapsack_capacity);
    
    }
    
    return -violation_amt * violation_cost;
}

__device__ int check_profit_CUDA(int sa_instances, int iteration_id, int n, int max_time, int knapsack_capacity, int violation_cost, int *demand_change, int *prefix_sum) {
    // optimized prefix sum implementation
    for(int t = threadIdx.y; t <= max_time; t += blockDim.y) {
        prefix_sum[t * sa_instances + iteration_id] = demand_change[t * sa_instances + iteration_id];
    }
    __syncthreads();

    int dist_to_prv = 1;
    int spacing = 2;
    while(dist_to_prv <= max_time) {
        for(int t = max_time - threadIdx.y * spacing; t >= 0; t -= spacing * blockDim.y) {
            int prv = t - dist_to_prv;
            if(prv >= 0) {
                prefix_sum[t * sa_instances + iteration_id] += prefix_sum[prv * sa_instances + iteration_id];
            }
        }
        spacing <<= 1;
        dist_to_prv <<= 1;
        __syncthreads();
    }

    int tid = threadIdx.y;
    if(tid == 0) {
        prefix_sum[(max_time + 1) * sa_instances + iteration_id] = prefix_sum[max_time * sa_instances + iteration_id];
        prefix_sum[max_time * sa_instances + iteration_id] = 0;
    }
    __syncthreads();
    
    
    spacing >>= 1;
    dist_to_prv >>= 1;
    while(spacing >= 2) {
        for(int t = max_time - threadIdx.y * spacing; t >= 0; t -= spacing * blockDim.y) {
            int prv = t - dist_to_prv;
            int prv_actual_idx = prv * sa_instances + iteration_id;
            int cur_actual_idx = t * sa_instances + iteration_id;
            if(prv >= 0) {
                int prv_val = prefix_sum[prv_actual_idx];
                prefix_sum[prv_actual_idx] = prefix_sum[cur_actual_idx];
                prefix_sum[cur_actual_idx] += prv_val;
            }
        }
        spacing >>= 1;
        dist_to_prv >>= 1;
        __syncthreads();
    }

    __shared__ int tmp[WORKERS_PER_INSTANCE];
    int max_violation = 0;
    for(int t = threadIdx.y; t <= max_time + 1; t += blockDim.y) {
        max_violation = max(max_violation, prefix_sum[t * sa_instances + iteration_id] - knapsack_capacity);
    }
    tmp[tid] = max_violation;
    __syncthreads();
    
    if(tid == 0) {
        for(int i = 0; i < blockDim.y; i++) {
            max_violation = max(max_violation, tmp[i]);
        }
    }

    return -max_violation * violation_cost;
}

int knapsack_simulated_annealing_CPP_kernel(int id, int max_time, int knapsack_capacity, vector<item> &items) {
    vector<bool> selected(items.size()), candidate_selected(items.size());
    vector<int> demand_change(max_time + 3);

    uint64_t seed = id * (16 * SIMULATED_ANNEALING_ITERATIONS);
    int optimal = 0;
    int selected_count = 0, candidate_selected_count = 0;
    int cur_profit = 0;


    double T = SIMULATED_ANNEALING_INITIAL_TEMPERATURE;
    for(int i = 0; i < SIMULATED_ANNEALING_ITERATIONS; i++) {
        int target = random_index(seed++, items.size());
        bool cur_val = candidate_selected[target];
        int delta = cur_val ? -1 : +1;
        candidate_selected[target] = !candidate_selected[target];
        candidate_selected_count += delta;
        demand_change[items[target].l] += delta * items[target].demand;
        demand_change[items[target].r + 1] -= delta * items[target].demand;
        cur_profit += delta * items[target].price;

        int adjusted_profit = cur_profit - check_profit_CPP(max_time, knapsack_capacity, demand_change, false);

        double prob = exp((adjusted_profit-optimal)/T);
        if(random_float(seed++) < prob) {
            optimal = adjusted_profit;
            selected[target] = candidate_selected[target];
            selected_count = candidate_selected_count;
        } else {
            candidate_selected[target] = selected[target];
            candidate_selected_count = selected_count;
            demand_change[items[target].l] -= delta * items[target].demand;
            demand_change[items[target].r + 1] += delta * items[target].demand;
            cur_profit -= delta * items[target].price;
        }

        T *= SIMULATED_ANNEALING_COOLING_RATE;
    }

    uint64_t seed2 = id * (16 * SIMULATED_ANNEALING_ITERATIONS);
    while(check_profit_CPP(max_time, max_time, demand_change, true) == -1) {
        while(true) {
            int item_to_remove = random_index(seed2++, items.size());
            if(candidate_selected[item_to_remove]) {
                candidate_selected[item_to_remove] = false;
                
                demand_change[items[item_to_remove].l] -= items[item_to_remove].demand;
                demand_change[items[item_to_remove].r + 1] += items[item_to_remove].demand;
                cur_profit -= items[item_to_remove].price;
                break;
            }
        }
    }
    return cur_profit;
}

__global__
void knapsack_simulated_annealing_CUDA_kernel(int n, int max_time, int knapsack_capacity, int violation_cost, int *demand_change, int *prefix_sum, const item * __restrict__ items, bool *selected, bool *candidate_selected) {
    int iteration_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    uint64_t seed = iteration_id * (16 * SIMULATED_ANNEALING_ITERATIONS);
    int optimal = 0;
    
    int selected_count = 0, candidate_selected_count = 0;
    int cur_profit = 0;

    double T = SIMULATED_ANNEALING_INITIAL_TEMPERATURE;
    for(int i = 0; i < SIMULATED_ANNEALING_ITERATIONS; i++) {
        item selected_item;
        int target, delta;
        if(threadIdx.y == 0) {
            int target_itemno = random_index(seed++, n);
            selected_item = items[target_itemno];
            target = target_itemno * stride + iteration_id;

            bool cur_val = candidate_selected[target];
            delta = cur_val ? -1 : +1;

            candidate_selected[target] = !candidate_selected[target];
            candidate_selected_count += delta;
            candidate_selected_count -= delta;

            demand_change[selected_item.l       * stride + iteration_id] += delta * selected_item.demand;
            demand_change[(selected_item.r + 1) * stride + iteration_id] -= delta * selected_item.demand;
            cur_profit += delta * selected_item.price;
        }
        __syncthreads();


        int adjusted_profit = cur_profit + check_profit_CUDA(stride, iteration_id, n, max_time, knapsack_capacity, violation_cost, demand_change, prefix_sum);
        if(threadIdx.y == 0) {
            double prob = exp((adjusted_profit-optimal)/T);
            if(random_float(seed++) < prob) { // whether accept
                optimal = adjusted_profit;
                selected[target] = candidate_selected[target];
                selected_count = candidate_selected_count;
            } else {
                candidate_selected[target] = selected[target];
                candidate_selected_count = selected_count;
                cur_profit -= delta * selected_item.price;
                demand_change[(selected_item.l)     * stride + iteration_id] -= delta * selected_item.demand;
                demand_change[(selected_item.r + 1) * stride + iteration_id] += delta * selected_item.demand;
            }
        }
        
        T *= SIMULATED_ANNEALING_COOLING_RATE;
        __syncthreads();
    }
}


int main() {
    int n, m;
    cin >> n >> m;

    int max_time = 0;
    vector<item> items(n);
    set<int> lr;
    map<int, int> remap_value;
    for(int i = 0; i < n; i++) {
        cin >> items[i].price >> items[i].demand >> items[i].l >> items[i].r;

        lr.insert(items[i].l);
        lr.insert(items[i].r);
        violation_cost = max(violation_cost, SIMULATED_ANNEALING_EXCESS_ITEM_PENALTY * items[i].price / items[i].demand + 1);
    } 
    {
        int id = 0;
        for(auto x : lr) {
            remap_value[x] = id++;
        }
        for(int i = 0; i < n; i++) {
            items[i].l = remap_value[items[i].l];
            items[i].r = remap_value[items[i].r];
            max_time = max(max_time, items[i].r);
        } 
    }
    

    double total = 0;
    
    size_t instances = SIMULATED_ANNEALING_INSTANCES;
    vector<int> cpu_output(instances);
    
    int mx = 0;
    for(int i = 0; i < instances; i++) {
        auto start = chrono::system_clock::now();
        int r = knapsack_simulated_annealing_CPP_kernel(i, max_time, m, items) ;
        auto end = chrono::system_clock::now();

        auto ms = chrono::duration_cast<chrono::nanoseconds>(end - start).count() / 1e6;
        /*
        if(i == 0) {
            first_run = ms;
        }
        */
        total += ms;

        cpu_output[i] = r;
        mx = max(mx, r);
        // cout << "simulated annealing returned: " << r << endl;
        // cout << "run took " << ms << endl;
        
    }
    cout << "CPU SA:  " << instances
     << " instances (n = " << n << ") took "
     << fixed << setprecision(6) << total << " ms\n"
     << "Max result: " << mx << "\n\n";
    //return;

    
    int *demand_change, *prefix_sum;
    item *items_cuda;
    bool *selected, *candidate_selected;

    const size_t workers = WORKERS_PER_INSTANCE;
    const size_t num_threads = min(instances, 1024 / workers); // 1024
    const size_t num_blocks = ceil_div(instances, num_threads);

    size_t instances_rounded_up = num_threads * num_blocks;
    //cout << instances_rounded_up << endl;
    cudaMalloc(&demand_change, instances_rounded_up * (max_time + 3) * sizeof(int));
    cudaMemset(demand_change, 0, instances_rounded_up * (max_time + 3) * sizeof(int));

    cudaMalloc(&prefix_sum, instances_rounded_up * (max_time + 3) * sizeof(int));

    cudaMalloc(&selected, instances_rounded_up * n * sizeof(bool));
    cudaMemset(selected, 0, instances_rounded_up * n * sizeof(bool));

    cudaMallocManaged(&candidate_selected, instances_rounded_up * n * sizeof(bool));
    cudaMemset(candidate_selected, 0, instances_rounded_up * n * sizeof(bool));

    cudaMalloc(&items_cuda, n * sizeof(item));
    cudaMemcpy(items_cuda, &items[0], n * sizeof(item), cudaMemcpyHostToDevice);

    int device = -1;
    cudaGetDevice(&device);
    dim3 threads_per_block(num_threads, workers);

    auto start = chrono::system_clock::now();
    knapsack_simulated_annealing_CUDA_kernel<<<num_blocks, threads_per_block>>>(n, max_time, m, violation_cost, demand_change, prefix_sum, items_cuda, selected, candidate_selected);
    (void)cudaDeviceSynchronize();
    
    
    cudaMemLocation memLocation;
    memLocation.type = cudaMemLocationTypeHost;
    cudaMemPrefetchAsync(candidate_selected, instances_rounded_up * n * sizeof(bool), memLocation, NULL);

    // cpu stage
    mx = 0;
    for(int i = 0; i < instances; i++) {
        vector<bool> candidate_selected_cpu(n);
        for(int j = 0; j < n; j++) {
            candidate_selected_cpu[j] = candidate_selected[j * instances_rounded_up + i];
        }
        vector<int> demand_change(max_time + 3);

        int cur_profit = 0;
        for(int j = 0; j < items.size(); j++) {
            if(candidate_selected_cpu[j]) {
                demand_change[items[j].l] += items[j].demand;
                demand_change[items[j].r + 1] -= items[j].demand;
                cur_profit += items[j].price;
            }
        }

        uint64_t seed2 = i * (16 * SIMULATED_ANNEALING_ITERATIONS);
        while(check_profit_CPP(max_time, m, demand_change, true) == -1) {
            while(true) {
                int item_to_remove = random_index(seed2++, items.size());
                if(candidate_selected[item_to_remove]) {
                    candidate_selected[item_to_remove] = false;
                    
                    demand_change[items[item_to_remove].l] -= items[item_to_remove].demand;
                    demand_change[items[item_to_remove].r + 1] += items[item_to_remove].demand;
                    cur_profit -= items[item_to_remove].price;
                    break;
                }
            }
        }

        if(true && cpu_output[i] != cur_profit) {
            cout << "detected error\n";
        }
        //cout << "simulated annealing CUDA returned: " << output << endl;
        mx = max(mx, cur_profit);
    }

    auto end = chrono::system_clock::now();

    auto ms = chrono::duration_cast<chrono::nanoseconds>(end - start).count() / 1e6;
    cout << "GPU SA:  " << instances
     << " instances (n = " << n << ") took "
     << fixed << setprecision(3) << ms << " ms\n"
     << "Max result: " << mx << "\n";
}