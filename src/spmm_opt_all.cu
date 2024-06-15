#include "spmm_opt.h"
#include <vector>
#include <algorithm>

#define X_BLOCK_SIZE 32
#define Y_BLOCK_SIZE 32
#define BLOCK_SIZE (X_BLOCK_SIZE * Y_BLOCK_SIZE)

#define TARGET_SIZE 256

inline int ceiling(int a, int b) { return (a + b - 1) / b; }

// 不做负载均衡的 naive 实现
__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in)
{
    __shared__ int s_idx[X_BLOCK_SIZE][Y_BLOCK_SIZE];
    __shared__ float s_val[X_BLOCK_SIZE][Y_BLOCK_SIZE];

    const int t_x = threadIdx.x / Y_BLOCK_SIZE;
    const int t_y = threadIdx.x % Y_BLOCK_SIZE;
    const int x = blockIdx.x * X_BLOCK_SIZE + t_x;
    const int y = blockIdx.y * Y_BLOCK_SIZE + t_y;

    if (x >= num_v)
        return;

    const int beg = __ldg(&ptr[x]);
    const int end = __ldg(&ptr[x + 1]);
    const float *offset_pos = vin + y;
    float sum = 0.0f;

    for (int i = beg; i < end; i += Y_BLOCK_SIZE)
    {
        const int rank = i + t_y;
        if (rank < end)
        {
            s_idx[t_x][t_y] = __ldg(&idx[rank]);
            s_val[t_x][t_y] = __ldg(&val[rank]);
        }
        else
        {
            s_idx[t_x][t_y] = 0;
            s_val[t_x][t_y] = 0.0f;
        }
        __syncthreads();
        if (y < feat_in)
            for (int j = 0; j < Y_BLOCK_SIZE; ++j)
                sum += s_val[t_x][j] * __ldg(offset_pos + (s_idx[t_x][j] * feat_in));
        __syncthreads();
    }

    if (y < feat_in)
        vout[x * feat_in + y] = sum;
}

// 做负载均衡的实现
__global__ void spmm_kernel_opt_target(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_in, int num_target, int *target, int *ptr_scheduled)
{
    __shared__ int s_idx[X_BLOCK_SIZE][Y_BLOCK_SIZE];
    __shared__ float s_val[X_BLOCK_SIZE][Y_BLOCK_SIZE];

    const int t_x = threadIdx.x / Y_BLOCK_SIZE;
    const int t_y = threadIdx.x % Y_BLOCK_SIZE;
    const int my_target = blockIdx.x * X_BLOCK_SIZE + t_x;

    if (my_target >= num_target)
        return;

    const int x = __ldg(&ptr_scheduled[my_target]);
    const int y = blockIdx.y * Y_BLOCK_SIZE + t_y;
    const int beg = __ldg(&target[my_target]);
    const int end = __ldg(&target[my_target + 1]);

    const float *offset_pos = vin + y;
    float sum = 0.0f;

    for (int i = beg; i < end; i += Y_BLOCK_SIZE)
    {
        const int rank = i + t_y;
        if (rank < end)
        {
            s_idx[t_x][t_y] = __ldg(&idx[rank]);
            s_val[t_x][t_y] = __ldg(&val[rank]);
        }
        else
        {
            s_idx[t_x][t_y] = 0;
            s_val[t_x][t_y] = 0.0f;
        }
        __syncthreads();
        if (y < feat_in)
            for (int j = 0; j < Y_BLOCK_SIZE; ++j)
                sum += s_val[t_x][j] * __ldg(&offset_pos[s_idx[t_x][j] * feat_in]);
        __syncthreads();
    }

    if (y < feat_in)
        atomicAdd(&vout[x * feat_in + y], sum);
}

void SpMMOpt::edgesort()
{
    int *h_ptr = (int *)malloc(sizeof(int) * (num_v + 1));
    int *h_idx = (int *)malloc(sizeof(int) * num_e);
    float *h_val = (float *)malloc(sizeof(float) * num_e);
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, sizeof(int) * (num_v + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_idx, d_idx, sizeof(int) * num_e, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_val, d_val, sizeof(float) * num_e, cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_v; i++)
    {
        std::vector<std::pair<int, float>> edges;
        for (int j = h_ptr[i]; j < h_ptr[i + 1]; j++)
            edges.push_back(std::make_pair(h_idx[j], h_val[j]));
        std::sort(edges.begin(), edges.end());
        for (int j = h_ptr[i]; j < h_ptr[i + 1]; j++)
        {
            h_idx[j] = edges[j - h_ptr[i]].first;
            h_val[j] = edges[j - h_ptr[i]].second;
        }
    }
    checkCudaErrors(cudaMemcpy(d_idx, h_idx, sizeof(int) * num_e, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, h_val, sizeof(float) * num_e, cudaMemcpyHostToDevice));
    free(h_ptr);
    free(h_idx);
    free(h_val);
}

void SpMMOpt::neighbor_grouping(int neighbor_num)
{
    int *h_ptr = (int *)malloc(sizeof(int) * (num_v + 1));
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, sizeof(int) * (num_v + 1), cudaMemcpyDeviceToHost));
    vector<int> t_ptr, t_rnk;
    for (int x = 0; x < num_v; ++x)
    {
        int begin = h_ptr[x], end = h_ptr[x + 1];
        for (int rnk = begin; rnk < end; rnk += neighbor_num)
        {
            t_ptr.push_back(x);
            t_rnk.push_back(rnk);
        }
    }
    t_rnk.push_back(num_e);
    num_target = t_ptr.size();
    checkCudaErrors(cudaMalloc(&ptr_scheduled, sizeof(int) * num_target));
    checkCudaErrors(cudaMalloc(&target, sizeof(int) * (num_target + 1)));
    checkCudaErrors(cudaMemcpy(ptr_scheduled, t_ptr.data(), sizeof(int) * num_target, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(target, t_rnk.data(), sizeof(int) * (num_target + 1), cudaMemcpyHostToDevice));
    free(h_ptr);
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    edgesort();
    neighbor_grouping(TARGET_SIZE);
    checkCudaErrors(cudaMemset(vout, 0, sizeof(float) * num_v * feat_in));

    // // 不做负载均衡的分块
    // block.x = BLOCK_SIZE;
    // block.y = 1;
    // block.z = 1;
    // grid.x = ceiling(num_v, X_BLOCK_SIZE);
    // grid.y = ceiling(feat_in, Y_BLOCK_SIZE);
    // grid.z = 1;

    // 做负载均衡的分块
    block.x = BLOCK_SIZE;
    block.y = 1;
    block.z = 1;
    grid.x = ceiling(num_target, X_BLOCK_SIZE);
    grid.y = ceiling(feat_in, Y_BLOCK_SIZE);
    grid.z = 1;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    spmm_kernel_opt_target<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, num_target, target, ptr_scheduled);
    cudaFree(ptr_scheduled);
    cudaFree(target);
    num_target = 0;
}