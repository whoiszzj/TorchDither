#include "image_dither_impl.cuh"


__device__ int get_target_flag(
        const int i,
        const int j,
        const int W
) {
    if (i == 0) {
        // the first row
        if (j == 0) {
            return 0;
        } else {
            return 1;
        }
    } else {
        if (j == 0) {
            return 2;
        } else if (j == W - 1) {
            return 3;
        } else {
            return 4;
        }
    }
}


__global__ void image_dither_kernel(
        const int H,
        const int W,
        float *__restrict__ image,
        int *__restrict__ lock_flags,
        int *__restrict__ dithered_image
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= H) {
        return;
    }
    const int i = row;
    for (int j = 0; j < W; ++j) {
        const int target_flag = get_target_flag(i, j, W);
        const int idx = i * W + j;
        while (atomicAdd(&lock_flags[idx], 0) != target_flag);
        const float old_pixel = atomicAdd(&image[idx], 0);
        const int new_pixel = (old_pixel < 0.5 ? 0 : 1);
        dithered_image[idx] = (new_pixel == 0 ? 0 : 1);
        const float quant_error = old_pixel - new_pixel;
        if (j + 1 < W) {
            const int next_idx = i * W + j + 1;
            atomicAdd(&image[next_idx], quant_error * 7.0 / 16.0);
            atomicAdd(&lock_flags[next_idx], 1);
        }
        if ((j - 1 >= 0) && (i + 1 < H)) {
            const int next_idx = (i + 1) * W + j - 1;
            atomicAdd(&image[next_idx], quant_error * 3.0 / 16.0);
            atomicAdd(&lock_flags[next_idx], 1);
        }
        if (i + 1 < H) {
            const int next_idx = (i + 1) * W + j;
            atomicAdd(&image[next_idx], quant_error * 5.0 / 16.0);
            atomicAdd(&lock_flags[next_idx], 1);
        }
        if ((j + 1 < W) && (i + 1 < H)) {
            const int next_idx = (i + 1) * W + j + 1;
            atomicAdd(&image[next_idx], quant_error * 1.0 / 16.0);
            atomicAdd(&lock_flags[next_idx], 1);
        }
    }
}

__global__ void scatter_coordinates(
        const int height, const int width, const int *dithered, const int *prefix_sum, int2 *xys
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int idx = y * width + x;

    if (dithered[idx] == 1) {
        int pos = prefix_sum[idx] - 1;
        xys[pos] = make_int2(x, y);
    }
}

torch::Tensor image_dither_cuda(
        const int H,
        const int W,
        const float *image
) {
    // define the lock flags
    int *lock_flags;
    cudaMalloc(&lock_flags, H * W * sizeof(int));
    cudaMemset(lock_flags, 0, H * W * sizeof(int));
    float *image_copy;
    cudaMalloc(&image_copy, H * W * sizeof(float));
    cudaMemcpy(image_copy, image, H * W * sizeof(float), cudaMemcpyDeviceToDevice);
    int *dithered_image;
    cudaMalloc(&dithered_image, H * W * sizeof(int));
    cudaMemset(dithered_image, 0, H * W * sizeof(int));

    image_dither_kernel << < (H + 255) / 256, 256 >> > (H, W, image_copy, lock_flags, dithered_image);
    CHECK_CUDA;

    int *prefix_sum;
    cudaMalloc(&prefix_sum, H * W * sizeof(int));
    cudaMemcpy(prefix_sum, dithered_image, H * W * sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::inclusive_scan(thrust::device, prefix_sum, prefix_sum + H * W, prefix_sum);

    int total_ones = 0;
    cudaMemcpy(&total_ones, prefix_sum + H * W - 1, sizeof(int), cudaMemcpyDeviceToHost);

    auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    torch::Tensor xys = torch::full({total_ones, 2}, 0, int_opts);
    int2 *xys_ptr = (int2 *) xys.contiguous().data_ptr<int>();
    const dim3 block(16, 16);
    const dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    scatter_coordinates << < grid, block >> > (H, W, dithered_image, prefix_sum, xys_ptr);
    CHECK_CUDA;

    cudaFree(lock_flags);
    cudaFree(image_copy);
    cudaFree(prefix_sum);
    cudaFree(dithered_image);
    return xys;
}