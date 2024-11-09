#include "image_dither.h"

torch::Tensor image_dither_wrapper(
        const torch::Tensor &image  // [H, W]
) {
    // define middle variables
    const int H = image.size(0);
    const int W = image.size(1);
    // asset image is on cuda
    TORCH_CHECK(image.is_cuda(), "image should be on cuda");
    auto xys = image_dither_cuda(
            H,
            W,
            image.contiguous().data<float>()
    ); // Cuda
    return xys;
}
