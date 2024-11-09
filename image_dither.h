#ifndef IMAGE_DITHER_H
#define IMAGE_DITHER_H

#include <torch/extension.h>
#include "image_dither_impl.cuh"

torch::Tensor image_dither_wrapper(
        const torch::Tensor &image  // [H, W]
);

#endif //IMAGE_DITHER_H