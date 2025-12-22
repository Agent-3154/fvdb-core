// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterize.cuh>
#include <fvdb/detail/ops/gsplat/GaussianRasterizeForward.h>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <nanovdb/math/Math.h>

#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <cuda/std/tuple>

#include <cstdint>
#include <optional>

namespace fvdb::detail::ops {
namespace {

// Structure to hold arguments and methods for the rasterize forward kernel
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED> struct RasterizeForwardArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, NUM_CHANNELS, IS_PACKED>;
    CommonArgs commonArgs;

    JaggedRAcc64<ScalarType, 2> mOutFeatures; // [[nPixels, NUM_CHANNELS]_0..._C]
    JaggedRAcc64<ScalarType, 2> mOutAlphas;   // [[nPixels, 1]_0..._C]
    JaggedRAcc64<int32_t, 1> mOutLastIds;     // [[nPixels]_0..._C]

    RasterizeForwardArgs(
        const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,       // [C, N] or [nnz]
        const torch::Tensor &features,        // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const std::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const std::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const uint32_t blockOffset,
        const torch::Tensor &tileOffsets,     // [C, numTilesH, numTilesW]
        const torch::Tensor &tileGaussianIds, // [totalIntersections]
        // output JaggedTensors:
        // In Dense mode, first dimension X = C * imageHeight * imageWidth
        // In Sparse mode, first dimension X = C * nPixels_i (i from 0 to C-1)
        const fvdb::JaggedTensor &outFeatures,                          // [X, NUM_CHANNELS]
        const fvdb::JaggedTensor &outAlphas,                            // [X, 1]
        const fvdb::JaggedTensor &outLastIds,                           // [X]
        const std::optional<torch::Tensor> &activeTiles = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &tilePixelMask =
            std::nullopt, // [AT, wordsPerTileBitmask] e.g. [AT, 4]
        const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &pixelMap        = std::nullopt)        // [AP]
        : commonArgs(means2d,
                     conics,
                     opacities,
                     features,
                     backgrounds,
                     masks,
                     imageWidth,
                     imageHeight,
                     imageOriginW,
                     imageOriginH,
                     tileSize,
                     blockOffset,
                     tileOffsets,
                     tileGaussianIds,
                     activeTiles,
                     tilePixelMask,
                     tilePixelCumsum,
                     pixelMap),
          mOutFeatures(initJaggedAccessor<ScalarType, 2>(outFeatures, "outFeatures")),
          mOutAlphas(initJaggedAccessor<ScalarType, 2>(outAlphas, "outAlphas")),
          mOutLastIds(initJaggedAccessor<int32_t, 1>(outLastIds, "outLastIds")) {}

    /// @brief Write the alpha value for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param alpha The alpha value to write
    __device__ void
    writeAlpha(uint64_t pixelIndex, ScalarType alpha) {
        mOutAlphas.data()[pixelIndex][0] = alpha;
    }

    /// @brief Write the last ID for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param lastId The last ID to write
    __device__ void
    writeLastId(uint64_t pixelIndex, int32_t lastId) {
        mOutLastIds.data()[pixelIndex] = lastId;
    }

    /// @brief Write the features for a pixel
    /// @param pixelIndex The index of the pixel
    /// @param f The function to write the features
    template <typename F>
    __device__ void
    writeFeatures(uint64_t pixelIndex, F &&f) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            mOutFeatures.data()[pixelIndex][k] = f(k);
        }
    }

    /// @brief Volume render a tile of Gaussians
    /// @param cameraId The ID of the camera
    /// @param firstGaussianIdInBlock The first Gaussian ID in the block
    /// @param lastGaussianIdInBlock The last Gaussian ID in the block
    /// @param blockSize The size of the block
    /// @param pixelIsActive Whether the pixel is active
    /// @param activePixelIndex The index of the active pixel
    /// @param row The row of the pixel
    /// @param col The column of the pixel
    __device__ void
    volumeRenderTileForward(const uint32_t cameraId,
                            const uint32_t row,
                            const uint32_t col,
                            const uint32_t firstGaussianIdInBlock,
                            const uint32_t lastGaussianIdInBlock,
                            const uint32_t blockSize,
                            const bool pixelIsActive,
                            const uint32_t activePixelIndex) {
        alignas(Gaussian2D<ScalarType>) extern __shared__ char s[];
        auto *sharedGaussians = reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        // NOTE: The accumulated transmittance is used in the backward pass, and
        // since it's a
        //       sum of many small numbers, we should really use double precision.
        //       However, this makes the backward pass 1.5x slower, so we stick with
        //       float for now and sort of just ignore small impact gaussians
        //       ¯\_(ツ)_/¯.
        ScalarType accumTransmittance = 1.0f;
        // index of most recent gaussian to write to this thread's pixel
        int32_t curIdx = -1;

        // We don't return right away if the pixel is not in the image since we want
        // to use this thread to load gaussians into shared memory
        bool done = !pixelIsActive;

        // Process Gaussians in batches of block size (i.e. one Gaussian per thread in the block)
        const uint32_t tidx = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;

        // (row, col) coordinates are relative to the specified image origin which may
        // be a crop so we need to add the origin to get the absolute pixel coordinates
        const ScalarType px = col + commonArgs.mImageOriginW + ScalarType{0.5f};
        const ScalarType py = row + commonArgs.mImageOriginH + ScalarType{0.5f};

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        ScalarType pixOut[NUM_CHANNELS] = {0.f};
        for (uint32_t b = 0; b < numBatches; ++b) {
            // Sync threads before we start integrating the next batch
            // If all threads are done, we can break early
            if (__syncthreads_count(done) == blockSize) {
                break;
            }

            // Each thread fetches one gaussian from front to back (mTileGaussianIds is depth
            // sorted)
            const uint32_t batchStart = firstGaussianIdInBlock + blockSize * b;
            const uint32_t idx        = batchStart + tidx;
            if (idx < lastGaussianIdInBlock) {
                const int32_t g =
                    commonArgs.mTileGaussianIds[idx]; // which gaussian we're rendering
                sharedGaussians[tidx] = commonArgs.getGaussian(g);
            }

            // Sync threads so all gaussians for this batch are loaded in shared
            // memory
            __syncthreads();

            // Volume render Gaussians in this batch
            if (pixelIsActive) { // skip inactive sparse pixels
                const uint32_t batchSize = min(blockSize, lastGaussianIdInBlock - batchStart);
                for (uint32_t t = 0; (t < batchSize) && !done; ++t) {
                    const Gaussian2D<ScalarType> gaussian = sharedGaussians[t];

                    const auto [gaussianIsValid, delta, expMinusSigma, alpha] =
                        commonArgs.evalGaussian(gaussian, px, py);

                    if (!gaussianIsValid) {
                        continue;
                    }

                    const ScalarType nextTransmittance = accumTransmittance * (1.0f - alpha);
                    if (nextTransmittance <= 1e-4f) { // this pixel is done: exclusive
                        done = true;
                        break;
                    }

                    const ScalarType vis       = alpha * accumTransmittance;
                    const auto featureAccessor = [&]() {
                        if constexpr (IS_PACKED) {
                            return commonArgs.mFeatures[gaussian.id];
                        } else {
                            const int32_t cid = gaussian.id / commonArgs.mNumGaussiansPerCamera;
                            const int32_t gid = gaussian.id % commonArgs.mNumGaussiansPerCamera;
                            return commonArgs.mFeatures[cid][gid];
                        }
                    }();
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                        pixOut[k] += featureAccessor[k] * vis;
                    }

                    curIdx             = batchStart + t;
                    accumTransmittance = nextTransmittance;
                }
            }
        }

        if (pixelIsActive) {
            const auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);
            writeAlpha(pixIdx, 1.0f - accumTransmittance);
            writeLastId(pixIdx, curIdx);
            writeFeatures(pixIdx, [&](uint32_t k) {
                return commonArgs.mHasBackgrounds
                           ? pixOut[k] + accumTransmittance * commonArgs.mBackgrounds[cameraId][k]
                           : pixOut[k];
            });
        }
    }
};

/// @brief Rasterize Gaussians to pixels
/// @param args The arguments for the rasterization
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansForward(RasterizeForwardArgs<ScalarType, NUM_CHANNELS, IS_PACKED> args) {
    auto &commonArgs = args.commonArgs;

    int32_t cameraId;
    int32_t tileRow;
    int32_t tileCol;
    uint32_t row;
    uint32_t col;

    cuda::std::tie(cameraId, tileRow, tileCol, row, col) =
        commonArgs.mIsSparse ? commonArgs.sparseCoordinates() : commonArgs.denseCoordinates();

    // NOTE: We keep threads which correspond to pixels outside the image bounds around
    //       to load gaussians from global memory, but they do not contribute to the output.

    // pixelInImage: Whether this pixel is inside the image bounds.
    // activePixelIndex: Index of this pixel in the output for the block if it is active (sparse
    // mode only).
    bool pixelInImage{false};
    uint32_t activePixelIndex{0};
    cuda::std::tie(pixelInImage, activePixelIndex) = commonArgs.activePixelIndex(row, col);

    if (commonArgs.mHasMasks && pixelInImage && !commonArgs.mMasks[cameraId][tileRow][tileCol]) {
        auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);
        args.writeFeatures(pixIdx, [&](uint32_t k) {
            return commonArgs.mHasBackgrounds ? commonArgs.mBackgrounds[cameraId][k] : 0.0f;
        });
        return;
    }

    int32_t firstGaussianIdInBlock;
    int32_t lastGaussianIdInBlock;
    cuda::std::tie(firstGaussianIdInBlock, lastGaussianIdInBlock) =
        commonArgs.tileGaussianRange(cameraId, tileRow, tileCol);

    args.volumeRenderTileForward(cameraId,
                                 row,
                                 col,
                                 firstGaussianIdInBlock,
                                 lastGaussianIdInBlock,
                                 blockDim.x * blockDim.y,
                                 pixelInImage,
                                 activePixelIndex);
}

/// @brief Get the shared memory requirements for the forward pass kernel
/// @param tileSize The size of the tile
/// @return The shared memory required in bytes
template <typename ScalarType>
size_t
getSharedMemRequirements(const size_t tileSize) {
    return tileSize * tileSize * sizeof(Gaussian2D<ScalarType>);
}

template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
launchRasterizeForwardKernel(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const std::optional<fvdb::JaggedTensor> &pixelsToRender = std::nullopt,
    const std::optional<torch::Tensor> &activeTiles         = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask       = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum     = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap            = std::nullopt) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    TORCH_CHECK_VALUE(tileOffsets.size(2) == (imageWidth + tileSize - 1) / tileSize,
                      "tileOffsets width must match the number of tiles in image size");
    TORCH_CHECK_VALUE(tileOffsets.size(1) == (imageHeight + tileSize - 1) / tileSize,
                      "tileOffsets height must match the number of tiles in image size");

    const bool packed = means2d.dim() == 2;

    const uint32_t C        = tileOffsets.size(0);          // number of cameras
    const uint32_t N        = packed ? 0 : means2d.size(1); // number of gaussians
    const uint32_t channels = features.size(-1);

    const uint32_t tileExtentH = tileOffsets.size(1);
    const uint32_t tileExtentW = tileOffsets.size(2);

    TORCH_CHECK_VALUE(pixelMap.has_value() == pixelsToRender.has_value(),
                      "pixelMap and pixelsToRender must be provided together");
    if (pixelMap.has_value()) {
        TORCH_CHECK_VALUE(pixelMap.value().size(0) == pixelsToRender.value().numel() / 2,
                          "pixelMap must have the same number of elements as pixelsToRender");
    }

    const auto sizes = pixelsToRender.has_value()
                           ? pixelsToRender.value().lsizes1()
                           : std::vector<int64_t>{C * imageHeight * imageWidth};
    std::vector<torch::Tensor> featuresToRenderVec;
    std::vector<torch::Tensor> alphasToRenderVec;
    std::vector<torch::Tensor> lastIdsToRenderVec;

    for (const auto &size: sizes) {
        featuresToRenderVec.push_back(
            torch::empty({size, channels}, features.options().dtype(torch::kFloat32)));
        alphasToRenderVec.push_back(
            torch::empty({size, 1}, features.options().dtype(torch::kFloat32)));
        lastIdsToRenderVec.push_back(torch::empty({size}, features.options().dtype(torch::kInt32)));
    }

    auto outFeatures = fvdb::JaggedTensor(featuresToRenderVec);
    auto outAlphas   = fvdb::JaggedTensor(alphasToRenderVec);
    auto outLastIds  = fvdb::JaggedTensor(lastIdsToRenderVec);

    auto args = RasterizeForwardArgs<ScalarType, NUM_CHANNELS, IS_PACKED>(means2d,
                                                                          conics,
                                                                          opacities,
                                                                          features,
                                                                          backgrounds,
                                                                          masks,
                                                                          imageWidth,
                                                                          imageHeight,
                                                                          imageOriginW,
                                                                          imageOriginH,
                                                                          tileSize,
                                                                          0,
                                                                          tileOffsets,
                                                                          tileGaussianIds,
                                                                          outFeatures,
                                                                          outAlphas,
                                                                          outLastIds,
                                                                          activeTiles,
                                                                          tilePixelMask,
                                                                          tilePixelCumsum,
                                                                          pixelMap);

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Thread blocks cooperatively cache a tile of Gaussians in shared memory
    const uint32_t sharedMem = getSharedMemRequirements<ScalarType>(tileSize);

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(rasterizeGaussiansForward<ScalarType, NUM_CHANNELS, IS_PACKED>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile_size.");
    }

    rasterizeGaussiansForward<<<args.commonArgs.getGridDim(),
                                args.commonArgs.getBlockDim(),
                                sharedMem,
                                stream>>>(args);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // In dense mode, we need to reshape the output tensors to the original image size
    // because they are packed into a single JaggedTensor so that the output code is the same
    // for dense and sparse modes.
    if (!args.commonArgs.mIsSparse) {
        outFeatures =
            fvdb::JaggedTensor(outFeatures.jdata().view({C, imageHeight, imageWidth, channels}));
        outAlphas  = fvdb::JaggedTensor(outAlphas.jdata().view({C, imageHeight, imageWidth, 1}));
        outLastIds = fvdb::JaggedTensor(outLastIds.jdata().view({C, imageHeight, imageWidth}));
    }

    return std::make_tuple(outFeatures, outAlphas, outLastIds);
}

template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
launchRasterizeForwardKernels(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const std::optional<fvdb::JaggedTensor> &pixelsToRender = std::nullopt,
    const std::optional<torch::Tensor> &activeTiles         = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask       = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum     = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap            = std::nullopt) {
    TORCH_CHECK_VALUE(tileOffsets.size(2) == (imageWidth + tileSize - 1) / tileSize,
                      "tileOffsets width must match the number of tiles in image size");
    TORCH_CHECK_VALUE(tileOffsets.size(1) == (imageHeight + tileSize - 1) / tileSize,
                      "tileOffsets height must match the number of tiles in image size");

    const bool packed = means2d.dim() == 2;

    const uint32_t C        = tileOffsets.size(0);          // number of cameras
    const uint32_t N        = packed ? 0 : means2d.size(1); // number of gaussians
    const uint32_t channels = features.size(-1);

    const uint32_t tileExtentH = tileOffsets.size(1);
    const uint32_t tileExtentW = tileOffsets.size(2);

    TORCH_CHECK_VALUE(pixelMap.has_value() == pixelsToRender.has_value(),
                      "pixelMap and pixelsToRender must be provided together");
    if (pixelMap.has_value()) {
        TORCH_CHECK_VALUE(pixelMap.value().size(0) == pixelsToRender.value().numel() / 2,
                          "pixelMap must have the same number of elements as pixelsToRender");
    }

    const auto sizes = pixelsToRender.has_value()
                           ? pixelsToRender.value().lsizes1()
                           : std::vector<int64_t>{C * imageHeight * imageWidth};
    std::vector<torch::Tensor> featuresToRenderVec;
    std::vector<torch::Tensor> alphasToRenderVec;
    std::vector<torch::Tensor> lastIdsToRenderVec;

    for (const auto &size: sizes) {
        featuresToRenderVec.push_back(
            torch::empty({size, channels}, features.options().dtype(torch::kFloat32)));
        alphasToRenderVec.push_back(
            torch::empty({size, 1}, features.options().dtype(torch::kFloat32)));
        lastIdsToRenderVec.push_back(torch::empty({size}, features.options().dtype(torch::kInt32)));
    }

    auto outFeatures = fvdb::JaggedTensor(featuresToRenderVec);
    auto outAlphas   = fvdb::JaggedTensor(alphasToRenderVec);
    auto outLastIds  = fvdb::JaggedTensor(lastIdsToRenderVec);

    auto isSparse      = activeTiles.has_value();
    uint32_t tileCount = isSparse ? activeTiles.value().size(0) : C * tileExtentH * tileExtentW;
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        uint32_t deviceTileOffset, deviceTileCount;
        std::tie(deviceTileOffset, deviceTileCount) = deviceChunk(tileCount, deviceId);

        if (deviceTileCount) {
            auto args = RasterizeForwardArgs<ScalarType, NUM_CHANNELS, IS_PACKED>(means2d,
                                                                                  conics,
                                                                                  opacities,
                                                                                  features,
                                                                                  backgrounds,
                                                                                  masks,
                                                                                  imageWidth,
                                                                                  imageHeight,
                                                                                  imageOriginW,
                                                                                  imageOriginH,
                                                                                  tileSize,
                                                                                  deviceTileOffset,
                                                                                  tileOffsets,
                                                                                  tileGaussianIds,
                                                                                  outFeatures,
                                                                                  outAlphas,
                                                                                  outLastIds,
                                                                                  activeTiles,
                                                                                  tilePixelMask,
                                                                                  tilePixelCumsum,
                                                                                  pixelMap);

            TORCH_CHECK(means2d.is_contiguous());
            TORCH_CHECK(conics.is_contiguous());
            TORCH_CHECK(opacities.is_contiguous());
            TORCH_CHECK(features.is_contiguous());

            nanovdb::util::cuda::memPrefetchAsync(means2d.const_data_ptr<ScalarType>(),
                                                  means2d.numel() * sizeof(ScalarType),
                                                  deviceId,
                                                  stream);
            nanovdb::util::cuda::memPrefetchAsync(conics.const_data_ptr<ScalarType>(),
                                                  conics.numel() * sizeof(ScalarType),
                                                  deviceId,
                                                  stream);
            nanovdb::util::cuda::memPrefetchAsync(opacities.const_data_ptr<ScalarType>(),
                                                  opacities.numel() * sizeof(ScalarType),
                                                  deviceId,
                                                  stream);
            nanovdb::util::cuda::memPrefetchAsync(features.const_data_ptr<ScalarType>(),
                                                  features.numel() * sizeof(ScalarType),
                                                  deviceId,
                                                  stream);

            // Thread blocks cooperatively cache a tile of Gaussians in shared memory
            const uint32_t sharedMem = getSharedMemRequirements<ScalarType>(tileSize);

            // TODO: an optimization can be done by passing the actual number of
            // channels into the kernel functions and avoid necessary global memory
            // writes. This requires moving the channel padding from python to C side.
            if (cudaFuncSetAttribute(rasterizeGaussiansForward<ScalarType, NUM_CHANNELS, IS_PACKED>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     sharedMem) != cudaSuccess) {
                AT_ERROR("Failed to set maximum shared memory size (requested ",
                         sharedMem,
                         " bytes), try lowering tile_size.");
            }

            const dim3 blockDim = {tileSize, tileSize, 1};
            const dim3 gridDim  = {deviceTileCount, 1, 1};

            rasterizeGaussiansForward<<<gridDim, blockDim, sharedMem, stream>>>(args);

            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    mergeStreams();

    // In dense mode, we need to reshape the output tensors to the original image size
    // because they are packed into a single JaggedTensor so that the output code is the same
    // for dense and sparse modes.
    if (!isSparse) {
        outFeatures =
            fvdb::JaggedTensor(outFeatures.jdata().view({C, imageHeight, imageWidth, channels}));
        outAlphas  = fvdb::JaggedTensor(outAlphas.jdata().view({C, imageHeight, imageWidth, 1}));
        outLastIds = fvdb::JaggedTensor(outLastIds.jdata().view({C, imageHeight, imageWidth}));
    }

    return std::make_tuple(outFeatures, outAlphas, outLastIds);
}

// Structure to hold arguments and methods for the tile sparse rasterize forward kernel
// This variant writes to regular tensors [C, T, tile_size, tile_size, D] instead of JaggedTensors
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
struct RasterizeForwardTileSparseArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, NUM_CHANNELS, IS_PACKED>;
    
    // Store dummy tensors for tilePixelMask and tilePixelCumsum to ensure they outlive commonArgs
    // These must be declared before commonArgs so they're initialized first
    torch::Tensor mTilePixelMaskDummy;
    torch::Tensor mTilePixelCumsumDummy;
    
    CommonArgs commonArgs;

    // Regular tensor outputs: [C, T, tile_size, tile_size, NUM_CHANNELS]
    using OutFeaturesAccessor = TorchRAcc64<ScalarType, 5>;
    using OutAlphasAccessor = TorchRAcc64<ScalarType, 5>;
    using OutLastIdsAccessor = TorchRAcc64<int32_t, 4>;
    using TileToOutputIdxAccessor = TorchRAcc64<int32_t, 3>;
    
    OutFeaturesAccessor mOutFeatures; // [C, T, tile_size, tile_size, NUM_CHANNELS]
    OutAlphasAccessor mOutAlphas;   // [C, T, tile_size, tile_size, 1]
    OutLastIdsAccessor mOutLastIds;     // [C, T, tile_size, tile_size]
    
    // Mapping from (camera_id, tile_row, tile_col) to output tile index t
    // Shape: [C, numTilesH, numTilesW], value is output tile index or -1 if not rendered
    TileToOutputIdxAccessor mTileToOutputIdx; // [C, numTilesH, numTilesW]
    uint32_t mNumTilesPerCamera; // T - number of tiles per camera

    RasterizeForwardTileSparseArgs(
        const torch::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,          // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,       // [C, N] or [nnz]
        const torch::Tensor &features,        // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const std::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const std::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const uint32_t blockOffset,
        const torch::Tensor &tileOffsets,     // [C, numTilesH, numTilesW]
        const torch::Tensor &tileGaussianIds, // [totalIntersections]
        const torch::Tensor &outFeatures,     // [C, T, tile_size, tile_size, NUM_CHANNELS]
        const torch::Tensor &outAlphas,       // [C, T, tile_size, tile_size, 1]
        const torch::Tensor &outLastIds,      // [C, T, tile_size, tile_size]
        const torch::Tensor &tileToOutputIdx, // [C, numTilesH, numTilesW]
        const uint32_t numTilesPerCamera,     // T
        const torch::Tensor &activeTiles)      // [AT]
        : // Create dummy tilePixelMask and tilePixelCumsum for tile sparse rendering
          // Since we render all pixels in each tile, all pixels are "active"
          mTilePixelMaskDummy([&]() {
              const uint32_t numActiveTiles = activeTiles.size(0);
              const uint32_t wordsPerTile = numWordsPerTileBitmask(tileSize);
              // Create tilePixelMask with all bits set (all pixels active)
              // Shape: [numActiveTiles, wordsPerTile]
              return torch::ones({numActiveTiles, wordsPerTile},
                                 activeTiles.options().dtype(torch::kUInt64));
          }()),
          mTilePixelCumsumDummy([&]() {
              const uint32_t numActiveTiles = activeTiles.size(0);
              const uint32_t pixelsPerTile = tileSize * tileSize;
              // Create tilePixelCumsum: [0, pixelsPerTile, 2*pixelsPerTile, ...]
              // Shape: [numActiveTiles]
              return torch::arange(0, numActiveTiles * pixelsPerTile, pixelsPerTile,
                                   activeTiles.options().dtype(torch::kInt64));
          }()),
          commonArgs(means2d,
                     conics,
                     opacities,
                     features,
                     backgrounds,
                     masks,
                     imageWidth,
                     imageHeight,
                     imageOriginW,
                     imageOriginH,
                     tileSize,
                     blockOffset,
                     tileOffsets,
                     tileGaussianIds,
                     activeTiles,
                     mTilePixelMaskDummy,  // All pixels active in each tile
                     mTilePixelCumsumDummy, // Cumulative sum for all pixels
                     std::nullopt),  // pixelMap not needed
          mOutFeatures(initAccessor<ScalarType, 5>(outFeatures, "outFeatures")),
          mOutAlphas(initAccessor<ScalarType, 5>(outAlphas, "outAlphas")),
          mOutLastIds(initAccessor<int32_t, 4>(outLastIds, "outLastIds")),
          mTileToOutputIdx(initAccessor<int32_t, 3>(tileToOutputIdx, "tileToOutputIdx")),
          mNumTilesPerCamera(numTilesPerCamera) {
        // Validate output tensor shapes
        TORCH_CHECK_VALUE(outFeatures.size(0) == commonArgs.mNumCameras, "outFeatures camera dimension mismatch");
        TORCH_CHECK_VALUE(outFeatures.size(1) == numTilesPerCamera, "outFeatures tile dimension mismatch");
        TORCH_CHECK_VALUE(outFeatures.size(2) == commonArgs.mTileSize, "outFeatures tile_size dimension mismatch");
        TORCH_CHECK_VALUE(outFeatures.size(3) == commonArgs.mTileSize, "outFeatures tile_size dimension mismatch");
        TORCH_CHECK_VALUE(outFeatures.size(4) == NUM_CHANNELS, "outFeatures channel dimension mismatch");
    }

    /// @brief Write the alpha value for a pixel
    /// @param cameraId Camera ID
    /// @param tileIdx Output tile index (0 to T-1)
    /// @param pixelY Pixel Y coordinate within tile (0 to tile_size-1)
    /// @param pixelX Pixel X coordinate within tile (0 to tile_size-1)
    /// @param alpha The alpha value to write
    __device__ void
    writeAlpha(uint32_t cameraId, uint32_t tileIdx, uint32_t pixelY, uint32_t pixelX, ScalarType alpha) {
        mOutAlphas[cameraId][tileIdx][pixelY][pixelX][0] = alpha;
    }

    /// @brief Write the last ID for a pixel
    /// @param cameraId Camera ID
    /// @param tileIdx Output tile index (0 to T-1)
    /// @param pixelY Pixel Y coordinate within tile (0 to tile_size-1)
    /// @param pixelX Pixel X coordinate within tile (0 to tile_size-1)
    /// @param lastId The last ID to write
    __device__ void
    writeLastId(uint32_t cameraId, uint32_t tileIdx, uint32_t pixelY, uint32_t pixelX, int32_t lastId) {
        mOutLastIds[cameraId][tileIdx][pixelY][pixelX] = lastId;
    }

    /// @brief Write the features for a pixel
    /// @param cameraId Camera ID
    /// @param tileIdx Output tile index (0 to T-1)
    /// @param pixelY Pixel Y coordinate within tile (0 to tile_size-1)
    /// @param pixelX Pixel X coordinate within tile (0 to tile_size-1)
    /// @param f The function to write the features
    template <typename F>
    __device__ void
    writeFeatures(uint32_t cameraId, uint32_t tileIdx, uint32_t pixelY, uint32_t pixelX, F &&f) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            mOutFeatures[cameraId][tileIdx][pixelY][pixelX][k] = f(k);
        }
    }

    /// @brief Volume render a tile of Gaussians
    /// @param cameraId The ID of the camera
    /// @param tileRow The row of the tile in the image
    /// @param tileCol The column of the tile in the image
    /// @param tileIdx The output tile index (0 to T-1)
    /// @param row The row of the pixel in the image
    /// @param col The column of the pixel in the image
    /// @param firstGaussianIdInBlock The first Gaussian ID in the block
    /// @param lastGaussianIdInBlock The last Gaussian ID in the block
    /// @param blockSize The size of the block
    __device__ void
    volumeRenderTileForward(const uint32_t cameraId,
                            const uint32_t tileRow,
                            const uint32_t tileCol,
                            const uint32_t tileIdx,
                            const uint32_t row,
                            const uint32_t col,
                            const uint32_t firstGaussianIdInBlock,
                            const uint32_t lastGaussianIdInBlock,
                            const uint32_t blockSize) {
        alignas(Gaussian2D<ScalarType>) extern __shared__ char s[];
        auto *sharedGaussians = reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        ScalarType accumTransmittance = 1.0f;
        int32_t curIdx = -1;

        // Check if pixel is within image bounds
        const bool pixelInImage = (row < commonArgs.mImageHeight && col < commonArgs.mImageWidth);
        if (!pixelInImage) {
            // Still need to participate in loading Gaussians, but don't write output
            // Process Gaussians in batches
            const uint32_t tidx = threadIdx.y * blockDim.x + threadIdx.x;
            const uint32_t numBatches =
                (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;
            for (uint32_t batch = 0; batch < numBatches; ++batch) {
                const uint32_t batchStart = firstGaussianIdInBlock + batch * blockSize;
                const uint32_t batchEnd   = min(batchStart + blockSize, lastGaussianIdInBlock);
                if (tidx < batchEnd - batchStart) {
                    const uint32_t gaussianIdx = batchStart + tidx;
                    sharedGaussians[tidx]      = commonArgs.getGaussian(gaussianIdx);
                }
                __syncthreads();

                const uint32_t numGaussiansInBatch = batchEnd - batchStart;
                for (uint32_t t = 0; t < numGaussiansInBatch; ++t) {
                    const Gaussian2D<ScalarType> &gaussian = sharedGaussians[t];
                    // Skip evaluation for out-of-bounds pixels
                }
                __syncthreads();
            }
            return;
        }

        // Pixel coordinates relative to tile
        const uint32_t pixelY = threadIdx.y;
        const uint32_t pixelX = threadIdx.x;

        // Process Gaussians in batches of block size
        const uint32_t tidx = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;

        ScalarType pixOut[NUM_CHANNELS];
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            pixOut[k] = 0.0f;
        }

        for (uint32_t batch = 0; batch < numBatches; ++batch) {
            const uint32_t batchStart = firstGaussianIdInBlock + batch * blockSize;
            const uint32_t batchEnd   = min(batchStart + blockSize, lastGaussianIdInBlock);
            if (tidx < batchEnd - batchStart) {
                const uint32_t gaussianIdx = batchStart + tidx;
                sharedGaussians[tidx]      = commonArgs.getGaussian(gaussianIdx);
            }
            __syncthreads();

            const uint32_t numGaussiansInBatch = batchEnd - batchStart;
            for (uint32_t t = 0; t < numGaussiansInBatch; ++t) {
                const Gaussian2D<ScalarType> &gaussian = sharedGaussians[t];
                const ScalarType px                    = static_cast<ScalarType>(col) + 0.5f;
                const ScalarType py                    = static_cast<ScalarType>(row) + 0.5f;

                const auto [gaussianIsValid, delta, expMinusSigma, alpha] =
                    commonArgs.evalGaussian(gaussian, px, py);

                if (gaussianIsValid) {
                    const ScalarType nextTransmittance = accumTransmittance * (1.0f - alpha);
                    if (nextTransmittance < 1e-4f) {
                        // Early termination
                        break;
                    }

                    const ScalarType vis = alpha * accumTransmittance;
                    const auto featureAccessor = [&]() {
                        if constexpr (IS_PACKED) {
                            return commonArgs.mFeatures[gaussian.id];
                        } else {
                            const int32_t cid = gaussian.id / commonArgs.mNumGaussiansPerCamera;
                            const int32_t gid = gaussian.id % commonArgs.mNumGaussiansPerCamera;
                            return commonArgs.mFeatures[cid][gid];
                        }
                    }();
                    PRAGMA_UNROLL
                    for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                        pixOut[k] += featureAccessor[k] * vis;
                    }

                    curIdx             = batchStart + t;
                    accumTransmittance = nextTransmittance;
                }
            }
        }

        // Write output
        writeAlpha(cameraId, tileIdx, pixelY, pixelX, 1.0f - accumTransmittance);
        writeLastId(cameraId, tileIdx, pixelY, pixelX, curIdx);
        writeFeatures(cameraId, tileIdx, pixelY, pixelX, [&](uint32_t k) {
            return commonArgs.mHasBackgrounds
                       ? pixOut[k] + accumTransmittance * commonArgs.mBackgrounds[cameraId][k]
                       : pixOut[k];
        });
    }
};

/// @brief Rasterize Gaussians to tiles (sparse tile rendering with regular tensor output)
/// @param args The arguments for the rasterization
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansForwardTileSparse(RasterizeForwardTileSparseArgs<ScalarType, NUM_CHANNELS, IS_PACKED> args) {
    auto &commonArgs = args.commonArgs;

    // Get camera and tile information from active tiles
    const int32_t tileOrdinal = blockIdx.x + commonArgs.mBlockOffset;
    const int32_t globalTile  = commonArgs.mActiveTiles[tileOrdinal];
    
    const int32_t cameraId = globalTile / (commonArgs.mNumTilesW * commonArgs.mNumTilesH);
    const int32_t tileId  = globalTile % (commonArgs.mNumTilesW * commonArgs.mNumTilesH);
    
    const int32_t tileRow = tileId / commonArgs.mNumTilesW;
    const int32_t tileCol = tileId % commonArgs.mNumTilesW;
    
    // Get output tile index from mapping
    const int32_t tileIdx = args.mTileToOutputIdx[cameraId][tileRow][tileCol];
    if (tileIdx < 0) {
        // This tile is not in the output (shouldn't happen if activeTiles is correct)
        return;
    }
    
    // Pixel coordinates in the image
    const uint32_t row = tileRow * commonArgs.mTileSize + threadIdx.y;
    const uint32_t col = tileCol * commonArgs.mTileSize + threadIdx.x;

    // Get Gaussian range for this tile
    int32_t firstGaussianIdInBlock;
    int32_t lastGaussianIdInBlock;
    cuda::std::tie(firstGaussianIdInBlock, lastGaussianIdInBlock) =
        commonArgs.tileGaussianRange(cameraId, tileRow, tileCol);

    args.volumeRenderTileForward(cameraId,
                                 tileRow,
                                 tileCol,
                                 static_cast<uint32_t>(tileIdx),
                                 row,
                                 col,
                                 firstGaussianIdInBlock,
                                 lastGaussianIdInBlock,
                                 blockDim.x * blockDim.y);
}

/// @brief Launch the tile sparse rasterization kernel
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launchRasterizeForwardTileSparseKernel(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, channels] or [nnz, channels]
    const torch::Tensor &opacities,                 // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const std::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    // tile mapping
    const torch::Tensor &tileToOutputIdx, // [C, tile_height, tile_width]
    const uint32_t numTilesPerCamera,     // T
    const torch::Tensor &activeTiles) {    // [AT]
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));

    TORCH_CHECK_VALUE(tileOffsets.size(2) == (imageWidth + tileSize - 1) / tileSize,
                      "tileOffsets width must match the number of tiles in image size");
    TORCH_CHECK_VALUE(tileOffsets.size(1) == (imageHeight + tileSize - 1) / tileSize,
                      "tileOffsets height must match the number of tiles in image size");

    const uint32_t C        = tileOffsets.size(0);          // number of cameras
    const uint32_t channels = features.size(-1);
    const uint32_t tileExtentH = tileOffsets.size(1);
    const uint32_t tileExtentW = tileOffsets.size(2);

    TORCH_CHECK_VALUE(tileToOutputIdx.size(0) == C, "tileToOutputIdx camera dimension mismatch");
    TORCH_CHECK_VALUE(tileToOutputIdx.size(1) == tileExtentH, "tileToOutputIdx height dimension mismatch");
    TORCH_CHECK_VALUE(tileToOutputIdx.size(2) == tileExtentW, "tileToOutputIdx width dimension mismatch");

    TORCH_CHECK_VALUE(activeTiles.size(0) > 0, "activeTiles must not be empty");
    TORCH_CHECK_VALUE(numTilesPerCamera > 0, "numTilesPerCamera must be positive");

    // Allocate output tensors internally
    auto outFeatures = torch::empty({C, numTilesPerCamera, tileSize, tileSize, channels},
                                     features.options().dtype(torch::kFloat32));
    auto outAlphas   = torch::empty({C, numTilesPerCamera, tileSize, tileSize, 1},
                                     features.options().dtype(torch::kFloat32));
    auto outLastIds  = torch::empty({C, numTilesPerCamera, tileSize, tileSize},
                                     features.options().dtype(torch::kInt32));

    auto args = RasterizeForwardTileSparseArgs<ScalarType, NUM_CHANNELS, IS_PACKED>(
        means2d,
        conics,
        opacities,
        features,
        backgrounds,
        masks,
        imageWidth,
        imageHeight,
        imageOriginW,
        imageOriginH,
        tileSize,
        0, // blockOffset
        tileOffsets,
        tileGaussianIds,
        outFeatures,
        outAlphas,
        outLastIds,
        tileToOutputIdx,
        numTilesPerCamera,
        activeTiles);

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Thread blocks cooperatively cache a tile of Gaussians in shared memory
    const uint32_t sharedMem = getSharedMemRequirements<ScalarType>(tileSize);

    if (cudaFuncSetAttribute(rasterizeGaussiansForwardTileSparse<ScalarType, NUM_CHANNELS, IS_PACKED>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile_size.");
    }

    // Launch one block per active tile
    const uint32_t numActiveTiles = activeTiles.size(0);
    const dim3 blockDim = {tileSize, tileSize, 1};
    const dim3 gridDim  = {numActiveTiles, 1, 1};

    rasterizeGaussiansForwardTileSparse<<<gridDim, blockDim, sharedMem, stream>>>(args);

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(outFeatures, outAlphas, outLastIds);
}

} // namespace

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward<torch::kCUDA>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,              // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds,          // [n_isects]
    const at::optional<torch::Tensor> &backgrounds // [C, D]
) {
    FVDB_FUNC_RANGE();
    const uint32_t channels = features.size(-1);
    const bool isPacked     = means2d.dim() == 2;

    const std::optional<torch::Tensor> masks = std::nullopt;

#define CALL_FWD_CUDA(N)                                                                        \
    case N: {                                                                                   \
        if (isPacked) {                                                                         \
            auto [outFeatures, outAlphas, outLastIds] =                                         \
                launchRasterizeForwardKernel<float, N, true>(means2d,                           \
                                                             conics,                            \
                                                             features,                          \
                                                             opacities,                         \
                                                             backgrounds,                       \
                                                             masks,                             \
                                                             imageWidth,                        \
                                                             imageHeight,                       \
                                                             imageOriginW,                      \
                                                             imageOriginH,                      \
                                                             tileSize,                          \
                                                             tileOffsets,                       \
                                                             tileGaussianIds);                  \
            return std::make_tuple(outFeatures.jdata(), outAlphas.jdata(), outLastIds.jdata()); \
        } else {                                                                                \
            auto [outFeatures, outAlphas, outLastIds] =                                         \
                launchRasterizeForwardKernel<float, N, false>(means2d,                          \
                                                              conics,                           \
                                                              features,                         \
                                                              opacities,                        \
                                                              backgrounds,                      \
                                                              masks,                            \
                                                              imageWidth,                       \
                                                              imageHeight,                      \
                                                              imageOriginW,                     \
                                                              imageOriginH,                     \
                                                              tileSize,                         \
                                                              tileOffsets,                      \
                                                              tileGaussianIds);                 \
            return std::make_tuple(outFeatures.jdata(), outAlphas.jdata(), outLastIds.jdata()); \
        }                                                                                       \
    }

    // Make channels a compile time constant and do everything in register space
    // but at the expense of making this code ugly. NOTE: We do powers of two and
    // powers of two plus one to handle rendering common feature channel
    // dimensions with an optional additional depth channel
    switch (channels) {
        CALL_FWD_CUDA(1)
        CALL_FWD_CUDA(2)
        CALL_FWD_CUDA(3)
        CALL_FWD_CUDA(4)
        CALL_FWD_CUDA(5)
        CALL_FWD_CUDA(8)
        CALL_FWD_CUDA(9)
        CALL_FWD_CUDA(16)
        CALL_FWD_CUDA(17)
        CALL_FWD_CUDA(32)
        CALL_FWD_CUDA(33)
        CALL_FWD_CUDA(64)
        CALL_FWD_CUDA(65)
        CALL_FWD_CUDA(128)
        CALL_FWD_CUDA(129)
        CALL_FWD_CUDA(192)
        CALL_FWD_CUDA(193)
        CALL_FWD_CUDA(256)
        CALL_FWD_CUDA(257)
        CALL_FWD_CUDA(512)
        CALL_FWD_CUDA(513)
    default: AT_ERROR("Unsupported number of channels: ", channels);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward<torch::kPrivateUse1>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,              // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds,          // [n_isects]
    const at::optional<torch::Tensor> &backgrounds // [C, D]
) {
    FVDB_FUNC_RANGE();
    const uint32_t channels = features.size(-1);
    const bool isPacked     = means2d.dim() == 2;

    const std::optional<torch::Tensor> masks = std::nullopt;

#define CALL_FWD_PRIVATEUSE1(N)                                                                 \
    case N: {                                                                                   \
        if (isPacked) {                                                                         \
            auto [outFeatures, outAlphas, outLastIds] =                                         \
                launchRasterizeForwardKernels<float, N, true>(means2d,                          \
                                                              conics,                           \
                                                              features,                         \
                                                              opacities,                        \
                                                              backgrounds,                      \
                                                              masks,                            \
                                                              imageWidth,                       \
                                                              imageHeight,                      \
                                                              imageOriginW,                     \
                                                              imageOriginH,                     \
                                                              tileSize,                         \
                                                              tileOffsets,                      \
                                                              tileGaussianIds);                 \
            return std::make_tuple(outFeatures.jdata(), outAlphas.jdata(), outLastIds.jdata()); \
        } else {                                                                                \
            auto [outFeatures, outAlphas, outLastIds] =                                         \
                launchRasterizeForwardKernels<float, N, false>(means2d,                         \
                                                               conics,                          \
                                                               features,                        \
                                                               opacities,                       \
                                                               backgrounds,                     \
                                                               masks,                           \
                                                               imageWidth,                      \
                                                               imageHeight,                     \
                                                               imageOriginW,                    \
                                                               imageOriginH,                    \
                                                               tileSize,                        \
                                                               tileOffsets,                     \
                                                               tileGaussianIds);                \
            return std::make_tuple(outFeatures.jdata(), outAlphas.jdata(), outLastIds.jdata()); \
        }                                                                                       \
    }

    // Make channels a compile time constant and do everything in register space
    // but at the expense of making this code ugly. NOTE: We do powers of two and
    // powers of two plus one to handle rendering common feature channel
    // dimensions with an optional additional depth channel
    switch (channels) {
        CALL_FWD_PRIVATEUSE1(1)
        CALL_FWD_PRIVATEUSE1(2)
        CALL_FWD_PRIVATEUSE1(3)
        CALL_FWD_PRIVATEUSE1(4)
        CALL_FWD_PRIVATEUSE1(5)
        CALL_FWD_PRIVATEUSE1(8)
        CALL_FWD_PRIVATEUSE1(9)
        CALL_FWD_PRIVATEUSE1(16)
        CALL_FWD_PRIVATEUSE1(17)
        CALL_FWD_PRIVATEUSE1(32)
        CALL_FWD_PRIVATEUSE1(33)
        CALL_FWD_PRIVATEUSE1(64)
        CALL_FWD_PRIVATEUSE1(65)
        CALL_FWD_PRIVATEUSE1(128)
        CALL_FWD_PRIVATEUSE1(129)
        CALL_FWD_PRIVATEUSE1(192)
        CALL_FWD_PRIVATEUSE1(193)
        CALL_FWD_PRIVATEUSE1(256)
        CALL_FWD_PRIVATEUSE1(257)
        CALL_FWD_PRIVATEUSE1(512)
        CALL_FWD_PRIVATEUSE1(513)
    default: AT_ERROR("Unsupported number of channels: ", channels);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianRasterizeForward<torch::kCPU>(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    // image size
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,              // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds,          // [n_isects]
    const at::optional<torch::Tensor> &backgrounds // [C, D]
) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeForward<torch::kCUDA>(
    // sparse pixel coordinates
    const fvdb::JaggedTensor &pixelsToRender, // [C, maxPixelsPerCamera, 2]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap,
    const at::optional<torch::Tensor> &backgrounds) {
    FVDB_FUNC_RANGE();
    const uint32_t channels = features.size(-1);
    const bool isPacked     = means2d.dim() == 2;

    const std::optional<torch::Tensor> masks = std::nullopt;

#define CALL_FWD_SPARSE_CUDA(N)                                                   \
    case N: {                                                                     \
        if (isPacked) {                                                           \
            return launchRasterizeForwardKernel<float, N, true>(means2d,          \
                                                                conics,           \
                                                                features,         \
                                                                opacities,        \
                                                                backgrounds,      \
                                                                masks,            \
                                                                imageWidth,       \
                                                                imageHeight,      \
                                                                imageOriginW,     \
                                                                imageOriginH,     \
                                                                tileSize,         \
                                                                tileOffsets,      \
                                                                tileGaussianIds,  \
                                                                pixelsToRender,   \
                                                                activeTiles,      \
                                                                tilePixelMask,    \
                                                                tilePixelCumsum,  \
                                                                pixelMap);        \
        } else {                                                                  \
            return launchRasterizeForwardKernel<float, N, false>(means2d,         \
                                                                 conics,          \
                                                                 features,        \
                                                                 opacities,       \
                                                                 backgrounds,     \
                                                                 masks,           \
                                                                 imageWidth,      \
                                                                 imageHeight,     \
                                                                 imageOriginW,    \
                                                                 imageOriginH,    \
                                                                 tileSize,        \
                                                                 tileOffsets,     \
                                                                 tileGaussianIds, \
                                                                 pixelsToRender,  \
                                                                 activeTiles,     \
                                                                 tilePixelMask,   \
                                                                 tilePixelCumsum, \
                                                                 pixelMap);       \
        }                                                                         \
    }

    // Make channels a compile time constant and do everything in register space
    // but at the expense of making this code ugly. NOTE: We do powers of two and
    // powers of two plus one to handle rendering common feature channel
    // dimensions with an optional additional depth channel
    switch (channels) {
        CALL_FWD_SPARSE_CUDA(1)
        CALL_FWD_SPARSE_CUDA(2)
        CALL_FWD_SPARSE_CUDA(3)
        CALL_FWD_SPARSE_CUDA(4)
        CALL_FWD_SPARSE_CUDA(5)
        CALL_FWD_SPARSE_CUDA(8)
        CALL_FWD_SPARSE_CUDA(9)
        CALL_FWD_SPARSE_CUDA(16)
        CALL_FWD_SPARSE_CUDA(17)
        CALL_FWD_SPARSE_CUDA(32)
        CALL_FWD_SPARSE_CUDA(33)
        CALL_FWD_SPARSE_CUDA(64)
        CALL_FWD_SPARSE_CUDA(65)
        CALL_FWD_SPARSE_CUDA(128)
        CALL_FWD_SPARSE_CUDA(129)
        CALL_FWD_SPARSE_CUDA(192)
        CALL_FWD_SPARSE_CUDA(193)
        CALL_FWD_SPARSE_CUDA(256)
        CALL_FWD_SPARSE_CUDA(257)
        CALL_FWD_SPARSE_CUDA(512)
        CALL_FWD_SPARSE_CUDA(513)
    default: AT_ERROR("Unsupported number of channels: ", channels);
    }
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeForward<torch::kPrivateUse1>(
    // sparse pixel coordinates
    const fvdb::JaggedTensor &pixelsToRender, // [C, maxPixelsPerCamera, 2]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap,
    const at::optional<torch::Tensor> &backgrounds) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "PrivateUse1 implementation not available");
}

template <>
std::tuple<fvdb::JaggedTensor, fvdb::JaggedTensor, fvdb::JaggedTensor>
dispatchGaussianSparseRasterizeForward<torch::kCPU>(
    // sparse pixel coordinates
    const fvdb::JaggedTensor &pixelsToRender, // [C, maxPixelsPerCamera, 2]
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2]
    const torch::Tensor &conics,    // [C, N, 3]
    const torch::Tensor &features,  // [C, N, D]
    const torch::Tensor &opacities, // [N]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,     // [C, tile_height, tile_width]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const torch::Tensor &activeTiles,
    const torch::Tensor &tilePixelMask,
    const torch::Tensor &tilePixelCumsum,
    const torch::Tensor &pixelMap,
    const at::optional<torch::Tensor> &backgrounds) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianTileSparseRasterizeForward<torch::kCUDA>(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &features,
    const torch::Tensor &opacities,
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
    const torch::Tensor &tileToOutputIdx,
    const uint32_t numTilesPerCamera,
    const torch::Tensor &activeTiles,
    const at::optional<torch::Tensor> &backgrounds) {
    FVDB_FUNC_RANGE();
    const uint32_t channels = features.size(-1);
    const bool isPacked     = means2d.dim() == 2;

    const std::optional<torch::Tensor> masks = std::nullopt;

#define CALL_TILE_SPARSE_FWD_CUDA(N)                                                              \
    case N: {                                                                                     \
        if (isPacked) {                                                                           \
            return launchRasterizeForwardTileSparseKernel<float, N, true>(means2d,                \
                                                                         conics,                 \
                                                                         features,                \
                                                                         opacities,               \
                                                                         backgrounds,             \
                                                                         masks,                   \
                                                                         imageWidth,              \
                                                                         imageHeight,             \
                                                                         imageOriginW,            \
                                                                         imageOriginH,            \
                                                                         tileSize,                \
                                                                         tileOffsets,             \
                                                                         tileGaussianIds,         \
                                                                         tileToOutputIdx,        \
                                                                         numTilesPerCamera,       \
                                                                         activeTiles);           \
        } else {                                                                                  \
            return launchRasterizeForwardTileSparseKernel<float, N, false>(means2d,               \
                                                                          conics,                 \
                                                                          features,                \
                                                                          opacities,               \
                                                                          backgrounds,             \
                                                                          masks,                   \
                                                                          imageWidth,              \
                                                                          imageHeight,             \
                                                                          imageOriginW,            \
                                                                          imageOriginH,            \
                                                                          tileSize,                \
                                                                          tileOffsets,             \
                                                                          tileGaussianIds,        \
                                                                          tileToOutputIdx,        \
                                                                          numTilesPerCamera,       \
                                                                          activeTiles);            \
        }                                                                                         \
    }

    switch (channels) {
        CALL_TILE_SPARSE_FWD_CUDA(1)
        CALL_TILE_SPARSE_FWD_CUDA(2)
        CALL_TILE_SPARSE_FWD_CUDA(3)
        CALL_TILE_SPARSE_FWD_CUDA(4)
        CALL_TILE_SPARSE_FWD_CUDA(5)
        CALL_TILE_SPARSE_FWD_CUDA(8)
        CALL_TILE_SPARSE_FWD_CUDA(9)
        CALL_TILE_SPARSE_FWD_CUDA(16)
        CALL_TILE_SPARSE_FWD_CUDA(17)
        CALL_TILE_SPARSE_FWD_CUDA(32)
        CALL_TILE_SPARSE_FWD_CUDA(33)
        CALL_TILE_SPARSE_FWD_CUDA(64)
        CALL_TILE_SPARSE_FWD_CUDA(65)
        default:
            AT_ERROR("Unsupported number of channels: ", channels);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianTileSparseRasterizeForward<torch::kCPU>(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &features,
    const torch::Tensor &opacities,
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
    const torch::Tensor &tileToOutputIdx,
    const uint32_t numTilesPerCamera,
    const torch::Tensor &activeTiles,
    const at::optional<torch::Tensor> &backgrounds) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianTileSparseRasterizeForward<torch::kPrivateUse1>(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &features,
    const torch::Tensor &opacities,
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,
    const torch::Tensor &tileGaussianIds,
    const torch::Tensor &tileToOutputIdx,
    const uint32_t numTilesPerCamera,
    const torch::Tensor &activeTiles,
    const at::optional<torch::Tensor> &backgrounds) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "PrivateUse1 implementation not available");
}

} // namespace fvdb::detail::ops
