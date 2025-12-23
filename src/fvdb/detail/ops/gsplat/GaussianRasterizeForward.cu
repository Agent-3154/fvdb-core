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
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
struct RasterizeForwardArgs {
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


template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
struct RasterizeTileSparseForwardArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, NUM_CHANNELS, IS_PACKED>;
    CommonArgs commonArgs;

    typename CommonArgs::template TorchRAcc64<ScalarType, 5> mOutFeatures; // [C, T, tile_size, tile_size, NUM_CHANNELS]
    typename CommonArgs::template TorchRAcc64<ScalarType, 5> mOutAlphas;   // [C, T, tile_size, tile_size, 1]
    typename CommonArgs::template TorchRAcc64<int32_t, 4> mOutLastIds;    // [C, T, tile_size, tile_size]
    typename CommonArgs::template TorchRAcc64<int32_t, 1> mTileOffsetsSparse; // [C * T + 1] sparse cumulative offsets
    typename CommonArgs::template TorchRAcc64<uint32_t, 1> mActiveTiles;      // [C * T] dense tile indices
    uint32_t mNumTilesPerCamera;

    RasterizeTileSparseForwardArgs(
        const torch::Tensor &means2d,         // [C, N, 2]
        const torch::Tensor &conics,          // [C, N, 3]
        const torch::Tensor &opacities,       // [C, N]
        const torch::Tensor &features,        // [C, N, NUM_CHANNELS]
        const std::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const std::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const uint32_t imageWidth,
        const uint32_t imageHeight,
        const uint32_t imageOriginW,
        const uint32_t imageOriginH,
        const uint32_t tileSize,
        const torch::Tensor &tileOffsetsSparse, // [C * T + 1] sparse cumulative offsets
        const torch::Tensor &tileGaussianIds,   // [totalIntersections]
        const uint32_t numTilesPerCamera,
        const torch::Tensor &activeTiles,      // [C * T] dense tile indices
        const torch::Tensor &outFeatures,      // [C, T, tile_size, tile_size, NUM_CHANNELS]
        const torch::Tensor &outAlphas,        // [C, T, tile_size, tile_size, 1]
        const torch::Tensor &outLastIds)       // [C, T, tile_size, tile_size]
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
                     0, // blockOffset
                     // Create a dummy dense tileOffsets for commonArgs (it requires [C, H, W] format)
                     torch::zeros({means2d.size(0), 
                                   static_cast<int64_t>((imageHeight + tileSize - 1) / tileSize),
                                   static_cast<int64_t>((imageWidth + tileSize - 1) / tileSize)},
                                  tileOffsetsSparse.options().dtype(torch::kInt32)), // dummy dense tileOffsets
                     tileGaussianIds,
                     std::nullopt, // we do not pass in activeTiles here
                     std::nullopt, // tilePixelMask
                     std::nullopt, // tilePixelCumsum
                     std::nullopt), // pixelMap
          mOutFeatures(initAccessor<ScalarType, 5>(outFeatures, "outFeatures")),
          mOutAlphas(initAccessor<ScalarType, 5>(outAlphas, "outAlphas")),
          mOutLastIds(initAccessor<int32_t, 4>(outLastIds, "outLastIds")),
          mTileOffsetsSparse(initAccessor<int32_t, 1>(tileOffsetsSparse, "tileOffsetsSparse")),
          mActiveTiles(initAccessor<uint32_t, 1>(activeTiles, "activeTiles")),
          mNumTilesPerCamera(numTilesPerCamera) {}

    /// @brief Get the Gaussian ID range for a tile using sparse offsets
    /// @param tileIndex Index into activeTiles array (0 to C*T-1)
    /// @return Pair of (firstGaussianId, lastGaussianId) for the tile
    __device__ std::pair<uint32_t, uint32_t>
    tileGaussianRangeSparse(uint32_t tileIndex) const {
        const uint32_t firstGaussianId = mTileOffsetsSparse[tileIndex];
        const uint32_t lastGaussianId  = mTileOffsetsSparse[tileIndex + 1];
        return {firstGaussianId, lastGaussianId};
    }

    /// @brief Write the alpha value for a pixel in a tile
    /// @param cameraId The camera ID
    /// @param tileId The tile ID within the camera
    /// @param row The row within the tile
    /// @param col The column within the tile
    /// @param alpha The alpha value to write
    __device__ void
    writeAlpha(uint32_t cameraId, uint32_t tileId, uint32_t row, uint32_t col, ScalarType alpha) {
        mOutAlphas[cameraId][tileId][row][col][0] = alpha;
    }

    /// @brief Write the last ID for a pixel in a tile
    /// @param cameraId The camera ID
    /// @param tileId The tile ID within the camera
    /// @param row The row within the tile
    /// @param col The column within the tile
    /// @param lastId The last ID to write
    __device__ void
    writeLastId(uint32_t cameraId, uint32_t tileId, uint32_t row, uint32_t col, int32_t lastId) {
        mOutLastIds[cameraId][tileId][row][col] = lastId;
    }

    /// @brief Write the features for a pixel in a tile
    /// @param cameraId The camera ID
    /// @param tileId The tile ID within the camera
    /// @param row The row within the tile
    /// @param col The column within the tile
    /// @param f The function to write the features
    template <typename F>
    __device__ void
    writeFeatures(uint32_t cameraId, uint32_t tileId, uint32_t row, uint32_t col, F &&f) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
            mOutFeatures[cameraId][tileId][row][col][k] = f(k);
        }
    }

    /// @brief Decode activeTiles entry to get camera and tile coordinates
    /// @param tileIndex Index into activeTiles array (0 to C*T-1)
    /// @return Tuple of (cameraId, tileY, tileX, tileId)
    __device__ std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>
    decodeActiveTile(uint32_t tileIndex) const {
        const uint32_t denseTileIndex = mActiveTiles[tileIndex];
        const uint32_t numTilesPerImage = commonArgs.mNumTilesH * commonArgs.mNumTilesW;
        
        // Decode: activeTiles[i] = cameraId * (numTilesH * numTilesW) + tileY * numTilesW + tileX
        const uint32_t cameraId = denseTileIndex / numTilesPerImage;
        const uint32_t tileFlat = denseTileIndex % numTilesPerImage;
        const uint32_t tileY = tileFlat / commonArgs.mNumTilesW;
        const uint32_t tileX = tileFlat % commonArgs.mNumTilesW;
        
        // tileId is the index within the camera's tile list (0 to T-1)
        const uint32_t tileId = tileIndex % mNumTilesPerCamera;
        
        return {cameraId, tileY, tileX, tileId};
    }

    /// @brief Volume render a tile of Gaussians
    /// @param cameraId The camera ID
    /// @param tileId The tile ID within the camera (0 to T-1)
    /// @param tileIndex Index into activeTiles array (cameraId * T + tileId)
    /// @param row The row within the tile (0 to tileSize-1)
    /// @param col The column within the tile (0 to tileSize-1)
    /// @param firstGaussianIdInBlock The first Gaussian ID in the block
    /// @param lastGaussianIdInBlock The last Gaussian ID in the block
    /// @param blockSize The size of the block
    __device__ void
    volumeRenderTileForward(const uint32_t cameraId,
                            const uint32_t tileId,
                            const uint32_t tileIndex,
                            const uint32_t row,
                            const uint32_t col,
                            const uint32_t firstGaussianIdInBlock,
                            const uint32_t lastGaussianIdInBlock,
                            const uint32_t blockSize) {
        alignas(Gaussian2D<ScalarType>) extern __shared__ char s[];
        auto *sharedGaussians = reinterpret_cast<Gaussian2D<ScalarType> *>(s); // [blockSize]

        // Decode tile coordinates from activeTiles
        // Bounds check: ensure tileIndex is valid for activeTiles
        const uint32_t maxActiveTileIndex = commonArgs.mNumCameras * mNumTilesPerCamera - 1;
        if (tileIndex > maxActiveTileIndex) {
            // Invalid tile index, write default values and return
            writeAlpha(cameraId, tileId, row, col, 0.0f);
            writeLastId(cameraId, tileId, row, col, -1);
            writeFeatures(cameraId, tileId, row, col, [&](uint32_t k) {
                return commonArgs.mHasBackgrounds ? commonArgs.mBackgrounds[cameraId][k] : 0.0f;
            });
            return;
        }
        
        const uint32_t denseTileIndex = mActiveTiles[tileIndex];
        const uint32_t numTilesPerImage = commonArgs.mNumTilesH * commonArgs.mNumTilesW;
        const uint32_t tileFlat = denseTileIndex % numTilesPerImage;
        const uint32_t tileY = tileFlat / commonArgs.mNumTilesW;
        const uint32_t tileX = tileFlat % commonArgs.mNumTilesW;

        // NOTE: The accumulated transmittance is used in the backward pass, and
        // since it's a sum of many small numbers, we should really use double precision.
        // However, this makes the backward pass 1.5x slower, so we stick with
        // float for now and sort of just ignore small impact gaussians
        // ¯\_(ツ)_/¯.
        ScalarType accumTransmittance = 1.0f;
        // index of most recent gaussian to write to this thread's pixel
        int32_t curIdx = -1;

        // Process Gaussians in batches of block size (i.e. one Gaussian per thread in the block)
        const uint32_t tidx = threadIdx.y * blockDim.x + threadIdx.x;
        const uint32_t numBatches =
            (lastGaussianIdInBlock - firstGaussianIdInBlock + blockSize - 1) / blockSize;

        // Compute absolute pixel coordinates
        // tileY and tileX are tile coordinates, row and col are within-tile coordinates
        const uint32_t absRow = tileY * commonArgs.mTileSize + row;
        const uint32_t absCol = tileX * commonArgs.mTileSize + col;
        
        // (row, col) coordinates are relative to the specified image origin which may
        // be a crop so we need to add the origin to get the absolute pixel coordinates
        const ScalarType px = absCol + commonArgs.mImageOriginW + ScalarType{0.5f};
        const ScalarType py = absRow + commonArgs.mImageOriginH + ScalarType{0.5f};

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        ScalarType pixOut[NUM_CHANNELS] = {0.f};
        bool done = false;
        
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
            } else {
                // Initialize with invalid gaussian to avoid reading uninitialized memory
                // This ensures all threads participate in the sync properly
                sharedGaussians[tidx] = Gaussian2D<ScalarType>();
            }

            // Sync threads so all gaussians for this batch are loaded in shared
            // memory
            __syncthreads();

            // Volume render Gaussians in this batch
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

        // Write outputs
        writeAlpha(cameraId, tileId, row, col, 1.0f - accumTransmittance);
        writeLastId(cameraId, tileId, row, col, curIdx);
        writeFeatures(cameraId, tileId, row, col, [&](uint32_t k) {
            return commonArgs.mHasBackgrounds
                       ? pixOut[k] + accumTransmittance * commonArgs.mBackgrounds[cameraId][k]
                       : pixOut[k];
        });
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

/// @brief Rasterize Gaussians to pixels for tile-sparse rendering
/// @param args The arguments for the rasterization
template <typename ScalarType, uint32_t NUM_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansTileSparseForward(RasterizeTileSparseForwardArgs<ScalarType, NUM_CHANNELS, IS_PACKED> args) {
    auto &commonArgs = args.commonArgs;

    // Each thread block processes one tile
    // blockIdx.x is the camera index (0 to C-1)
    // blockIdx.y is the tile index within the camera (0 to T-1)
    const uint32_t cameraId = blockIdx.x;
    const uint32_t tileId = blockIdx.y;
    
    // Compute tileIndex for accessing sparse offsets: tileIndex = cameraId * T + tileId
    const uint32_t tileIndex = cameraId * args.mNumTilesPerCamera + tileId;
    
    // Bounds check: ensure tileIndex is valid
    // tileOffsets has shape [C * T + 1], so valid tileIndex is 0 to C*T-1
    const uint32_t maxTileIndex = args.mNumTilesPerCamera * commonArgs.mNumCameras - 1;
    if (tileIndex > maxTileIndex) {
        return; // Invalid tile, exit early
    }
    
    // Each thread processes one pixel in the tile
    // threadIdx.y and threadIdx.x are the row and column within the tile
    const uint32_t row = threadIdx.y;
    const uint32_t col = threadIdx.x;
    
    // Bounds check for pixel coordinates
    if (row >= commonArgs.mTileSize || col >= commonArgs.mTileSize) {
        return; // Invalid pixel coordinates
    }

    // Get the Gaussian range for this tile using sparse offsets
    const auto [firstGaussianId, lastGaussianId] = args.tileGaussianRangeSparse(tileIndex);
    
    // Check if this tile has any Gaussians
    if (firstGaussianId >= lastGaussianId) {
        // No Gaussians for this tile, write background/default values
        args.writeAlpha(cameraId, tileId, row, col, 0.0f);
        args.writeLastId(cameraId, tileId, row, col, -1);
        args.writeFeatures(cameraId, tileId, row, col, [&](uint32_t k) {
            return commonArgs.mHasBackgrounds ? commonArgs.mBackgrounds[cameraId][k] : 0.0f;
        });
        return;
    }

    // Volume render this pixel
    const uint32_t blockSize = blockDim.x * blockDim.y; // tileSize * tileSize
    args.volumeRenderTileForward(cameraId,
                                 tileId,
                                 tileIndex,
                                 row,
                                 col,
                                 firstGaussianId,
                                 lastGaussianId,
                                 blockSize);
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

template <typename ScalarType, uint32_t NUM_CHANNELS>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launchRasterizeTileSparseForwardKernel(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2]
    const torch::Tensor &conics,                    // [C, N, 3]
    const torch::Tensor &features,                  // [C, N, channels]
    const torch::Tensor &opacities,                 // [C, N]
    const at::optional<torch::Tensor> &backgrounds, // [C, channels]
    const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
    // image size
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const uint32_t imageOriginW,
    const uint32_t imageOriginH,
    const uint32_t tileSize,
    // intersections
    const torch::Tensor &tileOffsets,     // [C * T + 1]
    const torch::Tensor &tileGaussianIds, // [n_isects]
    const uint32_t numTilesPerCamera,
    const torch::Tensor &activeTiles      // [C * T]
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));
    
    // std::cout<<tileOffsets.sizes()<<std::endl;
    // std::cout<<activeTiles.sizes()<<std::endl;
    
    const uint32_t C        = means2d.size(0);     // number of cameras
    const uint32_t N        = means2d.size(1);     // number of gaussians
    const uint32_t channels = features.size(-1);

    auto outFeatures = torch::zeros({C, numTilesPerCamera, tileSize, tileSize, channels}, means2d.options().dtype(torch::kFloat32));
    auto outAlphas   = torch::zeros({C, numTilesPerCamera, tileSize, tileSize, 1}, means2d.options().dtype(torch::kFloat32));
    auto outLastIds  = torch::zeros({C, numTilesPerCamera, tileSize, tileSize}, means2d.options().dtype(torch::kInt32));

    // std::cout<<"creating args"<<std::endl;
    // Convert activeTiles to int32_t if needed (commonArgs expects int32_t)
    // torch::Tensor activeTilesInt32 = activeTiles.dtype() == torch::kInt32 
    //     ? activeTiles 
    //     : activeTiles.to(torch::kInt32);
    
    // Create the args object with sparse tile offsets
    auto args = RasterizeTileSparseForwardArgs<ScalarType, NUM_CHANNELS, false>(
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
        tileOffsets,        // [C * T + 1] sparse offsets
        tileGaussianIds,
        numTilesPerCamera,
        activeTiles,   // [C * T], UInt32
        outFeatures,
        outAlphas,
        outLastIds);

    // Launch the CUDA kernel
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    // Thread blocks cooperatively cache a tile of Gaussians in shared memory
    const uint32_t sharedMem = getSharedMemRequirements<ScalarType>(tileSize);
    
    // Set shared memory attribute
    if (cudaFuncSetAttribute(rasterizeGaussiansTileSparseForward<ScalarType, NUM_CHANNELS, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             sharedMem) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMem,
                 " bytes), try lowering tile_size.");
    }
    
    // Grid: [C, T, 1] - one block per (camera, tile) pair
    // Block: tileSize x tileSize threads (one thread per pixel in tile)
    const dim3 gridDim(C, numTilesPerCamera, 1);
    const dim3 blockDim(tileSize, tileSize, 1);
    
    rasterizeGaussiansTileSparseForward<<<gridDim, blockDim, sharedMem, stream>>>(args);
    
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
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
    const uint32_t numTilesPerCamera,
    const torch::Tensor &activeTiles,
    const at::optional<torch::Tensor> &backgrounds
){
    FVDB_FUNC_RANGE();
    const uint32_t channels = features.size(-1);

    const std::optional<torch::Tensor> masks = std::nullopt;
    // std::cout << "tileOffsets: " << tileOffsets.sizes() << std::endl;
    // std::cout << "tileGaussianIds: " << tileGaussianIds.sizes() << std::endl;
    // std::cout << "numTilesPerCamera: " << numTilesPerCamera << std::endl;
    // std::cout << "activeTiles: " << activeTiles.sizes() << std::endl;

#define CALL_FWD_TILE_CUDA(N)                                                                        \
    case N: {                                                                                   \
        auto [outFeatures, outAlphas, outLastIds] =                                             \
            launchRasterizeTileSparseForwardKernel<float, N>(means2d,                          \
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
                                                              tileGaussianIds,                  \
                                                              numTilesPerCamera,                  \
                                                              activeTiles);               \
        return std::make_tuple(outFeatures, outAlphas, outLastIds);    \
    }

    // Make channels a compile time constant and do everything in register space
    // but at the expense of making this code ugly. NOTE: We do powers of two and
    // powers of two plus one to handle rendering common feature channel
    // dimensions with an optional additional depth channel
    switch (channels) {
        CALL_FWD_TILE_CUDA(1)
        CALL_FWD_TILE_CUDA(2)
        CALL_FWD_TILE_CUDA(3)
        CALL_FWD_TILE_CUDA(4)
        CALL_FWD_TILE_CUDA(5)
        CALL_FWD_TILE_CUDA(8)
        CALL_FWD_TILE_CUDA(9)
        CALL_FWD_TILE_CUDA(16)
        CALL_FWD_TILE_CUDA(17)
        CALL_FWD_TILE_CUDA(32)
        CALL_FWD_TILE_CUDA(33)
        CALL_FWD_TILE_CUDA(64)
        CALL_FWD_TILE_CUDA(65)
        CALL_FWD_TILE_CUDA(128)
        CALL_FWD_TILE_CUDA(129)
        CALL_FWD_TILE_CUDA(192)
        CALL_FWD_TILE_CUDA(193)
        CALL_FWD_TILE_CUDA(256)
        CALL_FWD_TILE_CUDA(257)
        CALL_FWD_TILE_CUDA(512)
        CALL_FWD_TILE_CUDA(513)
    default: AT_ERROR("Unsupported number of channels: ", channels);
    }
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
    const uint32_t numTilesPerCamera,
    const torch::Tensor &activeTiles,
    const at::optional<torch::Tensor> &backgrounds
){
    TORCH_CHECK_NOT_IMPLEMENTED(false, "PrivateUse1 implementation not available");
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
    const uint32_t numTilesPerCamera,
    const torch::Tensor &activeTiles,
    const at::optional<torch::Tensor> &backgrounds
)
{
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}
} // namespace fvdb::detail::ops
