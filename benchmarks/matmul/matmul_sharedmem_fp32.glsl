#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable

layout(binding = 0) buffer InputA { vec4 x[]; } inputA;
layout(binding = 1) buffer InputB { vec4 x[]; } inputB;
layout(binding = 2) buffer Output { vec4 x[]; } outputO;

layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

const uint C_ROWS = TILE_M / WG_Y;
const uint C_COLS = TILE_N / (4 * WG_X);

// Add padding to mitigate bank conflicts when indexing along the packed (vec4)
// dimension. The +1 stride follows the approach used in ggml's matmul kernel.
const uint SHMEM_STRIDE_A = TILE_K / 4 + 1;
const uint SHMEM_STRIDE_B = TILE_N / 4 + 1;

shared vec4 shmemA[TILE_M * SHMEM_STRIDE_A];
shared vec4 shmemB[TILE_K * SHMEM_STRIDE_B];

uint coordToOffset(uint i, uint j, uint stride) {
  return stride * i + j;
}

void main() {
  uvec2 tileID = gl_WorkGroupID.xy;
  uvec2 lid = gl_LocalInvocationID.xy;

  const uint tileRow = tileID.y * TILE_M;
  const uint tileCol = tileID.x * TILE_N;

  vec4 accum[C_ROWS][C_COLS];
  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      accum[i][j] = vec4(0.0f);
    }
  }

  const uint wgLinearSize = WG_X * WG_Y;
  const uint linear = lid.y * WG_X + lid.x;
  const uint strideA4 = strideA / 4;
  const uint strideB4 = strideB / 4;
  const uint strideC4 = strideC / 4;

  for (uint k0 = 0; k0 < K; k0 += TILE_K) {
    // Load the current tiles of A and B into shared memory.
    const uint numALoads = TILE_M * (TILE_K / 4);
    for (uint idx = linear; idx < numALoads; idx += wgLinearSize) {
      const uint row = idx / (TILE_K / 4);
      const uint col4 = idx - row * (TILE_K / 4);
      const uint globalRow = tileRow + row;
      const uint globalCol4 = (k0 / 4) + col4;
      shmemA[row * SHMEM_STRIDE_A + col4] =
          inputA.x[coordToOffset(globalRow, globalCol4, strideA4)];
    }

    const uint numBLoads = TILE_K * (TILE_N / 4);
    for (uint idx = linear; idx < numBLoads; idx += wgLinearSize) {
      const uint kRow = idx / (TILE_N / 4);
      const uint col4 = idx - kRow * (TILE_N / 4);
      const uint globalK = k0 + kRow;
      const uint globalCol4 = (tileCol / 4) + col4;
      shmemB[kRow * SHMEM_STRIDE_B + col4] =
          inputB.x[coordToOffset(globalK, globalCol4, strideB4)];
    }

    barrier();

    // Compute the partial products for this K tile.
    [[unroll]] for (uint kk = 0; kk < TILE_K / 4; ++kk) {
      [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        const uint row = lid.y + i * WG_Y;
        const vec4 aVec = shmemA[row * SHMEM_STRIDE_A + kk];
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
          const uint col4 = lid.x + j * WG_X;
          const uint baseB = (kk * 4) * SHMEM_STRIDE_B + col4;
          const vec4 b0 = shmemB[baseB + 0 * SHMEM_STRIDE_B];
          const vec4 b1 = shmemB[baseB + 1 * SHMEM_STRIDE_B];
          const vec4 b2 = shmemB[baseB + 2 * SHMEM_STRIDE_B];
          const vec4 b3 = shmemB[baseB + 3 * SHMEM_STRIDE_B];
          accum[i][j] += aVec.x * b0;
          accum[i][j] += aVec.y * b1;
          accum[i][j] += aVec.z * b2;
          accum[i][j] += aVec.w * b3;
        }
      }
    }

    barrier();
  }

  [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
    const uint row = lid.y + i * WG_Y;
    const uint globalRow = tileRow + row;
    [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
      const uint col4 = lid.x + j * WG_X;
      const uint globalCol4 = (tileCol / 4) + col4;
      outputO.x[coordToOffset(globalRow, globalCol4, strideC4)] = accum[i][j];
    }
  }
}
