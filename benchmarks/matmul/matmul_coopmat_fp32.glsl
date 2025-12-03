#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : enable

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

// Cooperative matrix tiles. These must match the workgroup launch parameters.
const uint CM = TILE_M;
const uint CN = TILE_N;
const uint CK = TILE_K;
const uint SUBGROUP_SIZE = 16;
const uint SUBGROUPS_PER_WG = WG_X / SUBGROUP_SIZE;

// Flattened buffers make coopMatLoad/store pointer math straightforward.
shared float shA[TILE_M * TILE_K];
shared float shB[TILE_K * TILE_N * SUBGROUPS_PER_WG];
shared float shC[TILE_M * TILE_N * SUBGROUPS_PER_WG];

uint coordToOffset(uint i, uint j, uint stride) {
  return stride * i + j;
}

float loadScalar(vec4 v, uint lane) {
  if (lane == 0) return v.x;
  if (lane == 1) return v.y;
  if (lane == 2) return v.z;
  return v.w;
}

void main() {
  // Immortalis G720 reports subgroup size 16; bail out if runtime differs.
  if (gl_SubgroupSize != SUBGROUP_SIZE) {
    return;
  }
  if (SUBGROUPS_PER_WG != 1) {
    return;
  }

  const uint subgroup_id = gl_SubgroupID;
  const uint tileRow = gl_WorkGroupID.y * TILE_M;
  const uint totalTileN = TILE_N * SUBGROUPS_PER_WG;
  const uint tileCol = gl_WorkGroupID.x * totalTileN + subgroup_id * TILE_N;
  const uint wgLinearSize = WG_X * WG_Y;
  const uint linear = gl_LocalInvocationID.y * WG_X + gl_LocalInvocationID.x;

  coopmat<float, gl_ScopeSubgroup, CM, CN, gl_MatrixUseAccumulator> accum =
      coopmat<float, gl_ScopeSubgroup, CM, CN, gl_MatrixUseAccumulator>(0.0f);

  const uint strideA4 = strideA / 4;
  const uint strideB4 = strideB / 4;
  const uint strideC4 = strideC / 4;

  for (uint k0 = 0; k0 < K; k0 += TILE_K) {
    // Load A tile into shared (row-major).
    const uint numALoads = TILE_M * TILE_K;
    for (uint idx = linear; idx < numALoads; idx += wgLinearSize) {
      const uint row = idx / TILE_K;
      const uint col = idx - row * TILE_K;
      const uint vecIndex =
          coordToOffset(tileRow + row, (k0 + col) / 4, strideA4);
      const vec4 packed = inputA.x[vecIndex];
      shA[row * TILE_K + col] = loadScalar(packed, (k0 + col) % 4);
    }

    // Load B tile into shared (row-major).
    const uint numBLoads = TILE_K * totalTileN;
    for (uint idx = linear; idx < numBLoads; idx += wgLinearSize) {
      const uint row = idx / totalTileN;
      const uint col = idx - row * totalTileN;
      const uint vecIndex =
          coordToOffset(k0 + row, (tileCol + col) / 4, strideB4);
      const vec4 packed = inputB.x[vecIndex];
      shB[row * totalTileN + col] = loadScalar(packed, (tileCol + col) % 4);
    }

    barrier();

    coopmat<float, gl_ScopeSubgroup, CM, CK, gl_MatrixUseA> aTile;
    coopMatLoad(aTile, shA, 0, TILE_K, gl_CooperativeMatrixLayoutRowMajor);
    coopmat<float, gl_ScopeSubgroup, CK, CN, gl_MatrixUseB> bTile;
    coopMatLoad(bTile, shB, subgroup_id * TILE_N, totalTileN,
                gl_CooperativeMatrixLayoutRowMajor);
    accum = coopMatMulAdd(aTile, bTile, accum);

    barrier();
  }

  coopMatStore(accum, shC, subgroup_id * TILE_N, totalTileN,
               gl_CooperativeMatrixLayoutRowMajor);
  barrier();

  // Each invocation writes out vec4 results for the tile.
  const uint numStores = TILE_M * (TILE_N / 4);
  for (uint idx = linear; idx < numStores; idx += wgLinearSize) {
    const uint row = idx / (TILE_N / 4);
    const uint col4 = idx - row * (TILE_N / 4);
    const uint globalRow = tileRow + row;
    const uint globalCol4 = (tileCol / 4) + col4;
    outputO.x[coordToOffset(globalRow, globalCol4, strideC4)] =
        vec4(shC[subgroup_id * TILE_N + row * totalTileN + 4 * col4 + 0],
             shC[subgroup_id * TILE_N + row * totalTileN + 4 * col4 + 1],
             shC[subgroup_id * TILE_N + row * totalTileN + 4 * col4 + 2],
             shC[subgroup_id * TILE_N + row * totalTileN + 4 * col4 + 3]);
  }
}
