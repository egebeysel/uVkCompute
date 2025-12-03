#version 450 core
#pragma use_vulkan_memory_model
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : enable

// Direct cooperative matrix loads from global memory (no shared staging).
// Assumes subgroup size 16 on Mali/Immortalis.

layout(binding = 0) buffer InputA { float x[]; } inputA;
layout(binding = 1) buffer InputB { float x[]; } inputB;
layout(binding = 2) buffer Output { float x[]; } outputO;

layout(local_size_x = WG_X, local_size_y = WG_Y, local_size_z = 1) in;

layout(constant_id = 0) const uint M = 1;
layout(constant_id = 1) const uint N = 1;
layout(constant_id = 2) const uint K = 1;

const uint strideA = K;
const uint strideB = N;
const uint strideC = N;

// Parameterize this to 16 and 4.
const uint lM = (TILE_K / 16) * 12 + 4;
const uint lN = lM;
const uint lK = lM;

const uint C_ROWS = TILE_M / lM;
const uint C_COLS = TILE_N / lN;

uint coordToOffset(uint i, uint j, uint stride) { return stride * i + j; }

void main()
{
    uvec2 tileID = uvec2(gl_WorkGroupID.xy);

    // Initialize result to zero
    coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            result[i][j] = coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0.0);
        }
    }

    // On each iteration, load a row of cooperative matrices from matrix A,
    // load a column of cooperative matrices from matrix B, and multiply all
    // pairs of those matrices.
    for (uint chunkK = 0; chunkK < K; chunkK += lK) {
        coopmat<float, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[C_ROWS];
        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            uint gi = TILE_M * tileID.y + lM * i;
            uint gk = chunkK;
            coopMatLoad(matA[i], inputA.x, coordToOffset(gi, gk, strideA), strideA, gl_CooperativeMatrixLayoutRowMajor);
        }
        coopmat<float, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gj = TILE_N * tileID.x + lN * j;
            uint gk = chunkK;
            coopMatLoad(matB, inputB.x, coordToOffset(gk, gj, strideB), strideB, gl_CooperativeMatrixLayoutRowMajor);
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
            }
        }
    }

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = TILE_M * tileID.y + lM * i;
            uint gj = TILE_N * tileID.x + lN * j;
            coopMatStore(result[i][j], outputO.x, coordToOffset(gi, gj, strideC), strideC, gl_CooperativeMatrixLayoutRowMajor);
        }
    }
}