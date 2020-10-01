#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"
#include <stdio.h>

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

struct dataobj
{
    void *restrict data;
    int *size;
    int *npsize;
    int *dsize;
    int *hsize;
    int *hofs;
    int *oofs;
};

struct profiler
{
    double section0;
    double section1;
    double section2;
};

void bf0(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r73_vec, float *restrict r74_vec, float *restrict r75_vec, float *restrict r76_vec, float *restrict r77_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, const int x0_blk0_size, const int x_size, const int y0_blk0_size, const int y_size, const int z_size, const int t0, const int t1, const int t2, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int sp_zi_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, float **restrict r131_vec, float **restrict r132_vec, const int time, const int tw);

int ForwardTTI(struct dataobj *restrict block_sizes_vec, struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, const float dt, struct dataobj *restrict epsilon_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict phi_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict theta_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x_size, const int y_size, const int z_size, const int sp_zi_m, const int time_M, const int time_m, struct profiler *timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int nthreads_nonaffine)
{
    int(*restrict block_sizes) __attribute__((aligned(64))) = (int(*))block_sizes_vec->data;
    float(*restrict delta)[delta_vec->size[1]][delta_vec->size[2]] __attribute__((aligned(64))) = (float(*)[delta_vec->size[1]][delta_vec->size[2]])delta_vec->data;
    int(*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[nnz_sp_source_mask_vec->size[1]])nnz_sp_source_mask_vec->data;
    float(*restrict phi)[phi_vec->size[1]][phi_vec->size[2]] __attribute__((aligned(64))) = (float(*)[phi_vec->size[1]][phi_vec->size[2]])phi_vec->data;
    float(*restrict save_src_u)[save_src_u_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_u_vec->size[1]])save_src_u_vec->data;
    float(*restrict save_src_v)[save_src_v_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_v_vec->size[1]])save_src_v_vec->data;
    int(*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]][source_id_vec->size[2]])source_id_vec->data;
    int(*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_mask_vec->size[1]][source_mask_vec->size[2]])source_mask_vec->data;
    int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]])sp_source_mask_vec->data;
    float(*restrict theta)[theta_vec->size[1]][theta_vec->size[2]] __attribute__((aligned(64))) = (float(*)[theta_vec->size[1]][theta_vec->size[2]])theta_vec->data;
    float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
    float(*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]])v_vec->data;

    float(*r73)[y_size + 3 + 3][z_size + 3 + 3];
    posix_memalign((void **)&r73, 64, sizeof(float[x_size + 3 + 3][y_size + 3 + 3][z_size + 3 + 3]));
    float(*r74)[y_size + 3 + 3][z_size + 3 + 3];
    posix_memalign((void **)&r74, 64, sizeof(float[x_size + 3 + 3][y_size + 3 + 3][z_size + 3 + 3]));
    float(*r75)[y_size + 3 + 3][z_size + 3 + 3];
    posix_memalign((void **)&r75, 64, sizeof(float[x_size + 3 + 3][y_size + 3 + 3][z_size + 3 + 3]));
    float(*r76)[y_size + 3 + 3][z_size + 3 + 3];
    posix_memalign((void **)&r76, 64, sizeof(float[x_size + 3 + 3][y_size + 3 + 3][z_size + 3 + 3]));
    float(*r77)[y_size + 3 + 3][z_size + 3 + 3];
    posix_memalign((void **)&r77, 64, sizeof(float[x_size + 3 + 3][y_size + 3 + 3][z_size + 3 + 3]));
    float **r131;
    posix_memalign((void **)&r131, 64, sizeof(float *) * nthreads);
    float **r132;
    posix_memalign((void **)&r132, 64, sizeof(float *) * nthreads);

    int y0_blk0_size = block_sizes[3];
    int x0_blk0_size = block_sizes[2];
    int yb_size = block_sizes[1];
    int xb_size = block_sizes[0];
    int sf = 6;
    int t_blk_size = 2 * sf * (time_M - time_m);

    printf(" Tiles: %d, %d ::: Blocks %d, %d \n", xb_size, yb_size, x0_blk0_size, y0_blk0_size);

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        posix_memalign((void **)&r131[tid], 64, sizeof(float[x0_blk0_size + 3 + 3][y0_blk0_size + 3 + 3][z_size + 3 + 3]));
        posix_memalign((void **)&r132[tid], 64, sizeof(float[x0_blk0_size + 3 + 3][y0_blk0_size + 3 + 3][z_size + 3 + 3]));
    }

    /* Flush denormal numbers to zero in hardware */
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
/* Begin section0 */
#pragma omp parallel num_threads(nthreads)
    {
#pragma omp for collapse(2) schedule(static, 1)
        for (int x = x_m - 3; x <= x_M + 3; x += 1)
        {
            for (int y = y_m - 3; y <= y_M + 3; y += 1)
            {
#pragma omp simd aligned(delta, phi, theta : 64)
                for (int z = z_m - 3; z <= z_M + 3; z += 1)
                {
                    r73[x + 3][y + 3][z + 3] = sqrt(2 * delta[x + 12][y + 12][z + 12] + 1);
                    r74[x + 3][y + 3][z + 3] = cos(theta[x + 12][y + 12][z + 12]);
                    r75[x + 3][y + 3][z + 3] = sin(phi[x + 12][y + 12][z + 12]);
                    r76[x + 3][y + 3][z + 3] = sin(theta[x + 12][y + 12][z + 12]);
                    r77[x + 3][y + 3][z + 3] = cos(phi[x + 12][y + 12][z + 12]);
                }
            }
        }
    }
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec - start_section0.tv_sec) + (double)(end_section0.tv_usec - start_section0.tv_usec) / 1000000;

    printf(" Tiles: %d, %d ::: Blocks %d, %d \n", xb_size, yb_size, x0_blk0_size, y0_blk0_size);

    for (int t_blk = time_m; t_blk <= 1 + sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block
    {
        for (int xb = x_m - 1; xb <= (x_M + sf * (time_M - time_m)); xb += xb_size)
        {
            //printf(" Change of outer xblock %d \n", xb);
            for (int yb = y_m - 1; yb <= (y_M + sf * (time_M - time_m)); yb += yb_size)
            {
                for (int time = t_blk, t0 = (time) % (3), t1 = (time + 2) % (3), t2 = (time + 1) % (3); time <= 2 + min(t_blk + t_blk_size - 1, sf * (time_M - time_m)); time += sf, t0 = (((time / sf) % (time_M - time_m + 1))) % (3), t1 = (((time / sf) % (time_M - time_m + 1)) + 2) % (3), t2 = (((time / sf) % (time_M - time_m + 1)) + 1) % (3))
                {
                    int tw = ((time / sf) % (time_M - time_m + 1));

                    struct timeval start_section1, end_section1;
                    gettimeofday(&start_section1, NULL);
                    /* Begin section1 */
                    bf0(damp_vec, dt, epsilon_vec, (float *)r73, (float *)r74, (float *)r75, (float *)r76, (float *)r77, u_vec, v_vec, vp_vec, nnz_sp_source_mask_vec, sp_source_mask_vec, save_src_u_vec, save_src_v_vec, source_id_vec, source_mask_vec, x0_blk0_size, x_size, y0_blk0_size, y_size, z_size, t0, t1, t2, x_M, x_m, y_M, y_m, z_M, z_m, sp_zi_m, nthreads, xb, yb, xb_size, yb_size, (float **)r131, (float **)r132, time, tw);
                    // x_M - (x_M - x_m + 1)%(x0_blk0_size), x_m, y_M - (y_M - y_m + 1)%(y0_blk0_size), y_m,
                    /* End section1 */
                    gettimeofday(&end_section1, NULL);
                    timers->section1 += (double)(end_section1.tv_sec - start_section1.tv_sec) + (double)(end_section1.tv_usec - start_section1.tv_usec) / 1000000;
                }
            }
        }
    }

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        free(r131[tid]);
        free(r132[tid]);
    }
    free(r73);
    free(r74);
    free(r75);
    free(r76);
    free(r77);
    free(r131);
    free(r132);
    return 0;
}

void bf0(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r73_vec, float *restrict r74_vec, float *restrict r75_vec, float *restrict r76_vec, float *restrict r77_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, const int x0_blk0_size, const int x_size, const int y0_blk0_size, const int y_size, const int z_size, const int t0, const int t1, const int t2, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int sp_zi_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, float **restrict r131_vec, float **restrict r132_vec, const int time, const int tw)
{
    float(*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[damp_vec->size[1]][damp_vec->size[2]])damp_vec->data;
    float(*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__((aligned(64))) = (float(*)[epsilon_vec->size[1]][epsilon_vec->size[2]])epsilon_vec->data;
    float(*restrict r73)[y_size + 3 + 3][z_size + 3 + 3] __attribute__((aligned(64))) = (float(*)[y_size + 3 + 3][z_size + 3 + 3]) r73_vec;
    float(*restrict r74)[y_size + 3 + 3][z_size + 3 + 3] __attribute__((aligned(64))) = (float(*)[y_size + 3 + 3][z_size + 3 + 3]) r74_vec;
    float(*restrict r75)[y_size + 3 + 3][z_size + 3 + 3] __attribute__((aligned(64))) = (float(*)[y_size + 3 + 3][z_size + 3 + 3]) r75_vec;
    float(*restrict r76)[y_size + 3 + 3][z_size + 3 + 3] __attribute__((aligned(64))) = (float(*)[y_size + 3 + 3][z_size + 3 + 3]) r76_vec;
    float(*restrict r77)[y_size + 3 + 3][z_size + 3 + 3] __attribute__((aligned(64))) = (float(*)[y_size + 3 + 3][z_size + 3 + 3]) r77_vec;
    float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
    float(*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]])v_vec->data;
    float(*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[vp_vec->size[1]][vp_vec->size[2]])vp_vec->data;
    float **r131 = (float **)r131_vec;
    float **r132 = (float **)r132_vec;

    int(*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[nnz_sp_source_mask_vec->size[1]])nnz_sp_source_mask_vec->data;
    float(*restrict save_src_u)[save_src_u_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_u_vec->size[1]])save_src_u_vec->data;
    float(*restrict save_src_v)[save_src_v_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_v_vec->size[1]])save_src_v_vec->data;
    int(*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]][source_id_vec->size[2]])source_id_vec->data;
    int(*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_mask_vec->size[1]][source_mask_vec->size[2]])source_mask_vec->data;
    int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]])sp_source_mask_vec->data;

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        float(*restrict r118)[y0_blk0_size + 3 + 3][z_size + 3 + 3] __attribute__((aligned(64))) = (float(*)[y0_blk0_size + 3 + 3][z_size + 3 + 3]) r131[tid];
        float(*restrict r119)[y0_blk0_size + 3 + 3][z_size + 3 + 3] __attribute__((aligned(64))) = (float(*)[y0_blk0_size + 3 + 3][z_size + 3 + 3]) r132[tid];
#pragma omp for collapse(2) schedule(dynamic, 1)
        for (int x0_blk0 = max((x_m + time), xb); x0_blk0 <= min((x_M + time), (xb + xb_size)); x0_blk0 += x0_blk0_size)
        {
            for (int y0_blk0 = max((y_m + time), yb); y0_blk0 <= min((y_M + time), (yb + yb_size)); y0_blk0 += y0_blk0_size)
            {
                for (int x = x0_blk0 - 3, xs = 0; x <= min(min((x_M + time), (xb + xb_size + 2)), (x0_blk0 + x0_blk0_size + 2)); x++, xs++)
                {
                    for (int y = y0_blk0 - 3, ys = 0; y <= min(min((y_M + time), (yb + yb_size + 2)), (y0_blk0 + y0_blk0_size + 2)); y++, ys++)
                    {
#pragma omp simd aligned(u, v : 64)
                        for (int z = z_m - 3; z <= z_M + 3; z += 1)
                        {
                            r118[xs][ys][z + 3] = -(1.66666669e-3F * (-u[t0][x - time + 9][y - time + 12][z + 12] + u[t0][x - time + 15][y - time + 12][z + 12]) + 1.50000002e-2F * (u[t0][x - time + 10][y - time + 12][z + 12] - u[t0][x - time + 14][y - time + 12][z + 12]) + 7.50000011e-2F * (-u[t0][x - time + 11][y - time + 12][z + 12] + u[t0][x - time + 13][y - time + 12][z + 12])) * r76[x - time + 3][y - time + 3][z + 3] * r77[x - time + 3][y - time + 3][z + 3] - (1.66666669e-3F * (-u[t0][x - time + 12][y - time + 9][z + 12] + u[t0][x - time + 12][y - time + 15][z + 12]) + 1.50000002e-2F * (u[t0][x - time + 12][y - time + 10][z + 12] - u[t0][x - time + 12][y - time + 14][z + 12]) + 7.50000011e-2F * (-u[t0][x - time + 12][y - time + 11][z + 12] + u[t0][x - time + 12][y - time + 13][z + 12])) * r75[x - time + 3][y - time + 3][z + 3] * r76[x - time + 3][y - time + 3][z + 3] - (1.66666669e-3F * (-u[t0][x - time + 12][y - time + 12][z + 9] + u[t0][x - time + 12][y - time + 12][z + 15]) + 1.50000002e-2F * (u[t0][x - time + 12][y - time + 12][z + 10] - u[t0][x - time + 12][y - time + 12][z + 14]) + 7.50000011e-2F * (-u[t0][x - time + 12][y - time + 12][z + 11] + u[t0][x - time + 12][y - time + 12][z + 13])) * r74[x - time + 3][y - time + 3][z + 3];
                            r119[xs][ys][z + 3] = -(1.66666669e-3F * (-v[t0][x - time + 9][y - time + 12][z + 12] + v[t0][x - time + 15][y - time + 12][z + 12]) + 1.50000002e-2F * (v[t0][x - time + 10][y - time + 12][z + 12] - v[t0][x - time + 14][y - time + 12][z + 12]) + 7.50000011e-2F * (-v[t0][x - time + 11][y - time + 12][z + 12] + v[t0][x - time + 13][y - time + 12][z + 12])) * r76[x - time + 3][y - time + 3][z + 3] * r77[x - time + 3][y - time + 3][z + 3] - (1.66666669e-3F * (-v[t0][x - time + 12][y - time + 9][z + 12] + v[t0][x - time + 12][y - time + 15][z + 12]) + 1.50000002e-2F * (v[t0][x - time + 12][y - time + 10][z + 12] - v[t0][x - time + 12][y - time + 14][z + 12]) + 7.50000011e-2F * (-v[t0][x - time + 12][y - time + 11][z + 12] + v[t0][x - time + 12][y - time + 13][z + 12])) * r75[x - time + 3][y - time + 3][z + 3] * r76[x - time + 3][y - time + 3][z + 3] - (1.66666669e-3F * (-v[t0][x - time + 12][y - time + 12][z + 9] + v[t0][x - time + 12][y - time + 12][z + 15]) + 1.50000002e-2F * (v[t0][x - time + 12][y - time + 12][z + 10] - v[t0][x - time + 12][y - time + 12][z + 14]) + 7.50000011e-2F * (-v[t0][x - time + 12][y - time + 12][z + 11] + v[t0][x - time + 12][y - time + 12][z + 13])) * r74[x - time + 3][y - time + 3][z + 3];
                        }
                    }
                }
                for (int x = x0_blk0, xs = 0; x <= min(min((x_M + time), (xb + xb_size - 1)), (x0_blk0 + x0_blk0_size - 1)); x++, xs++)
                {
                    for (int y = y0_blk0, ys = 0; y <= min(min((y_M + time), (yb + yb_size - 1)), (y0_blk0 + y0_blk0_size - 1)); y++, ys++)
                    {
#pragma omp simd aligned(damp, epsilon, u, v, vp : 64)
                        for (int z = z_m; z <= z_M; z += 1)
                        {
                            float r130 = 1.0 / dt;
                            float r129 = 1.0 / (dt * dt);
                            float r128 = 1.50000002e-2F * (-r119[xs + 1][ys + 3][z + 3] * r76[x - time + 1][y - time + 3][z + 3] * r77[x - time + 1][y - time + 3][z + 3] - r119[xs + 3][ys + 1][z + 3] * r75[x - time + 3][y - time + 1][z + 3] * r76[x - time + 3][y - time + 1][z + 3] - r119[xs + 3][ys + 3][z + 1] * r74[x - time + 3][y - time + 3][z + 1] + r119[xs + 3][ys + 3][z + 5] * r74[x - time + 3][y - time + 3][z + 5] + r119[xs + 3][ys + 5][z + 3] * r75[x - time + 3][y - time + 5][z + 3] * r76[x - time + 3][y - time + 5][z + 3] + r119[xs + 5][ys + 3][z + 3] * r76[x - time + 5][y - time + 3][z + 3] * r77[x - time + 5][y - time + 3][z + 3]);
                            float r127 = 1.66666669e-3F * (r119[xs][ys + 3][z + 3] * r76[x - time][y - time + 3][z + 3] * r77[x - time][y - time + 3][z + 3] + r119[xs + 3][ys][z + 3] * r75[x - time + 3][y - time][z + 3] * r76[x - time + 3][y - time][z + 3] + r119[xs + 3][ys + 3][z] * r74[x - time + 3][y - time + 3][z] - r119[xs + 3][ys + 3][z + 6] * r74[x - time + 3][y - time + 3][z + 6] - r119[xs + 3][ys + 6][z + 3] * r75[x - time + 3][y - time + 6][z + 3] * r76[x - time + 3][y - time + 6][z + 3] - r119[xs + 6][ys + 3][z + 3] * r76[x - time + 6][y - time + 3][z + 3] * r77[x - time + 6][y - time + 3][z + 3]);
                            float r126 = 7.50000011e-2F * (r119[xs + 2][ys + 3][z + 3] * r76[x - time + 2][y - time + 3][z + 3] * r77[x - time + 2][y - time + 3][z + 3] + r119[xs + 3][ys + 2][z + 3] * r75[x - time + 3][y - time + 2][z + 3] * r76[x - time + 3][y - time + 2][z + 3] + r119[xs + 3][ys + 3][z + 2] * r74[x - time + 3][y - time + 3][z + 2] - r119[xs + 3][ys + 3][z + 4] * r74[x - time + 3][y - time + 3][z + 4] - r119[xs + 3][ys + 4][z + 3] * r75[x - time + 3][y - time + 4][z + 3] * r76[x - time + 3][y - time + 4][z + 3] - r119[xs + 4][ys + 3][z + 3] * r76[x - time + 4][y - time + 3][z + 3] * r77[x - time + 4][y - time + 3][z + 3]);
                            float r125 = 1.0 / (vp[x - time + 12][y - time + 12][z + 12] * vp[x - time + 12][y - time + 12][z + 12]);
                            float r124 = 1.0 / (r125 * r129 + r130 * damp[x - time + 1][y - time + 1][z + 1]);
                            float r123 = 1.66666669e-3F * (-r118[xs][ys + 3][z + 3] * r76[x - time][y - time + 3][z + 3] * r77[x - time][y - time + 3][z + 3] - r118[xs + 3][ys][z + 3] * r75[x - time + 3][y - time][z + 3] * r76[x - time + 3][y - time][z + 3] - r118[xs + 3][ys + 3][z] * r74[x - time + 3][y - time + 3][z] + r118[xs + 3][ys + 3][z + 6] * r74[x - time + 3][y - time + 3][z + 6] + r118[xs + 3][ys + 6][z + 3] * r75[x - time + 3][y - time + 6][z + 3] * r76[x - time + 3][y - time + 6][z + 3] + r118[xs + 6][ys + 3][z + 3] * r76[x - time + 6][y - time + 3][z + 3] * r77[x - time + 6][y - time + 3][z + 3]) + 1.50000002e-2F * (r118[xs + 1][ys + 3][z + 3] * r76[x - time + 1][y - time + 3][z + 3] * r77[x - time + 1][y - time + 3][z + 3] + r118[xs + 3][ys + 1][z + 3] * r75[x - time + 3][y - time + 1][z + 3] * r76[x - time + 3][y - time + 1][z + 3] + r118[xs + 3][ys + 3][z + 1] * r74[x - time + 3][y - time + 3][z + 1] - r118[xs + 3][ys + 3][z + 5] * r74[x - time + 3][y - time + 3][z + 5] - r118[xs + 3][ys + 5][z + 3] * r75[x - time + 3][y - time + 5][z + 3] * r76[x - time + 3][y - time + 5][z + 3] - r118[xs + 5][ys + 3][z + 3] * r76[x - time + 5][y - time + 3][z + 3] * r77[x - time + 5][y - time + 3][z + 3]) + 7.50000011e-2F * (-r118[xs + 2][ys + 3][z + 3] * r76[x - time + 2][y - time + 3][z + 3] * r77[x - time + 2][y - time + 3][z + 3] - r118[xs + 3][ys + 2][z + 3] * r75[x - time + 3][y - time + 2][z + 3] * r76[x - time + 3][y - time + 2][z + 3] - r118[xs + 3][ys + 3][z + 2] * r74[x - time + 3][y - time + 3][z + 2] + r118[xs + 3][ys + 3][z + 4] * r74[x - time + 3][y - time + 3][z + 4] + r118[xs + 3][ys + 4][z + 3] * r75[x - time + 3][y - time + 4][z + 3] * r76[x - time + 3][y - time + 4][z + 3] + r118[xs + 4][ys + 3][z + 3] * r76[x - time + 4][y - time + 3][z + 3] * r77[x - time + 4][y - time + 3][z + 3]) - 6.01250588e-7F * (u[t0][x - time + 6][y - time + 12][z + 12] + u[t0][x - time + 12][y - time + 6][z + 12] + u[t0][x - time + 12][y - time + 12][z + 6] + u[t0][x - time + 12][y - time + 12][z + 18] + u[t0][x - time + 12][y - time + 18][z + 12] + u[t0][x - time + 18][y - time + 12][z + 12]) + 1.03896102e-5F * (u[t0][x - time + 7][y - time + 12][z + 12] + u[t0][x - time + 12][y - time + 7][z + 12] + u[t0][x - time + 12][y - time + 12][z + 7] + u[t0][x - time + 12][y - time + 12][z + 17] + u[t0][x - time + 12][y - time + 17][z + 12] + u[t0][x - time + 17][y - time + 12][z + 12]) - 8.92857123e-5F * (u[t0][x - time + 8][y - time + 12][z + 12] + u[t0][x - time + 12][y - time + 8][z + 12] + u[t0][x - time + 12][y - time + 12][z + 8] + u[t0][x - time + 12][y - time + 12][z + 16] + u[t0][x - time + 12][y - time + 16][z + 12] + u[t0][x - time + 16][y - time + 12][z + 12]) + 5.29100517e-4F * (u[t0][x - time + 9][y - time + 12][z + 12] + u[t0][x - time + 12][y - time + 9][z + 12] + u[t0][x - time + 12][y - time + 12][z + 9] + u[t0][x - time + 12][y - time + 12][z + 15] + u[t0][x - time + 12][y - time + 15][z + 12] + u[t0][x - time + 15][y - time + 12][z + 12]) - 2.67857137e-3F * (u[t0][x - time + 10][y - time + 12][z + 12] + u[t0][x - time + 12][y - time + 10][z + 12] + u[t0][x - time + 12][y - time + 12][z + 10] + u[t0][x - time + 12][y - time + 12][z + 14] + u[t0][x - time + 12][y - time + 14][z + 12] + u[t0][x - time + 14][y - time + 12][z + 12]) + 1.71428568e-2F * (u[t0][x - time + 11][y - time + 12][z + 12] + u[t0][x - time + 12][y - time + 11][z + 12] + u[t0][x - time + 12][y - time + 12][z + 11] + u[t0][x - time + 12][y - time + 12][z + 13] + u[t0][x - time + 12][y - time + 13][z + 12] + u[t0][x - time + 13][y - time + 12][z + 12]) - 8.94833313e-2F * u[t0][x - time + 12][y - time + 12][z + 12];
                            float r116 = r129 * (-2.0F * u[t0][x - time + 12][y - time + 12][z + 12] + u[t1][x - time + 12][y - time + 12][z + 12]);
                            float r117 = r129 * (-2.0F * v[t0][x - time + 12][y - time + 12][z + 12] + v[t1][x - time + 12][y - time + 12][z + 12]);
                            u[t2][x - time + 12][y - time + 12][z + 12] = r124 * ((-r116) * r125 + r123 * (2 * epsilon[x - time + 12][y - time + 12][z + 12] + 1) + r130 * (damp[x - time + 1][y - time + 1][z + 1] * u[t0][x - time + 12][y - time + 12][z + 12]) + (r126 + r127 + r128) * r73[x - time + 3][y - time + 3][z + 3]);
                            v[t2][x - time + 12][y - time + 12][z + 12] = r124 * ((-r117) * r125 + r123 * r73[x - time + 3][y - time + 3][z + 3] + r126 + r127 + r128 + r130 * (damp[x - time + 1][y - time + 1][z + 1] * v[t0][x - time + 12][y - time + 12][z + 12]));
                        }
                        int sp_zi_M = nnz_sp_source_mask[x - time][y - time] - 1;
                        for (int sp_zi = sp_zi_m; sp_zi <= sp_zi_M; sp_zi += 1)
                        {
                            int zind = sp_source_mask[x - time][y - time][sp_zi];
                            float r22 = save_src_u[tw][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
                            u[t2][x - time + 12][y - time + 12][zind + 12] += r22;
                            float r23 = save_src_v[tw][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
                            v[t2][x - time + 12][y - time + 12][zind + 12] += r23;
                            //printf("Source injection at time %d , at : x: %d, y: %d, %d, %f, %f \n", tw, x - time + 4, y - time + 4, zind + 4, r22, r23);
                        }
                    }
                }
            }
        }
    }
}
