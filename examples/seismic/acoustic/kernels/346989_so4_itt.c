#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "ittnotify.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include <stdio.h>
#include "omp.h"
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

int Kernel(struct dataobj *restrict block_sizes_vec, struct dataobj *restrict damp_vec, const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict usol_vec, struct dataobj *restrict vp_vec, const int sp_zi_m, const int time_M, const int time_m, struct profiler *timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads)
{
  int (*restrict block_sizes) __attribute__ ((aligned (64))) = (int (*)) block_sizes_vec->data;
  float(*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[damp_vec->size[1]][damp_vec->size[2]])damp_vec->data;
  int(*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[nnz_sp_source_mask_vec->size[1]])nnz_sp_source_mask_vec->data;
  float(*restrict save_src)[save_src_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_vec->size[1]])save_src_vec->data;
  int(*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]][source_id_vec->size[2]])source_id_vec->data;
  float(*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__((aligned(64))) = (float(*)[source_mask_vec->size[1]][source_mask_vec->size[2]])source_mask_vec->data;
  int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]])sp_source_mask_vec->data;
  float(*restrict usol)[usol_vec->size[1]][usol_vec->size[2]][usol_vec->size[3]] __attribute__((aligned(64))) = (float(*)[usol_vec->size[1]][usol_vec->size[2]][usol_vec->size[3]])usol_vec->data;
  float(*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[vp_vec->size[1]][vp_vec->size[2]])vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  __itt_resume();
  int xb_size = block_sizes[0];
  int y0_blk0_size = block_sizes[3];
  int x0_blk0_size = block_sizes[2];
  int yb_size = block_sizes[1];
  printf(" Tiles: %d, %d ::: Blocks %d, %d \n", xb_size, yb_size, x0_blk0_size, y0_blk0_size);

  int sf = 2;
  int t_blk_size = 2 * sf * (time_M - time_m);

  for (int t_blk = time_m; t_blk < sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block
  {
    for (int xb = x_m; xb <= (x_M + sf * (time_M - time_m)); xb += xb_size + 1)
    {
      for (int yb = y_m; yb <= (y_M + sf * (time_M - time_m)); yb += yb_size + 1)
      {
        for (int time = t_blk, t0 = (time + 1) % (3), t1 = (time) % (3), t2 = (time + 2) % (3); time <= 1 + min(t_blk + t_blk_size - 1, sf * (time_M - time_m)); time += sf, t0 = (((time / sf) % (time_M - time_m + 1)) + 1) % (3), t1 = (((time / sf) % (time_M - time_m + 1))) % (3), t2 = (((time / sf) % (time_M - time_m + 1)) + 2) % (3))
        {
          int tw = ((time / sf) % (time_M - time_m + 1));
#pragma omp parallel num_threads(nthreads)
          {
#pragma omp for collapse(2) schedule(dynamic, 1)
            for (int x0_blk0 = max((x_m + time), xb); x0_blk0 <= min((x_M + time), (xb + xb_size)); x0_blk0 += x0_blk0_size)
            {
              for (int y0_blk0 = max((y_m + time), yb); y0_blk0 <= min((y_M + time), (yb + yb_size)); y0_blk0 += y0_blk0_size)
              {
                for (int x = x0_blk0; x <= min(min((x_M + time), (xb + xb_size)), (x0_blk0 + x0_blk0_size - 1)); x++)
                {
                  for (int y = y0_blk0; y <= min(min((y_M + time), (yb + yb_size)), (y0_blk0 + y0_blk0_size - 1)); y++)
                  {
#pragma omp simd aligned(damp, usol, vp : 32)
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      float r14 = -2.5F * usol[t1][x - time + 4][y - time + 4][z + 4];
                      float r13 = 1.0 / dt;
                      float r12 = 1.0 / (dt * dt);
                      float r11 = 1.0 / (vp[x - time + 4][y - time + 4][z + 4] * vp[x - time + 4][y - time + 4][z + 4]);
                      usol[t0][x - time + 4][y - time + 4][z + 4] = (r11 * (-r12 * (-2.0F * usol[t1][x - time + 4][y - time + 4][z + 4] + usol[t2][x - time + 4][y - time + 4][z + 4])) + r13 * (damp[x - time + 1][y - time + 1][z + 1] * usol[t1][x - time + 4][y - time + 4][z + 4]) + (r14 - 8.33333333e-2F * (usol[t1][x - time + 4][y - time + 4][z + 2] + usol[t1][x - time + 4][y - time + 4][z + 6]) + 1.33333333F * (usol[t1][x - time + 4][y - time + 4][z + 3] + usol[t1][x - time + 4][y - time + 4][z + 5])) / ((h_z * h_z)) + (r14 - 8.33333333e-2F * (usol[t1][x - time + 4][y - time + 2][z + 4] + usol[t1][x - time + 4][y - time + 6][z + 4]) + 1.33333333F * (usol[t1][x - time + 4][y - time + 3][z + 4] + usol[t1][x - time + 4][y - time + 5][z + 4])) / ((h_y * h_y)) + (r14 - 8.33333333e-2F * (usol[t1][x - time + 2][y - time + 4][z + 4] + usol[t1][x - time + 6][y - time + 4][z + 4]) + 1.33333333F * (usol[t1][x - time + 3][y - time + 4][z + 4] + usol[t1][x - time + 5][y - time + 4][z + 4])) / ((h_x * h_x))) / (r11 * r12 + r13 * damp[x - time + 1][y - time + 1][z + 1]);
                    }
#pragma omp simd aligned(damp, usol, vp : 32)
                    for (int sp_zi = sp_zi_m; sp_zi <= nnz_sp_source_mask[x - time][y - time] - 1; sp_zi += 1)
                    {
                      int zind = sp_source_mask[x - time][y - time][sp_zi];
                      float r0 = save_src[((time / sf) % (time_M - time_m + 1))][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
                      usol[t0][x - time + 4][y - time + 4][zind + 4] += r0;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  /* End sectiom*/
  __itt_pause();
  return 0;
}
