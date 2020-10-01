#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
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

struct profiler
{
  double section0;
};

int Kernel(struct dataobj *restrict block_sizes_vec, const float h_x, const float h_y, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_fxx_vec, struct dataobj *restrict save_src_fyy_vec, struct dataobj *restrict save_src_fzz_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict tau_sol_xx_vec, struct dataobj *restrict tau_sol_xy_vec, struct dataobj *restrict tau_sol_xz_vec, struct dataobj *restrict tau_sol_yy_vec, struct dataobj *restrict tau_sol_yz_vec, struct dataobj *restrict tau_sol_zz_vec, struct dataobj *restrict v_sol_x_vec, struct dataobj *restrict v_sol_y_vec, struct dataobj *restrict v_sol_z_vec, const int sp_zi_m, const int time_M, const int time_m, struct profiler *timers, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int nthreads_nonaffine)
{
  int(*restrict block_sizes) __attribute__((aligned(64))) = (int(*))block_sizes_vec->data;
  int(*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[nnz_sp_source_mask_vec->size[1]])nnz_sp_source_mask_vec->data;
  float(*restrict save_src_fxx)[save_src_fxx_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_fxx_vec->size[1]])save_src_fxx_vec->data;
  float(*restrict save_src_fyy)[save_src_fyy_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_fyy_vec->size[1]])save_src_fyy_vec->data;
  float(*restrict save_src_fzz)[save_src_fzz_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_fzz_vec->size[1]])save_src_fzz_vec->data;
  int(*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]][source_id_vec->size[2]])source_id_vec->data;
  int(*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_mask_vec->size[1]][source_mask_vec->size[2]])source_mask_vec->data;
  int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]])sp_source_mask_vec->data;
  float(*restrict tau_sol_xx)[tau_sol_xx_vec->size[1]][tau_sol_xx_vec->size[2]][tau_sol_xx_vec->size[3]] __attribute__((aligned(64))) = (float(*)[tau_sol_xx_vec->size[1]][tau_sol_xx_vec->size[2]][tau_sol_xx_vec->size[3]])tau_sol_xx_vec->data;
  float(*restrict tau_sol_xy)[tau_sol_xy_vec->size[1]][tau_sol_xy_vec->size[2]][tau_sol_xy_vec->size[3]] __attribute__((aligned(64))) = (float(*)[tau_sol_xy_vec->size[1]][tau_sol_xy_vec->size[2]][tau_sol_xy_vec->size[3]])tau_sol_xy_vec->data;
  float(*restrict tau_sol_xz)[tau_sol_xz_vec->size[1]][tau_sol_xz_vec->size[2]][tau_sol_xz_vec->size[3]] __attribute__((aligned(64))) = (float(*)[tau_sol_xz_vec->size[1]][tau_sol_xz_vec->size[2]][tau_sol_xz_vec->size[3]])tau_sol_xz_vec->data;
  float(*restrict tau_sol_yy)[tau_sol_yy_vec->size[1]][tau_sol_yy_vec->size[2]][tau_sol_yy_vec->size[3]] __attribute__((aligned(64))) = (float(*)[tau_sol_yy_vec->size[1]][tau_sol_yy_vec->size[2]][tau_sol_yy_vec->size[3]])tau_sol_yy_vec->data;
  float(*restrict tau_sol_yz)[tau_sol_yz_vec->size[1]][tau_sol_yz_vec->size[2]][tau_sol_yz_vec->size[3]] __attribute__((aligned(64))) = (float(*)[tau_sol_yz_vec->size[1]][tau_sol_yz_vec->size[2]][tau_sol_yz_vec->size[3]])tau_sol_yz_vec->data;
  float(*restrict tau_sol_zz)[tau_sol_zz_vec->size[1]][tau_sol_zz_vec->size[2]][tau_sol_zz_vec->size[3]] __attribute__((aligned(64))) = (float(*)[tau_sol_zz_vec->size[1]][tau_sol_zz_vec->size[2]][tau_sol_zz_vec->size[3]])tau_sol_zz_vec->data;
  float(*restrict v_sol_x)[v_sol_x_vec->size[1]][v_sol_x_vec->size[2]][v_sol_x_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_sol_x_vec->size[1]][v_sol_x_vec->size[2]][v_sol_x_vec->size[3]])v_sol_x_vec->data;
  float(*restrict v_sol_y)[v_sol_y_vec->size[1]][v_sol_y_vec->size[2]][v_sol_y_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_sol_y_vec->size[1]][v_sol_y_vec->size[2]][v_sol_y_vec->size[3]])v_sol_y_vec->data;
  float(*restrict v_sol_z)[v_sol_z_vec->size[1]][v_sol_z_vec->size[2]][v_sol_z_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_sol_z_vec->size[1]][v_sol_z_vec->size[2]][v_sol_z_vec->size[3]])v_sol_z_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  int xb_size = block_sizes[0];
  int y0_blk0_size = block_sizes[3];
  int x0_blk0_size = block_sizes[2];
  int yb_size = block_sizes[1];

  int sf = 12;
  int t_blk_size = 2 * sf * (time_M - time_m);
  //int xb_size = 64;
  //int yb_size = 64;

  //x0_blk0_size = 8;
  //y0_blk0_size = 8;

  printf(" Tiles: %d, %d ::: Blocks %d, %d \n", xb_size, yb_size, x0_blk0_size, y0_blk0_size);
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  for (int t_blk = time_m; t_blk < sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block
  {
    for (int xb = x_m; xb <= (x_M + sf * (time_M - time_m)); xb += xb_size)
    {
      //printf(" Change of outer xblock %d \n", xb);
      for (int yb = y_m; yb <= (y_M + sf * (time_M - time_m)); yb += yb_size)
      {
        for (int time = t_blk, t0 = (time) % (2), t1 = (time + 1) % (2); time <= 1 + min(t_blk + t_blk_size - 1, sf * (time_M - time_m)); time += sf, t0 = (((time / sf) % (time_M - time_m + 1)) + 1) % (2), t1 = (((time / sf) % (time_M - time_m + 1))) % (2))
        {
          int tw = ((time / sf) % (time_M - time_m + 1));
#pragma omp parallel num_threads(nthreads)
          {
//printf(" Change of time block :  %d \n", tw);
#pragma omp for collapse(2) schedule(dynamic, 1)
            for (int x0_blk0 = max((x_m + time), xb); x0_blk0 <= min((x_M + time), (xb + xb_size)); x0_blk0 += x0_blk0_size)
            {
              for (int y0_blk0 = max((y_m + time), yb); y0_blk0 <= min((y_M + time), (yb + yb_size)); y0_blk0 += y0_blk0_size)
              {
                //printf(" Change of inner xblock %d \n", x0_blk0);
                for (int x = x0_blk0; x <= min(min((x_M + time), (xb + xb_size - 1)), (x0_blk0 + x0_blk0_size - 1)); x++)
                {
                  for (int y = y0_blk0; y <= min(min((y_M + time), (yb + yb_size - 1)), (y0_blk0 + y0_blk0_size - 1)); y++)
                  {
                    //printf(" Updating velocity x %d \n", x - time + 4);
                    //printf(" \n PDE update : \n");
#pragma omp simd aligned(tau_sol_xx, tau_sol_xz, tau_sol_zz, v_sol_x, v_sol_z : 32)
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      //printf(" Updating velocity x %d z: %d \n", x - time + 4, z + 4);
                      float r26 = 1.0 / h_z;
                      float r25 = 1.0 / h_y;
                      float r24 = 1.0 / h_x;
                      v_sol_x[t1][x - time + 12][y - time + 12][z + 12] = r24 * (1.43043849497863e-5F * (tau_sol_xx[t0][x - time + 7][y - time + 12][z + 12] - tau_sol_xx[t0][x - time + 18][y - time + 12][z + 12]) + 2.35051066246684e-4F * (-tau_sol_xx[t0][x - time + 8][y - time + 12][z + 12] + tau_sol_xx[t0][x - time + 17][y - time + 12][z + 12]) + 1.94276901694599e-3F * (tau_sol_xx[t0][x - time + 9][y - time + 12][z + 12] - tau_sol_xx[t0][x - time + 16][y - time + 12][z + 12]) + 1.14234818208691e-2F * (-tau_sol_xx[t0][x - time + 10][y - time + 12][z + 12] + tau_sol_xx[t0][x - time + 15][y - time + 12][z + 12]) + 6.34637878910706e-2F * (tau_sol_xx[t0][x - time + 11][y - time + 12][z + 12] - tau_sol_xx[t0][x - time + 14][y - time + 12][z + 12]) + 7.9964372742749e-1F * (-tau_sol_xx[t0][x - time + 12][y - time + 12][z + 12] + tau_sol_xx[t0][x - time + 13][y - time + 12][z + 12])) + r25 * (1.43043849497863e-5F * (tau_sol_xy[t0][x - time + 12][y - time + 6][z + 12] - tau_sol_xy[t0][x - time + 12][y - time + 17][z + 12]) + 2.35051066246684e-4F * (-tau_sol_xy[t0][x - time + 12][y - time + 7][z + 12] + tau_sol_xy[t0][x - time + 12][y - time + 16][z + 12]) + 1.94276901694599e-3F * (tau_sol_xy[t0][x - time + 12][y - time + 8][z + 12] - tau_sol_xy[t0][x - time + 12][y - time + 15][z + 12]) + 1.14234818208691e-2F * (-tau_sol_xy[t0][x - time + 12][y - time + 9][z + 12] + tau_sol_xy[t0][x - time + 12][y - time + 14][z + 12]) + 6.34637878910706e-2F * (tau_sol_xy[t0][x - time + 12][y - time + 10][z + 12] - tau_sol_xy[t0][x - time + 12][y - time + 13][z + 12]) + 7.9964372742749e-1F * (-tau_sol_xy[t0][x - time + 12][y - time + 11][z + 12] + tau_sol_xy[t0][x - time + 12][y - time + 12][z + 12])) + r26 * (1.43043849497863e-5F * (tau_sol_xz[t0][x - time + 12][y - time + 12][z + 6] - tau_sol_xz[t0][x - time + 12][y - time + 12][z + 17]) + 2.35051066246684e-4F * (-tau_sol_xz[t0][x - time + 12][y - time + 12][z + 7] + tau_sol_xz[t0][x - time + 12][y - time + 12][z + 16]) + 1.94276901694599e-3F * (tau_sol_xz[t0][x - time + 12][y - time + 12][z + 8] - tau_sol_xz[t0][x - time + 12][y - time + 12][z + 15]) + 1.14234818208691e-2F * (-tau_sol_xz[t0][x - time + 12][y - time + 12][z + 9] + tau_sol_xz[t0][x - time + 12][y - time + 12][z + 14]) + 6.34637878910706e-2F * (tau_sol_xz[t0][x - time + 12][y - time + 12][z + 10] - tau_sol_xz[t0][x - time + 12][y - time + 12][z + 13]) + 7.9964372742749e-1F * (-tau_sol_xz[t0][x - time + 12][y - time + 12][z + 11] + tau_sol_xz[t0][x - time + 12][y - time + 12][z + 12])) + v_sol_x[t0][x - time + 12][y - time + 12][z + 12];
                      v_sol_y[t1][x - time + 12][y - time + 12][z + 12] = r24 * (1.43043849497863e-5F * (tau_sol_xy[t0][x - time + 6][y - time + 12][z + 12] - tau_sol_xy[t0][x - time + 17][y - time + 12][z + 12]) + 2.35051066246684e-4F * (-tau_sol_xy[t0][x - time + 7][y - time + 12][z + 12] + tau_sol_xy[t0][x - time + 16][y - time + 12][z + 12]) + 1.94276901694599e-3F * (tau_sol_xy[t0][x - time + 8][y - time + 12][z + 12] - tau_sol_xy[t0][x - time + 15][y - time + 12][z + 12]) + 1.14234818208691e-2F * (-tau_sol_xy[t0][x - time + 9][y - time + 12][z + 12] + tau_sol_xy[t0][x - time + 14][y - time + 12][z + 12]) + 6.34637878910706e-2F * (tau_sol_xy[t0][x - time + 10][y - time + 12][z + 12] - tau_sol_xy[t0][x - time + 13][y - time + 12][z + 12]) + 7.9964372742749e-1F * (-tau_sol_xy[t0][x - time + 11][y - time + 12][z + 12] + tau_sol_xy[t0][x - time + 12][y - time + 12][z + 12])) + r25 * (1.43043849497863e-5F * (tau_sol_yy[t0][x - time + 12][y - time + 7][z + 12] - tau_sol_yy[t0][x - time + 12][y - time + 18][z + 12]) + 2.35051066246684e-4F * (-tau_sol_yy[t0][x - time + 12][y - time + 8][z + 12] + tau_sol_yy[t0][x - time + 12][y - time + 17][z + 12]) + 1.94276901694599e-3F * (tau_sol_yy[t0][x - time + 12][y - time + 9][z + 12] - tau_sol_yy[t0][x - time + 12][y - time + 16][z + 12]) + 1.14234818208691e-2F * (-tau_sol_yy[t0][x - time + 12][y - time + 10][z + 12] + tau_sol_yy[t0][x - time + 12][y - time + 15][z + 12]) + 6.34637878910706e-2F * (tau_sol_yy[t0][x - time + 12][y - time + 11][z + 12] - tau_sol_yy[t0][x - time + 12][y - time + 14][z + 12]) + 7.9964372742749e-1F * (-tau_sol_yy[t0][x - time + 12][y - time + 12][z + 12] + tau_sol_yy[t0][x - time + 12][y - time + 13][z + 12])) + r26 * (1.43043849497863e-5F * (tau_sol_yz[t0][x - time + 12][y - time + 12][z + 6] - tau_sol_yz[t0][x - time + 12][y - time + 12][z + 17]) + 2.35051066246684e-4F * (-tau_sol_yz[t0][x - time + 12][y - time + 12][z + 7] + tau_sol_yz[t0][x - time + 12][y - time + 12][z + 16]) + 1.94276901694599e-3F * (tau_sol_yz[t0][x - time + 12][y - time + 12][z + 8] - tau_sol_yz[t0][x - time + 12][y - time + 12][z + 15]) + 1.14234818208691e-2F * (-tau_sol_yz[t0][x - time + 12][y - time + 12][z + 9] + tau_sol_yz[t0][x - time + 12][y - time + 12][z + 14]) + 6.34637878910706e-2F * (tau_sol_yz[t0][x - time + 12][y - time + 12][z + 10] - tau_sol_yz[t0][x - time + 12][y - time + 12][z + 13]) + 7.9964372742749e-1F * (-tau_sol_yz[t0][x - time + 12][y - time + 12][z + 11] + tau_sol_yz[t0][x - time + 12][y - time + 12][z + 12])) + v_sol_y[t0][x - time + 12][y - time + 12][z + 12];
                      v_sol_z[t1][x - time + 12][y - time + 12][z + 12] = r24 * (1.43043849497863e-5F * (tau_sol_xz[t0][x - time + 6][y - time + 12][z + 12] - tau_sol_xz[t0][x - time + 17][y - time + 12][z + 12]) + 2.35051066246684e-4F * (-tau_sol_xz[t0][x - time + 7][y - time + 12][z + 12] + tau_sol_xz[t0][x - time + 16][y - time + 12][z + 12]) + 1.94276901694599e-3F * (tau_sol_xz[t0][x - time + 8][y - time + 12][z + 12] - tau_sol_xz[t0][x - time + 15][y - time + 12][z + 12]) + 1.14234818208691e-2F * (-tau_sol_xz[t0][x - time + 9][y - time + 12][z + 12] + tau_sol_xz[t0][x - time + 14][y - time + 12][z + 12]) + 6.34637878910706e-2F * (tau_sol_xz[t0][x - time + 10][y - time + 12][z + 12] - tau_sol_xz[t0][x - time + 13][y - time + 12][z + 12]) + 7.9964372742749e-1F * (-tau_sol_xz[t0][x - time + 11][y - time + 12][z + 12] + tau_sol_xz[t0][x - time + 12][y - time + 12][z + 12])) + r25 * (1.43043849497863e-5F * (tau_sol_yz[t0][x - time + 12][y - time + 6][z + 12] - tau_sol_yz[t0][x - time + 12][y - time + 17][z + 12]) + 2.35051066246684e-4F * (-tau_sol_yz[t0][x - time + 12][y - time + 7][z + 12] + tau_sol_yz[t0][x - time + 12][y - time + 16][z + 12]) + 1.94276901694599e-3F * (tau_sol_yz[t0][x - time + 12][y - time + 8][z + 12] - tau_sol_yz[t0][x - time + 12][y - time + 15][z + 12]) + 1.14234818208691e-2F * (-tau_sol_yz[t0][x - time + 12][y - time + 9][z + 12] + tau_sol_yz[t0][x - time + 12][y - time + 14][z + 12]) + 6.34637878910706e-2F * (tau_sol_yz[t0][x - time + 12][y - time + 10][z + 12] - tau_sol_yz[t0][x - time + 12][y - time + 13][z + 12]) + 7.9964372742749e-1F * (-tau_sol_yz[t0][x - time + 12][y - time + 11][z + 12] + tau_sol_yz[t0][x - time + 12][y - time + 12][z + 12])) + r26 * (1.43043849497863e-5F * (tau_sol_zz[t0][x - time + 12][y - time + 12][z + 7] - tau_sol_zz[t0][x - time + 12][y - time + 12][z + 18]) + 2.35051066246684e-4F * (-tau_sol_zz[t0][x - time + 12][y - time + 12][z + 8] + tau_sol_zz[t0][x - time + 12][y - time + 12][z + 17]) + 1.94276901694599e-3F * (tau_sol_zz[t0][x - time + 12][y - time + 12][z + 9] - tau_sol_zz[t0][x - time + 12][y - time + 12][z + 16]) + 1.14234818208691e-2F * (-tau_sol_zz[t0][x - time + 12][y - time + 12][z + 10] + tau_sol_zz[t0][x - time + 12][y - time + 12][z + 15]) + 6.34637878910706e-2F * (tau_sol_zz[t0][x - time + 12][y - time + 12][z + 11] - tau_sol_zz[t0][x - time + 12][y - time + 12][z + 14]) + 7.9964372742749e-1F * (-tau_sol_zz[t0][x - time + 12][y - time + 12][z + 12] + tau_sol_zz[t0][x - time + 12][y - time + 12][z + 13])) + v_sol_z[t0][x - time + 12][y - time + 12][z + 12];
                    }
                  }
                }
              }
            }
          }
#pragma omp parallel num_threads(nthreads)
          {
#pragma omp for collapse(2) schedule(dynamic, 1)
            for (int x0_blk0 = max((x_m + time), xb - 6); x0_blk0 <= +min((x_M + time), (xb - 6 + xb_size)); x0_blk0 += x0_blk0_size)
            {
              for (int y0_blk0 = max((y_m + time), yb - 6); y0_blk0 <= +min((y_M + time), (yb - 6 + yb_size)); y0_blk0 += y0_blk0_size)
              {
                for (int x = x0_blk0; x <= min(min((x_M + time), (xb - 6 + xb_size - 1)), (x0_blk0 + x0_blk0_size - 1)); x++)
                {
                  for (int y = y0_blk0; y <= min(min((y_M + time), (yb - 6 + yb_size - 1)), (y0_blk0 + y0_blk0_size - 1)); y++)
                  {
//printf(" Updating stress x %d \n", x - time + 4);
#pragma omp simd aligned(tau_sol_xx, tau_sol_xz, tau_sol_zz, v_sol_x, v_sol_z : 32)
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      //printf(" Updating x %d z: %d \n", x - time + 4, z + 4);
                      float r53 = -v_sol_z[t1][x - time + 12][y - time + 12][z + 12];
                      float r52 = -v_sol_y[t1][x - time + 12][y - time + 12][z + 12];
                      float r51 = -v_sol_x[t1][x - time + 12][y - time + 12][z + 12];
                      float r50 = v_sol_y[t1][x - time + 12][y - time + 8][z + 12] - v_sol_y[t1][x - time + 12][y - time + 15][z + 12];
                      float r49 = -v_sol_y[t1][x - time + 12][y - time + 11][z + 12] + v_sol_y[t1][x - time + 12][y - time + 12][z + 12];
                      float r48 = -v_sol_y[t1][x - time + 12][y - time + 9][z + 12] + v_sol_y[t1][x - time + 12][y - time + 14][z + 12];
                      float r47 = -v_sol_y[t1][x - time + 12][y - time + 7][z + 12] + v_sol_y[t1][x - time + 12][y - time + 16][z + 12];
                      float r46 = v_sol_y[t1][x - time + 12][y - time + 6][z + 12] - v_sol_y[t1][x - time + 12][y - time + 17][z + 12];
                      float r45 = v_sol_y[t1][x - time + 12][y - time + 10][z + 12] - v_sol_y[t1][x - time + 12][y - time + 13][z + 12];
                      float r44 = 1.0 / h_y;
                      float r43 = v_sol_z[t1][x - time + 12][y - time + 12][z + 8] - v_sol_z[t1][x - time + 12][y - time + 12][z + 15];
                      float r42 = -v_sol_z[t1][x - time + 12][y - time + 12][z + 11] + v_sol_z[t1][x - time + 12][y - time + 12][z + 12];
                      float r41 = -v_sol_z[t1][x - time + 12][y - time + 12][z + 9] + v_sol_z[t1][x - time + 12][y - time + 12][z + 14];
                      float r40 = -v_sol_z[t1][x - time + 12][y - time + 12][z + 7] + v_sol_z[t1][x - time + 12][y - time + 12][z + 16];
                      float r39 = v_sol_z[t1][x - time + 12][y - time + 12][z + 6] - v_sol_z[t1][x - time + 12][y - time + 12][z + 17];
                      float r38 = v_sol_z[t1][x - time + 12][y - time + 12][z + 10] - v_sol_z[t1][x - time + 12][y - time + 12][z + 13];
                      float r37 = 1.0 / h_z;
                      float r36 = v_sol_x[t1][x - time + 8][y - time + 12][z + 12] - v_sol_x[t1][x - time + 15][y - time + 12][z + 12];
                      float r35 = -v_sol_x[t1][x - time + 11][y - time + 12][z + 12] + v_sol_x[t1][x - time + 12][y - time + 12][z + 12];
                      float r34 = -v_sol_x[t1][x - time + 9][y - time + 12][z + 12] + v_sol_x[t1][x - time + 14][y - time + 12][z + 12];
                      float r33 = -v_sol_x[t1][x - time + 7][y - time + 12][z + 12] + v_sol_x[t1][x - time + 16][y - time + 12][z + 12];
                      float r32 = v_sol_x[t1][x - time + 6][y - time + 12][z + 12] - v_sol_x[t1][x - time + 17][y - time + 12][z + 12];
                      float r31 = v_sol_x[t1][x - time + 10][y - time + 12][z + 12] - v_sol_x[t1][x - time + 13][y - time + 12][z + 12];
                      float r30 = 1.0 / h_x;
                      float r29 = r30 * (4.11245345534138e-1F * r31 + 9.26924144746152e-5F * r32 + 1.52313090927851e-3F * r33 + 7.40241621992317e-2F * r34 + 5.18169135373014F * r35 + 1.258914322981e-2F * r36);
                      float r28 = r37 * (4.11245345534138e-1F * r38 + 9.26924144746152e-5F * r39 + 1.52313090927851e-3F * r40 + 7.40241621992317e-2F * r41 + 5.18169135373014F * r42 + 1.258914322981e-2F * r43);
                      float r27 = r44 * (4.11245345534138e-1F * r45 + 9.26924144746152e-5F * r46 + 1.52313090927851e-3F * r47 + 7.40241621992317e-2F * r48 + 5.18169135373014F * r49 + 1.258914322981e-2F * r50);
                      tau_sol_xx[t1][x - time + 12][y - time + 12][z + 12] = r27 + r28 + r30 * (8.22490691068276e-1F * r31 + 1.8538482894923e-4F * r32 + 3.04626181855702e-3F * r33 + 1.48048324398463e-1F * r34 + 1.03633827074603e+1F * r35 + 2.517828645962e-2F * r36) + tau_sol_xx[t0][x - time + 12][y - time + 12][z + 12];
                      tau_sol_xy[t1][x - time + 12][y - time + 12][z + 12] = r30 * (2.59084567686507F * (r52 + v_sol_y[t1][x - time + 13][y - time + 12][z + 12]) + 4.63462072373076e-5F * (v_sol_y[t1][x - time + 7][y - time + 12][z + 12] - v_sol_y[t1][x - time + 18][y - time + 12][z + 12]) + 7.61565454639255e-4F * (-v_sol_y[t1][x - time + 8][y - time + 12][z + 12] + v_sol_y[t1][x - time + 17][y - time + 12][z + 12]) + 6.29457161490501e-3F * (v_sol_y[t1][x - time + 9][y - time + 12][z + 12] - v_sol_y[t1][x - time + 16][y - time + 12][z + 12]) + 3.70120810996159e-2F * (-v_sol_y[t1][x - time + 10][y - time + 12][z + 12] + v_sol_y[t1][x - time + 15][y - time + 12][z + 12]) + 2.05622672767069e-1F * (v_sol_y[t1][x - time + 11][y - time + 12][z + 12] - v_sol_y[t1][x - time + 14][y - time + 12][z + 12])) + r44 * (2.59084567686507F * (r51 + v_sol_x[t1][x - time + 12][y - time + 13][z + 12]) + 4.63462072373076e-5F * (v_sol_x[t1][x - time + 12][y - time + 7][z + 12] - v_sol_x[t1][x - time + 12][y - time + 18][z + 12]) + 7.61565454639255e-4F * (-v_sol_x[t1][x - time + 12][y - time + 8][z + 12] + v_sol_x[t1][x - time + 12][y - time + 17][z + 12]) + 6.29457161490501e-3F * (v_sol_x[t1][x - time + 12][y - time + 9][z + 12] - v_sol_x[t1][x - time + 12][y - time + 16][z + 12]) + 3.70120810996159e-2F * (-v_sol_x[t1][x - time + 12][y - time + 10][z + 12] + v_sol_x[t1][x - time + 12][y - time + 15][z + 12]) + 2.05622672767069e-1F * (v_sol_x[t1][x - time + 12][y - time + 11][z + 12] - v_sol_x[t1][x - time + 12][y - time + 14][z + 12])) + tau_sol_xy[t0][x - time + 12][y - time + 12][z + 12];
                      tau_sol_xz[t1][x - time + 12][y - time + 12][z + 12] = r30 * (2.59084567686507F * (r53 + v_sol_z[t1][x - time + 13][y - time + 12][z + 12]) + 4.63462072373076e-5F * (v_sol_z[t1][x - time + 7][y - time + 12][z + 12] - v_sol_z[t1][x - time + 18][y - time + 12][z + 12]) + 7.61565454639255e-4F * (-v_sol_z[t1][x - time + 8][y - time + 12][z + 12] + v_sol_z[t1][x - time + 17][y - time + 12][z + 12]) + 6.29457161490501e-3F * (v_sol_z[t1][x - time + 9][y - time + 12][z + 12] - v_sol_z[t1][x - time + 16][y - time + 12][z + 12]) + 3.70120810996159e-2F * (-v_sol_z[t1][x - time + 10][y - time + 12][z + 12] + v_sol_z[t1][x - time + 15][y - time + 12][z + 12]) + 2.05622672767069e-1F * (v_sol_z[t1][x - time + 11][y - time + 12][z + 12] - v_sol_z[t1][x - time + 14][y - time + 12][z + 12])) + r37 * (2.59084567686507F * (r51 + v_sol_x[t1][x - time + 12][y - time + 12][z + 13]) + 4.63462072373076e-5F * (v_sol_x[t1][x - time + 12][y - time + 12][z + 7] - v_sol_x[t1][x - time + 12][y - time + 12][z + 18]) + 7.61565454639255e-4F * (-v_sol_x[t1][x - time + 12][y - time + 12][z + 8] + v_sol_x[t1][x - time + 12][y - time + 12][z + 17]) + 6.29457161490501e-3F * (v_sol_x[t1][x - time + 12][y - time + 12][z + 9] - v_sol_x[t1][x - time + 12][y - time + 12][z + 16]) + 3.70120810996159e-2F * (-v_sol_x[t1][x - time + 12][y - time + 12][z + 10] + v_sol_x[t1][x - time + 12][y - time + 12][z + 15]) + 2.05622672767069e-1F * (v_sol_x[t1][x - time + 12][y - time + 12][z + 11] - v_sol_x[t1][x - time + 12][y - time + 12][z + 14])) + tau_sol_xz[t0][x - time + 12][y - time + 12][z + 12];
                      tau_sol_yy[t1][x - time + 12][y - time + 12][z + 12] = r28 + r29 + r44 * (8.22490691068276e-1F * r45 + 1.8538482894923e-4F * r46 + 3.04626181855702e-3F * r47 + 1.48048324398463e-1F * r48 + 1.03633827074603e+1F * r49 + 2.517828645962e-2F * r50) + tau_sol_yy[t0][x - time + 12][y - time + 12][z + 12];
                      tau_sol_yz[t1][x - time + 12][y - time + 12][z + 12] = r37 * (2.59084567686507F * (r52 + v_sol_y[t1][x - time + 12][y - time + 12][z + 13]) + 4.63462072373076e-5F * (v_sol_y[t1][x - time + 12][y - time + 12][z + 7] - v_sol_y[t1][x - time + 12][y - time + 12][z + 18]) + 7.61565454639255e-4F * (-v_sol_y[t1][x - time + 12][y - time + 12][z + 8] + v_sol_y[t1][x - time + 12][y - time + 12][z + 17]) + 6.29457161490501e-3F * (v_sol_y[t1][x - time + 12][y - time + 12][z + 9] - v_sol_y[t1][x - time + 12][y - time + 12][z + 16]) + 3.70120810996159e-2F * (-v_sol_y[t1][x - time + 12][y - time + 12][z + 10] + v_sol_y[t1][x - time + 12][y - time + 12][z + 15]) + 2.05622672767069e-1F * (v_sol_y[t1][x - time + 12][y - time + 12][z + 11] - v_sol_y[t1][x - time + 12][y - time + 12][z + 14])) + r44 * (2.59084567686507F * (r53 + v_sol_z[t1][x - time + 12][y - time + 13][z + 12]) + 4.63462072373076e-5F * (v_sol_z[t1][x - time + 12][y - time + 7][z + 12] - v_sol_z[t1][x - time + 12][y - time + 18][z + 12]) + 7.61565454639255e-4F * (-v_sol_z[t1][x - time + 12][y - time + 8][z + 12] + v_sol_z[t1][x - time + 12][y - time + 17][z + 12]) + 6.29457161490501e-3F * (v_sol_z[t1][x - time + 12][y - time + 9][z + 12] - v_sol_z[t1][x - time + 12][y - time + 16][z + 12]) + 3.70120810996159e-2F * (-v_sol_z[t1][x - time + 12][y - time + 10][z + 12] + v_sol_z[t1][x - time + 12][y - time + 15][z + 12]) + 2.05622672767069e-1F * (v_sol_z[t1][x - time + 12][y - time + 11][z + 12] - v_sol_z[t1][x - time + 12][y - time + 14][z + 12])) + tau_sol_yz[t0][x - time + 12][y - time + 12][z + 12];
                      tau_sol_zz[t1][x - time + 12][y - time + 12][z + 12] = r27 + r29 + r37 * (8.22490691068276e-1F * r38 + 1.8538482894923e-4F * r39 + 3.04626181855702e-3F * r40 + 1.48048324398463e-1F * r41 + 1.03633827074603e+1F * r42 + 2.517828645962e-2F * r43) + tau_sol_zz[t0][x - time + 12][y - time + 12][z + 12];
                    }
                    for (int sp_zi = sp_zi_m; sp_zi <= nnz_sp_source_mask[x - time][y - time] - 1; sp_zi += 1)
                    {
                      //printf("\n Source_injection at : ");
                      int zind = sp_source_mask[x - time][y - time][sp_zi];
                      float r0 = save_src_fxx[((time / sf) % (time_M - time_m + 1))][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
                      float r1 = save_src_fyy[((time / sf) % (time_M - time_m + 1))][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
                      float r2 = save_src_fzz[((time / sf) % (time_M - time_m + 1))][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
                      tau_sol_xx[t1][x - time + 12][y - time + 12][zind + 12] += r0;
                      tau_sol_yy[t1][x - time + 12][y - time + 12][zind + 12] += r1;
                      tau_sol_zz[t1][x - time + 12][y - time + 12][zind + 12] += r2;
                      //printf(" Time %d , at : %d, %d \n", tw, x - time + 4, zind + 4);
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

  /* End section0 */
  gettimeofday(&end_section0, NULL);
  timers->section0 += (double)(end_section0.tv_sec - start_section0.tv_sec) + (double)(end_section0.tv_usec - start_section0.tv_usec) / 1000000;
  return 0;
}
/* Backdoor edit at Tue Jul 28 12:03:11 2020*/
/* Backdoor edit at Tue Jul 28 12:27:09 2020*/
/* Backdoor edit at Tue Jul 28 12:28:10 2020*/
/* Backdoor edit at Tue Jul 28 15:17:20 2020*/
