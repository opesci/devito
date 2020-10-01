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

void bf0(float *restrict r18_vec, float *restrict r19_vec, float *restrict r20_vec, float *restrict r21_vec, float *restrict r34_vec, float *restrict r35_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, const int x_size, const int y_size, const int z_size, const int time, const int t0, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, const int tw);

void bf1(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r17_vec, float *restrict r18_vec, float *restrict r19_vec, float *restrict r20_vec, float *restrict r21_vec, float *restrict r34_vec, float *restrict r35_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, const int x_size, const int y_size, const int z_size, const int time, const int t0, const int t1, const int t2, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int sp_zi_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, const int tw);

int ForwardTTI(struct dataobj *restrict block_sizes_vec, struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, const float dt, struct dataobj *restrict epsilon_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict phi_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict theta_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x_size, const int y_size, const int z_size, const int sp_zi_m, const int time_M, const int time_m, struct profiler *timers, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int nthreads_nonaffine)
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

  float(*r21)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r21, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r20)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r20, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r19)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r19, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r18)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r18, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r17)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r17, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r34)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r34, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float(*r35)[y_size + 1][z_size + 1];
  posix_memalign((void **)&r35, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
/* Begin section0 */
#pragma omp parallel num_threads(nthreads)
  {
#pragma omp for collapse(1) schedule(static, 1)
    for (int x = x_m - 1; x <= x_M; x += 1)
    {
      for (int y = y_m - 1; y <= y_M; y += 1)
      {
#pragma omp simd aligned(delta, phi, theta : 32)
        for (int z = z_m - 1; z <= z_M; z += 1)
        {
          r21[x + 1][y + 1][z + 1] = cos(phi[x + 4][y + 4][z + 4]);
          r20[x + 1][y + 1][z + 1] = sin(theta[x + 4][y + 4][z + 4]);
          r19[x + 1][y + 1][z + 1] = sin(phi[x + 4][y + 4][z + 4]);
          r18[x + 1][y + 1][z + 1] = cos(theta[x + 4][y + 4][z + 4]);
          r17[x + 1][y + 1][z + 1] = sqrt(2 * delta[x + 4][y + 4][z + 4] + 1);
        }
      }
    }
  }
  /* End section0 */
  gettimeofday(&end_section0, NULL);
  timers->section0 += (double)(end_section0.tv_sec - start_section0.tv_sec) + (double)(end_section0.tv_usec - start_section0.tv_usec) / 1000000;

  int y0_blk0_size = 8; //block_sizes[3];
  int x0_blk0_size = 8; //block_sizes[2];
  int yb_size = 16; // block_sizes[1];
  int xb_size = 16; //block_sizes[0];
  int sf = 4;
  int t_blk_size = 2 * sf * (time_M - time_m);

  printf(" Tiles: %d, %d ::: Blocks %d, %d \n", xb_size, yb_size, x0_blk0_size, y0_blk0_size);

  for (int t_blk = time_m; t_blk < sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block
  {
    for (int xb = x_m; xb <= (x_M + sf * (time_M - time_m)); xb += xb_size)
    {
      //printf(" Change of outer xblock %d \n", xb);
      for (int yb = y_m; yb <= (y_M + sf * (time_M - time_m)); yb += yb_size)
      {
        //printf(" Timestep tw: %d, Updating x: %d y: %d \n", xb, yb);

        for (int time = t_blk, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3); time <= 1 + min(t_blk + t_blk_size - 1, sf * (time_M - time_m)); time += sf, t0 = (((time / sf) % (time_M - time_m + 1))) % (3), t1 = (((time / sf) % (time_M - time_m + 1)) + 1) % (3), t2 = (((time / sf) % (time_M - time_m + 1)) + 2) % (3))
        {
          int tw = ((time / sf) % (time_M - time_m + 1));
          struct timeval start_section1, end_section1;
          gettimeofday(&start_section1, NULL);
          /* Begin section1 */

          //bf0((float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, x_size, y_size, z_size, time, t0, x0_blk0_size, x_M - (x_M - x_m + 2) % (x0_blk0_size), x_m - 1, y0_blk0_size, y_M - (y_M - y_m + 2) % (y0_blk0_size), y_m - 1, z_M, z_m, nthreads, xb, yb, xb_size, yb_size, tw);
          bf0((float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, x_size, y_size, z_size, time, t0, x0_blk0_size, x_M, x_m , y0_blk0_size, y_M , y_m , z_M, z_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf("\n BF0 - 1 IS OVER");

          //bf0((float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, x_size, y_size, z_size, time, t0, x0_blk0_size, x_M - (x_M - x_m + 2) % (x0_blk0_size), x_m - 1, (y_M - y_m + 2) % (y0_blk0_size), y_M, y_M - (y_M - y_m + 2) % (y0_blk0_size) + 1, z_M, z_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf(" BF0 - 2 IS OVER");
          //bf0((float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, x_size, y_size, z_size, time, t0, (x_M - x_m + 2) % (x0_blk0_size), x_M, x_M - (x_M - x_m + 2) % (x0_blk0_size) + 1, y0_blk0_size, y_M - (y_M - y_m + 2) % (y0_blk0_size), y_m - 1, z_M, z_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf(" BF0 - 3 IS OVER");

          //bf0((float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, x_size, y_size, z_size, time, t0, (x_M - x_m + 2) % (x0_blk0_size), x_M, x_M - (x_M - x_m + 2) % (x0_blk0_size) + 1, (y_M - y_m + 2) % (y0_blk0_size), y_M, y_M - (y_M - y_m + 2) % (y0_blk0_size) + 1, z_M, z_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf(" BF0 - 4 IS OVER");

          /*==============================================*/
          //bf1(damp_vec, dt, epsilon_vec, (float *)r17, (float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, vp_vec, nnz_sp_source_mask_vec, sp_source_mask_vec, save_src_u_vec, save_src_v_vec, source_id_vec, source_mask_vec, x_size, y_size, z_size, time, t0, t1, t2, x1_blk0_size, -2 + x_M - (x_M - x_m + 1) % (x1_blk0_size), x_m, y1_blk0_size, -2 + y_M - (y_M - y_m + 1) % (y1_blk0_size), y_m, z_M, z_m, sp_zi_m, nthreads, xb, yb, xb_size, yb_size, tw);
          bf1(damp_vec, dt, epsilon_vec, (float *)r17, (float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, vp_vec, nnz_sp_source_mask_vec, sp_source_mask_vec, save_src_u_vec, save_src_v_vec, source_id_vec, source_mask_vec, x_size, y_size, z_size, time, t0, t1, t2, x0_blk0_size, x_M, x_m, y0_blk0_size, y_M, y_m, z_M, z_m, sp_zi_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf("\n BF1 - 1 IS OVER");

          //bf1(damp_vec, dt, epsilon_vec, (float *)r17, (float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, vp_vec, nnz_sp_source_mask_vec, sp_source_mask_vec, save_src_u_vec, save_src_v_vec, source_id_vec, source_mask_vec, x_size, y_size, z_size, time, t0, t1, t2, x1_blk0_size, x_M - (x_M - x_m + 1) % (x1_blk0_size), x_m, (y_M - y_m + 1) % (y1_blk0_size), y_M, y_M - (y_M - y_m + 1) % (y1_blk0_size) + 1, z_M, z_m, sp_zi_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf(" BF1 - 2 IS OVER");

          //bf1(damp_vec, dt, epsilon_vec, (float *)r17, (float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, vp_vec, nnz_sp_source_mask_vec, sp_source_mask_vec, save_src_u_vec, save_src_v_vec, source_id_vec, source_mask_vec, x_size, y_size, z_size, time, t0, t1, t2, (x_M - x_m + 1) % (x1_blk0_size), x_M, x_M - (x_M - x_m + 1) % (x1_blk0_size) + 1, y1_blk0_size, y_M - (y_M - y_m + 1) % (y1_blk0_size), y_m, z_M, z_m, sp_zi_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf(" BF1 - 3 IS OVER");

          //bf1(damp_vec, dt, epsilon_vec, (float *)r17, (float *)r18, (float *)r19, (float *)r20, (float *)r21, (float *)r34, (float *)r35, u_vec, v_vec, vp_vec, nnz_sp_source_mask_vec, sp_source_mask_vec, save_src_u_vec, save_src_v_vec, source_id_vec, source_mask_vec, x_size, y_size, z_size, time, t0, t1, t2, (x_M - x_m + 1) % (x1_blk0_size), x_M, x_M - (x_M - x_m + 1) % (x1_blk0_size) + 1, (y_M - y_m + 1) % (y1_blk0_size), y_M, y_M - (y_M - y_m + 1) % (y1_blk0_size) + 1, z_M, z_m, sp_zi_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf(" BF1 - 4 IS OVER");

          /* End section1 */
          gettimeofday(&end_section1, NULL);
          timers->section1 += (double)(end_section1.tv_sec - start_section1.tv_sec) + (double)(end_section1.tv_usec - start_section1.tv_usec) / 1000000;
        }
      }
    }
  }
  /*
  for (int time = time_m, t1 = (time + 1) % (3); time <= time_M; time += 1, t1 = (time + 1) % (3))
  {
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
 Begin section2
#pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F / 3.0F) * (x_M - x_m + 1) / nthreads_nonaffine));
#pragma omp for collapse(1) schedule(dynamic, chunk_size)
      for (int x = x_m; x <= x_M; x += 1)
      {
        for (int y = y_m; y <= y_M; y += 1)
        {
        }
      }
    }
     End section2
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec - start_section2.tv_sec) + (double)(end_section2.tv_usec - start_section2.tv_usec) / 1000000;
  }
*/
  free(r21);
  free(r20);
  free(r19);
  free(r18);
  free(r17);
  free(r34);
  free(r35);
  return 0;
}

void bf0(float *restrict r18_vec, float *restrict r19_vec, float *restrict r20_vec, float *restrict r21_vec, float *restrict r34_vec, float *restrict r35_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, const int x_size, const int y_size, const int z_size, const int time, const int t0, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, const int tw)
{
  float(*restrict r18)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r18_vec;
  float(*restrict r19)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r19_vec;
  float(*restrict r20)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r20_vec;
  float(*restrict r21)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r21_vec;
  float(*restrict r34)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r34_vec;
  float(*restrict r35)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r35_vec;
  float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
  float(*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]])v_vec->data;

  if (x0_blk0_size == 0)
  {
    return;
  }

#pragma omp parallel num_threads(nthreads)
  {
#pragma omp for collapse(1) schedule(dynamic, 1)
    for (int x0_blk0 = max((x_m + time), xb); x0_blk0 <= min((x_M + time), (xb + xb_size)); x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = max((y_m + time), yb); y0_blk0 <= min((y_M + time), (yb + yb_size)); y0_blk0 += y0_blk0_size)
      {
        //printf(" Change of inner x0_blk0 %d \n", x0_blk0);
        for (int x = x0_blk0; x <= min(min((x_M + time), (xb + xb_size - 1)), (x0_blk0 + x0_blk0_size - 1)); x++)
        {
          for (int y = y0_blk0; y <= min(min((y_M + time), (yb + yb_size - 1)), (y0_blk0 + y0_blk0_size - 1)); y++)
          {
            //printf(" bf0 Timestep tw: %d, Updating x: %d y: %d \n", tw, x-time+1, y-time+1);
#pragma omp simd aligned(u, v : 32)
            for (int z = z_m - 1 ; z <= z_M; z += 1)
            {
              //printf(" bf0 Updating x: %d y: %d z: %d \n", x - time + 1, y - time + 1,  z + 1);
              float r39 = -v[t0][x - time + 4][y - time + 4][z + 4];
              r35[x - time + 1][y - time + 1][z + 1] = 1.0e-1F * (-(r39 + v[t0][x - time + 4][y - time + 4][z + 5]) * r18[x - time + 1][y - time + 1][z + 1] - (r39 + v[t0][x - time + 4][y - time + 5][z + 4]) * r19[x - time + 1][y - time + 1][z + 1] * r20[x - time + 1][y - time + 1][z + 1] - (r39 + v[t0][x - time + 5][y - time + 4][z + 4]) * r20[x - time + 1][y - time + 1][z + 1] * r21[x - time + 1][y - time + 1][z + 1]);
              float r40 = -u[t0][x - time + 4][y - time + 4][z + 4];
              r34[x - time + 1][y - time + 1][z + 1] = 1.0e-1F * (-(r40 + u[t0][x - time + 4][y - time + 4][z + 5]) * r18[x - time + 1][y - time + 1][z + 1] - (r40 + u[t0][x - time + 4][y - time + 5][z + 4]) * r19[x - time + 1][y - time + 1][z + 1] * r20[x - time + 1][y - time + 1][z + 1] - (r40 + u[t0][x - time + 5][y - time + 4][z + 4]) * r20[x - time + 1][y - time + 1][z + 1] * r21[x - time + 1][y - time + 1][z + 1]);
            }
          }
        }
      }
    }
  }
}

void bf1(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r17_vec, float *restrict r18_vec, float *restrict r19_vec, float *restrict r20_vec, float *restrict r21_vec, float *restrict r34_vec, float *restrict r35_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, const int x_size, const int y_size, const int z_size, const int time, const int t0, const int t1, const int t2, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int sp_zi_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, const int tw)
{
  float(*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[damp_vec->size[1]][damp_vec->size[2]])damp_vec->data;
  float(*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__((aligned(64))) = (float(*)[epsilon_vec->size[1]][epsilon_vec->size[2]])epsilon_vec->data;
  float(*restrict r17)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r17_vec;
  float(*restrict r18)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r18_vec;
  float(*restrict r19)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r19_vec;
  float(*restrict r20)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r20_vec;
  float(*restrict r21)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r21_vec;
  float(*restrict r34)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r34_vec;
  float(*restrict r35)[y_size + 1][z_size + 1] __attribute__((aligned(64))) = (float(*)[y_size + 1][z_size + 1]) r35_vec;
  float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
  float(*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]])v_vec->data;
  float(*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[vp_vec->size[1]][vp_vec->size[2]])vp_vec->data;

  int(*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__((aligned(64))) = (int(*)[nnz_sp_source_mask_vec->size[1]])nnz_sp_source_mask_vec->data;
  float(*restrict save_src_u)[save_src_u_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_u_vec->size[1]])save_src_u_vec->data;
  float(*restrict save_src_v)[save_src_v_vec->size[1]] __attribute__((aligned(64))) = (float(*)[save_src_v_vec->size[1]])save_src_v_vec->data;
  int(*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_id_vec->size[1]][source_id_vec->size[2]])source_id_vec->data;
  int(*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[source_mask_vec->size[1]][source_mask_vec->size[2]])source_mask_vec->data;
  int(*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__((aligned(64))) = (int(*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]])sp_source_mask_vec->data;
  //printf("In bf1 \n");

  if (x1_blk0_size == 0)
  {
    return;
  }
#pragma omp parallel num_threads(nthreads)
  {
#pragma omp for collapse(1) schedule(dynamic, 1)
    for (int x1_blk0 = max((x_m + time), xb - 2 ); x1_blk0 <= +min((x_M + time), (xb - 2 + xb_size)); x1_blk0 += x1_blk0_size)
    {
      //printf(" Change of inner x1_blk0 %d \n", x1_blk0);
      for (int y1_blk0 = max((y_m + time), yb - 2 ); y1_blk0 <= +min((y_M + time), (yb - 2 + yb_size)); y1_blk0 += y1_blk0_size)
      {
        for (int x = x1_blk0; x <= min(min((x_M + time), (xb - 2 + xb_size - 1)), (x1_blk0 + x1_blk0_size - 1)); x++)
        {
          for (int y = y1_blk0; y <= min(min((y_M + time), (yb - 2 + yb_size - 1)), (y1_blk0 + y1_blk0_size - 1)); y++)
          {
            //printf(" bf1 Timestep tw: %d, Updating x: %d y: %d \n", tw, x - time + 4, y - time + 4);
            #pragma omp simd aligned(damp, epsilon, u, v, vp : 32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              //printf(" bf1 Updating x: %d y: %d z: %d \n", x - time + 4, y - time + 4,  z + 4);

              //printf(" bf1 Updating x: %d y: %d z: %d \n", x - time + 4, y - time + 4,  z + 4);
              float r46 = 1.0 / dt;
              float r45 = 1.0 / (dt * dt);
              float r44 = r18[x - time + 1][y - time + 1][z] * r35[x - time + 1][y - time + 1][z] - r18[x - time + 1][y - time + 1][z + 1] * r35[x - time + 1][y - time + 1][z + 1] + r19[x - time + 1][y - time][z + 1] * r20[x - time + 1][y - time][z + 1] * r35[x - time + 1][y - time][z + 1] - r19[x - time + 1][y - time + 1][z + 1] * r20[x - time + 1][y - time + 1][z + 1] * r35[x - time + 1][y - time + 1][z + 1] + r20[x - time][y - time + 1][z + 1] * r21[x - time][y - time + 1][z + 1] * r35[x - time][y - time + 1][z + 1] - r20[x - time + 1][y - time + 1][z + 1] * r21[x - time + 1][y - time + 1][z + 1] * r35[x - time + 1][y - time + 1][z + 1];
              float r43 = pow(vp[x - time + 4][y - time + 4][z + 4], -2);
              float r42 = 1.0e-1F * (-r18[x - time + 1][y - time + 1][z] * r34[x - time + 1][y - time + 1][z] + r18[x - time + 1][y - time + 1][z + 1] * r34[x - time + 1][y - time + 1][z + 1] - r19[x - time + 1][y - time][z + 1] * r20[x - time + 1][y - time][z + 1] * r34[x - time + 1][y - time][z + 1] + r19[x - time + 1][y - time + 1][z + 1] * r20[x - time + 1][y - time + 1][z + 1] * r34[x - time + 1][y - time + 1][z + 1] - r20[x - time][y - time + 1][z + 1] * r21[x - time][y - time + 1][z + 1] * r34[x - time][y - time + 1][z + 1] + r20[x - time + 1][y - time + 1][z + 1] * r21[x - time + 1][y - time + 1][z + 1] * r34[x - time + 1][y - time + 1][z + 1]) - 8.33333315e-4F * (u[t0][x - time + 2][y - time + 4][z + 4] + u[t0][x - time + 4][y - time + 2][z + 4] + u[t0][x - time + 4][y - time + 4][z + 2] + u[t0][x - time + 4][y - time + 4][z + 6] + u[t0][x - time + 4][y - time + 6][z + 4] + u[t0][x - time + 6][y - time + 4][z + 4]) + 1.3333333e-2F * (u[t0][x - time + 3][y - time + 4][z + 4] + u[t0][x - time + 4][y - time + 3][z + 4] + u[t0][x - time + 4][y - time + 4][z + 3] + u[t0][x - time + 4][y - time + 4][z + 5] + u[t0][x - time + 4][y - time + 5][z + 4] + u[t0][x - time + 5][y - time + 4][z + 4]) - 7.49999983e-2F * u[t0][x - time + 4][y - time + 4][z + 4];
              float r41 = 1.0 / (r43 * r45 + r46 * damp[x - time + 1][y - time + 1][z + 1]);
              float r32 = r45 * (-2.0F * u[t0][x - time + 4][y - time + 4][z + 4] + u[t2][x - time + 4][y - time + 4][z + 4]);
              float r33 = r45 * (-2.0F * v[t0][x - time + 4][y - time + 4][z + 4] + v[t2][x - time + 4][y - time + 4][z + 4]);
              u[t1][x - time + 4][y - time + 4][z + 4] = r41 * ((-r32) * r43 + r42 * (2 * epsilon[x - time + 4][y - time + 4][z + 4] + 1) + 1.0e-1F * r44 * r17[x - time + 1][y - time + 1][z + 1] + r46 * (damp[x - time + 1][y - time + 1][z + 1] * u[t0][x - time + 4][y - time + 4][z + 4]));
              v[t1][x - time + 4][y - time + 4][z + 4] = r41 * ((-r33) * r43 + r42 * r17[x - time + 1][y - time + 1][z + 1] + 1.0e-1F * r44 + r46 * (damp[x - time + 1][y - time + 1][z + 1] * v[t0][x - time + 4][y - time + 4][z + 4]));
            }
            //int sp_zi_M = nnz_sp_source_mask[x - time][y - time] - 1;
            for (int sp_zi = sp_zi_m; sp_zi <= nnz_sp_source_mask[x - time][y - time] - 1; sp_zi += 1)
            {
              int zind = sp_source_mask[x - time][y - time][sp_zi];
              float r22 = save_src_u[tw][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
//#pragma omp atomic update
              u[t1][x - time + 4][y - time + 4][zind + 4] += r22;
              float r23 = save_src_v[tw][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
//#pragma omp atomic update
              v[t1][x - time + 4][y - time + 4][zind + 4] += r23;
              //printf("Source injection at time %d , at : x: %d, y: %d, %d, %f, %f \n", tw, x - time + 4, y - time + 4, zind + 4, r22, r23);
            }
          }
        }
      }
    }
  }
}
/* Backdoor edit at Wed Sep  9 19:03:00 2020*/

