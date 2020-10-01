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

void bf0(float *restrict r50_vec, float *restrict r51_vec, float *restrict r52_vec, float *restrict r53_vec, float *restrict r82_vec, float *restrict r83_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, const int x_size, const int y_size, const int z_size, const int time, const int t0, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, const int tw);

void bf1(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r49_vec, float *restrict r50_vec, float *restrict r51_vec, float *restrict r52_vec, float *restrict r53_vec, float *restrict r82_vec, float *restrict r83_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, const int x_size, const int y_size, const int z_size, const int time, const int t0, const int t1, const int t2, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int sp_zi_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, const int tw);

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

  float (*r49)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r49, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r50)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r50, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r51)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r51, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r52)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r52, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r53)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r53, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r82)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r82, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));
  float (*r83)[y_size + 2 + 2][z_size + 2 + 2];
  posix_memalign((void**)&r83, 64, sizeof(float[x_size + 2 + 2][y_size + 2 + 2][z_size + 2 + 2]));

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  /* Begin section0 */
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(static,1)
    for (int x = x_m - 2; x <= x_M + 2; x += 1)
    {
      for (int y = y_m - 2; y <= y_M + 2; y += 1)
      {
        #pragma omp simd aligned(delta,phi,theta:32)
        for (int z = z_m - 2; z <= z_M + 2; z += 1)
        {
          r49[x + 2][y + 2][z + 2] = sqrt(2*delta[x + 8][y + 8][z + 8] + 1);
          r50[x + 2][y + 2][z + 2] = cos(theta[x + 8][y + 8][z + 8]);
          r51[x + 2][y + 2][z + 2] = cos(phi[x + 8][y + 8][z + 8]);
          r52[x + 2][y + 2][z + 2] = sin(theta[x + 8][y + 8][z + 8]);
          r53[x + 2][y + 2][z + 2] = sin(phi[x + 8][y + 8][z + 8]);
        }
      }
    }
  }
  /* End section0 */
  gettimeofday(&end_section0, NULL);
  timers->section0 += (double)(end_section0.tv_sec - start_section0.tv_sec) + (double)(end_section0.tv_usec - start_section0.tv_usec) / 1000000;

  int y0_blk0_size = block_sizes[3];
  int x0_blk0_size = block_sizes[2];
  int yb_size = block_sizes[1];
  int xb_size = block_sizes[0];
  int sf = 4;
  int t_blk_size = 2 * sf * (time_M - time_m);

  printf(" Tiles: %d, %d ::: Blocks %d, %d \n", xb_size, yb_size, x0_blk0_size, y0_blk0_size);

  for (int t_blk = time_m; t_blk <= 1 + sf * (time_M - time_m); t_blk += sf * t_blk_size) // for each t block
  {
    for (int xb = x_m-2 ; xb <= (x_M + 2 + sf * (time_M - time_m)); xb += xb_size)
    {
      //printf(" Change of outer xblock %d \n", xb);
      for (int yb = y_m-2 ; yb <= (y_M+2 + sf * (time_M - time_m)); yb += yb_size)
      {
        //printf(" Timestep tw: %d, Updating x: %d y: %d \n", xb, yb);

        for (int time = t_blk, t0 = (time) % (3), t1 = (time + 2) % (3), t2 = (time + 1) % (3); time <= 2 + min(t_blk + t_blk_size - 1, sf * (time_M - time_m)); time += sf, t0 = (((time / sf) % (time_M - time_m + 1))) % (3), t1 = (((time / sf) % (time_M - time_m + 1)) + 2) % (3), t2 = (((time / sf) % (time_M - time_m + 1)) + 1) % (3))
        {
          int tw = ((time / sf) % (time_M - time_m + 1));
          struct timeval start_section1, end_section1;
          gettimeofday(&start_section1, NULL);
          /* Begin section1 */

          bf0((float *)r50, (float *)r51, (float *)r52, (float *)r53, (float *)r82, (float *)r83, u_vec, v_vec, x_size, y_size, z_size, time, t0, x0_blk0_size, x_M + 2, x_m-2, y0_blk0_size, y_M+2, y_m-2, z_M, z_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf("\n BF0 - 1 IS OVER");

          /*==============================================*/
          bf1(damp_vec, dt, epsilon_vec, (float *)r49, (float *)r50, (float *)r51, (float *)r52, (float *)r53, (float *)r82, (float *)r83, u_vec, v_vec, vp_vec, nnz_sp_source_mask_vec, sp_source_mask_vec, save_src_u_vec, save_src_v_vec, source_id_vec, source_mask_vec, x_size, y_size, z_size, time, t0, t1, t2, x0_blk0_size, x_M, x_m, y0_blk0_size, y_M, y_m , z_M, z_m, sp_zi_m, nthreads, xb, yb, xb_size, yb_size, tw);
          //printf("\n BF1 - 1 IS OVER");

          /* End section1 */
          gettimeofday(&end_section1, NULL);
          timers->section1 += (double)(end_section1.tv_sec - start_section1.tv_sec) + (double)(end_section1.tv_usec - start_section1.tv_usec) / 1000000;
        }
      }
    }
  }

  free(r53);
  free(r52);
  free(r51);
  free(r50);
  free(r49);
  free(r82);
  free(r83);
  return 0;
}

void bf0(float *restrict r50_vec, float *restrict r51_vec, float *restrict r52_vec, float *restrict r53_vec, float *restrict r82_vec, float *restrict r83_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, const int x_size, const int y_size, const int z_size, const int time, const int t0, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, const int tw)
{
  float (*restrict r50)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r50_vec;
  float (*restrict r51)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r51_vec;
  float (*restrict r52)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r52_vec;
  float (*restrict r53)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r53_vec;
  float (*restrict r82)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r82_vec;
  float (*restrict r83)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r83_vec;
  float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
  float(*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__((aligned(64))) = (float(*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]])v_vec->data;

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
          //printf(" bf0 Timestep tw: %d, Updating x: %d \n", tw, x - time + 1);
          for (int y = y0_blk0; y <= min(min((y_M + time), (yb + yb_size - 1)), (y0_blk0 + y0_blk0_size - 1)); y++)
          {
#pragma omp simd aligned(u, v : 32)
            for (int z = z_m - 2; z <= z_M + 2; z += 1)
            {
              //printf(" bf0 Updating x: %d y: %d z: %d \n", x - time + 2, y - time + 2,  z + 2);
              r82[x - time + 2][y - time + 2][z + 2] = -(8.33333346e-3F*(u[t0][x - time + 6][y - time + 8][z + 8] - u[t0][x - time + 10][y - time + 8][z + 8]) + 6.66666677e-2F*(-u[t0][x - time + 7][y - time + 8][z + 8] + u[t0][x - time + 9][y - time + 8][z + 8]))*r51[x - time + 2][y - time + 2][z + 2]*r52[x - time + 2][y - time + 2][z + 2] - (8.33333346e-3F*(u[t0][x - time + 8][y - time + 6][z + 8] - u[t0][x - time + 8][y - time + 10][z + 8]) + 6.66666677e-2F*(-u[t0][x - time + 8][y - time + 7][z + 8] + u[t0][x - time + 8][y - time + 9][z + 8]))*r52[x - time + 2][y - time + 2][z + 2]*r53[x - time + 2][y - time + 2][z + 2] - (8.33333346e-3F*(u[t0][x - time + 8][y - time + 8][z + 6] - u[t0][x - time + 8][y - time + 8][z + 10]) + 6.66666677e-2F*(-u[t0][x - time + 8][y - time + 8][z + 7] + u[t0][x - time + 8][y - time + 8][z + 9]))*r50[x - time + 2][y - time + 2][z + 2];
              r83[x - time + 2][y - time + 2][z + 2] = -(8.33333346e-3F*(v[t0][x - time + 6][y - time + 8][z + 8] - v[t0][x - time + 10][y - time + 8][z + 8]) + 6.66666677e-2F*(-v[t0][x - time + 7][y - time + 8][z + 8] + v[t0][x - time + 9][y - time + 8][z + 8]))*r51[x - time + 2][y - time + 2][z + 2]*r52[x - time + 2][y - time + 2][z + 2] - (8.33333346e-3F*(v[t0][x - time + 8][y - time + 6][z + 8] - v[t0][x - time + 8][y - time + 10][z + 8]) + 6.66666677e-2F*(-v[t0][x - time + 8][y - time + 7][z + 8] + v[t0][x - time + 8][y - time + 9][z + 8]))*r52[x - time + 2][y - time + 2][z + 2]*r53[x - time + 2][y - time + 2][z + 2] - (8.33333346e-3F*(v[t0][x - time + 8][y - time + 8][z + 6] - v[t0][x - time + 8][y - time + 8][z + 10]) + 6.66666677e-2F*(-v[t0][x - time + 8][y - time + 8][z + 7] + v[t0][x - time + 8][y - time + 8][z + 9]))*r50[x - time + 2][y - time + 2][z + 2];
              //printf("bf0 Timestep tw: %d, Updating x: %d y: %d value: %f \n", tw, x - time + 2, y - time + 2, v[t0][x - time + 9][y - time + 8][z + 8]);

            }
          }
        }
      }
    }
  }
}

void bf1(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r49_vec, float *restrict r50_vec, float *restrict r51_vec, float *restrict r52_vec, float *restrict r53_vec, float *restrict r82_vec, float *restrict r83_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict save_src_v_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, const int x_size, const int y_size, const int z_size, const int time, const int t0, const int t1, const int t2, const int x1_blk0_size, const int x_M, const int x_m, const int y1_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int sp_zi_m, const int nthreads, const int xb, const int yb, const int xb_size, const int yb_size, const int tw)
{
  float(*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[damp_vec->size[1]][damp_vec->size[2]])damp_vec->data;
  float(*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__((aligned(64))) = (float(*)[epsilon_vec->size[1]][epsilon_vec->size[2]])epsilon_vec->data;
  float (*restrict r49)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r49_vec;
  float (*restrict r50)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r50_vec;
  float (*restrict r51)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r51_vec;
  float (*restrict r52)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r52_vec;
  float (*restrict r53)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r53_vec;
  float (*restrict r82)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r82_vec;
  float (*restrict r83)[y_size + 2 + 2][z_size + 2 + 2] __attribute__ ((aligned (64))) = (float (*)[y_size + 2 + 2][z_size + 2 + 2]) r83_vec;
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

#pragma omp parallel num_threads(nthreads)
  {
#pragma omp for collapse(1) schedule(dynamic, 1)
    for (int x1_blk0 = max((x_m + time), xb - 0); x1_blk0 <= +min((x_M + time), (xb - 0 + xb_size)); x1_blk0 += x1_blk0_size)
    {
      //printf(" Change of inner x1_blk0 %d \n", x1_blk0);
      for (int y1_blk0 = max((y_m + time), yb - 0); y1_blk0 <= +min((y_M + time), (yb - 0 + yb_size)); y1_blk0 += y1_blk0_size)
      {
        for (int x = x1_blk0; x <= min(min((x_M + time), (xb - 0 + xb_size - 1)), (x1_blk0 + x1_blk0_size - 1)); x++)
        {
          //printf(" bf1 Timestep tw: %d, Updating x: %d \n", tw, x - time + 4);
          for (int y = y1_blk0; y <= min(min((y_M + time), (yb - 0 + yb_size - 1)), (y1_blk0 + y1_blk0_size - 1)); y++)
          {
//printf(" bf1 Timestep tw: %d, Updating x: %d y: %d \n", tw, x - time + 4, y - time + 4);
#pragma omp simd aligned(damp, epsilon, u, v, vp : 32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              //printf(" bf1 Updating x: %d y: %d z: %d \n", x - time + 4, y - time + 4,  z + 4);
              //printf(" bf1 Updating x: %d y: %d z: %d \n", x - time + 4, y - time + 4,  z + 4);
              float r93 = 1.0/dt;
              float r92 = 1.0/(dt*dt);
              float r91 = 6.66666677e-2F*(r50[x - time + 2][y - time + 2][z + 1]*r83[x - time + 2][y - time + 2][z + 1] - r50[x - time + 2][y - time + 2][z + 3]*r83[x - time + 2][y - time + 2][z + 3] + r51[x - time + 1][y - time + 2][z + 2]*r52[x - time + 1][y - time + 2][z + 2]*r83[x - time + 1][y - time + 2][z + 2] - r51[x - time + 3][y - time + 2][z + 2]*r52[x - time + 3][y - time + 2][z + 2]*r83[x - time + 3][y - time + 2][z + 2] + r52[x - time + 2][y - time + 1][z + 2]*r53[x - time + 2][y - time + 1][z + 2]*r83[x - time + 2][y - time + 1][z + 2] - r52[x - time + 2][y - time + 3][z + 2]*r53[x - time + 2][y - time + 3][z + 2]*r83[x - time + 2][y - time + 3][z + 2]);
              float r90 = 8.33333346e-3F*(-r50[x - time + 2][y - time + 2][z]*r83[x - time + 2][y - time + 2][z] + r50[x - time + 2][y - time + 2][z + 4]*r83[x - time + 2][y - time + 2][z + 4] - r51[x - time] [y - time + 2][z + 2]*r52[x - time] [y - time + 2][z + 2]*r83[x - time] [y - time + 2][z + 2] + r51[x - time + 4][y - time + 2][z + 2]*r52[x - time + 4][y - time + 2][z + 2]*r83[x - time + 4][y - time + 2][z + 2] - r52[x - time + 2][y - time][z + 2]*r53[x - time + 2][y - time][z + 2]*r83[x - time + 2][y - time][z + 2] + r52[x - time + 2][y - time + 4][z + 2]*r53[x - time + 2][y - time + 4][z + 2]*r83[x - time + 2][y - time + 4][z + 2]);
              float r89 = pow(vp[x - time + 8][y - time + 8][z + 8], -2);
              float r88 = 1.0/(r89*r92 + r93*damp[x - time + 1][y - time + 1][z + 1]);
              float r87 = 8.33333346e-3F*(r50[x - time + 2][y - time + 2][z]*r82[x - time + 2][y - time + 2][z] - r50[x - time + 2][y - time + 2][z + 4]*r82[x - time + 2][y - time + 2][z + 4] + r51[x - time] [y - time + 2][z + 2]*r52[x - time] [y - time + 2][z + 2]*r82[x - time] [y - time + 2][z + 2] - r51[x - time + 4][y - time + 2][z + 2]*r52[x - time + 4][y - time + 2][z + 2]*r82[x - time + 4][y - time + 2][z + 2] + r52[x - time + 2][y - time][z + 2]*r53[x - time + 2][y - time][z + 2]*r82[x - time + 2][y - time][z + 2] - r52[x - time + 2][y - time + 4][z + 2]*r53[x - time + 2][y - time + 4][z + 2]*r82[x - time + 2][y - time + 4][z + 2]) + 6.66666677e-2F*(-r50[x - time + 2][y - time + 2][z + 1]*r82[x - time + 2][y - time + 2][z + 1] + r50[x - time + 2][y - time + 2][z + 3]*r82[x - time + 2][y - time + 2][z + 3] - r51[x - time + 1][y - time + 2][z + 2]*r52[x - time + 1][y - time + 2][z + 2]*r82[x - time + 1][y - time + 2][z + 2] + r51[x - time + 3][y - time + 2][z + 2]*r52[x - time + 3][y - time + 2][z + 2]*r82[x - time + 3][y - time + 2][z + 2] - r52[x - time + 2][y - time + 1][z + 2]*r53[x - time + 2][y - time + 1][z + 2]*r82[x - time + 2][y - time + 1][z + 2] + r52[x - time + 2][y - time + 3][z + 2]*r53[x - time + 2][y - time + 3][z + 2]*r82[x - time + 2][y - time + 3][z + 2]) - 1.78571425e-5F*(u[t0][x - time + 4][y - time + 8][z + 8] + u[t0][x - time + 8][y - time + 4][z + 8] + u[t0][x - time + 8][y - time + 8][z + 4] + u[t0][x - time + 8][y - time + 8][z + 12] + u[t0][x - time + 8][y - time + 12][z + 8] + u[t0][x - time + 12][y - time + 8][z + 8]) + 2.53968248e-4F*(u[t0][x - time + 5][y - time + 8][z + 8] + u[t0][x - time + 8][y - time + 5][z + 8] + u[t0][x - time + 8][y - time + 8][z + 5] + u[t0][x - time + 8][y - time + 8][z + 11] + u[t0][x - time + 8][y - time + 11][z + 8] + u[t0][x - time + 11][y - time + 8][z + 8]) - 1.99999996e-3F*(u[t0][x - time + 6][y - time + 8][z + 8] + u[t0][x - time + 8][y - time + 6][z + 8] + u[t0][x - time + 8][y - time + 8][z + 6] + u[t0][x - time + 8][y - time + 8][z + 10] + u[t0][x - time + 8][y - time + 10][z + 8] + u[t0][x - time + 10][y - time + 8][z + 8]) + 1.59999996e-2F*(u[t0][x - time + 7][y - time + 8][z + 8] + u[t0][x - time + 8][y - time + 7][z + 8] + u[t0][x - time + 8][y - time + 8][z + 7] + u[t0][x - time + 8][y - time + 8][z + 9] + u[t0][x - time + 8][y - time + 9][z + 8] + u[t0][x - time + 9][y - time + 8][z + 8]) - 8.54166647e-2F*u[t0][x - time + 8][y - time + 8][z + 8];
              float r80 = r92*(-2.0F*u[t0][x - time + 8][y - time + 8][z + 8] + u[t1][x - time + 8][y - time + 8][z + 8]);
              float r81 = r92*(-2.0F*v[t0][x - time + 8][y - time + 8][z + 8] + v[t1][x - time + 8][y - time + 8][z + 8]);
              u[t2][x - time + 8][y - time + 8][z + 8] = r88*((-r80)*r89 + r87*(2*epsilon[x - time + 8][y - time + 8][z + 8] + 1) + r93*(damp[x - time + 1][y - time + 1][z + 1]*u[t0][x - time + 8][y - time + 8][z + 8]) + (r90 + r91)*r49[x - time + 2][y - time + 2][z + 2]);
              v[t2][x - time + 8][y - time + 8][z + 8] = r88*((-r81)*r89 + r87*r49[x - time + 2][y - time + 2][z + 2] + r90 + r91 + r93*(damp[x - time + 1][y - time + 1][z + 1]*v[t0][x - time + 8][y - time + 8][z + 8]));
            }
            //int sp_zi_M = nnz_sp_source_mask[x - time][y - time] - 1;
            for (int sp_zi = sp_zi_m; sp_zi <= nnz_sp_source_mask[x - time][y - time] - 1; sp_zi += 1)
            {
              int zind = sp_source_mask[x - time][y - time][sp_zi];
              float r0 = save_src_u[tw][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
              //#pragma omp atomic update
              u[t2][x - time + 8][y - time + 8][zind + 8] += r0;
              float r1 = save_src_v[tw][source_id[x - time][y - time][zind]] * source_mask[x - time][y - time][zind];
              //#pragma omp atomic update
              v[t2][x - time + 8][y - time + 8][zind + 8] += r1;
              printf("Source injection at time %d , at : x: %d, y: %d, %d, %f, %f \n", tw, x - time + 8, y - time + 8, zind + 8, r0, r1);
            }
          }
        }
      }
    }
  }
}
