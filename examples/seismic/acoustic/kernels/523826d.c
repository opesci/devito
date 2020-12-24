#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"

struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
  double section1;
} ;

void bf0(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int t0, const int t1, const int t2, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads);

int Kernel(struct dataobj *restrict block_sizes_vec, struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_u_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int sp_zi_m, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, const int nthreads_nonaffine, struct profiler * timers)
{
  int (*restrict block_sizes) __attribute__ ((aligned (64))) = (int (*)) block_sizes_vec->data;
  int (*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__ ((aligned (64))) = (int (*)[nnz_sp_source_mask_vec->size[1]]) nnz_sp_source_mask_vec->data;
  float (*restrict save_src_u)[save_src_u_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[save_src_u_vec->size[1]]) save_src_u_vec->data;
  int (*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[source_id_vec->size[1]][source_id_vec->size[2]]) source_id_vec->data;
  int (*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[source_mask_vec->size[1]][source_mask_vec->size[2]]) source_mask_vec->data;
  int (*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]]) sp_source_mask_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    bf0(damp_vec,dt,u_vec,vp_vec,t0,t1,t2,x0_blk0_size,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,y0_blk0_size,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads);
    bf0(damp_vec,dt,u_vec,vp_vec,t0,t1,t2,x0_blk0_size,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,(y_M - y_m + 1)%(y0_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads);
    bf0(damp_vec,dt,u_vec,vp_vec,t0,t1,t2,(x_M - x_m + 1)%(x0_blk0_size),x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,y0_blk0_size,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads);
    bf0(damp_vec,dt,u_vec,vp_vec,t0,t1,t2,(x_M - x_m + 1)%(x0_blk0_size),x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,(y_M - y_m + 1)%(y0_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads);
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
  }
  int y0_blk0_size = block_sizes[3];
  int x0_blk0_size = block_sizes[2];
  int yb_size = block_sizes[1];
  int xb_size = block_sizes[0];
  for (int time = time_m, t2 = (time + 1)%(3); time <= time_M; time += 1, t2 = (time + 1)%(3))
  {
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(x_M - x_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int x = x_m; x <= x_M; x += 1)
      {
        #pragma omp simd aligned(nnz_sp_source_mask,save_src_u,source_id,source_mask,sp_source_mask,u:32)
        for (int y = y_m; y <= y_M; y += 1)
        {
          int sp_zi_M = nnz_sp_source_mask[x][y] - 1;
          for (int sp_zi = sp_zi_m; sp_zi <= sp_zi_M; sp_zi += 1)
          {
            int zind = sp_source_mask[x][y][sp_zi];
            float r0 = save_src_u[time][source_id[x][y][zind]]*source_mask[x][y][zind];
            u[t2][x + 4][y + 4][zind + 4] += r0;
          }
        }
      }
    }
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
  }
  return 0;
}

void bf0(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int t0, const int t1, const int t2, const int x0_blk0_size, const int x_M, const int x_m, const int y0_blk0_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  if (x0_blk0_size == 0 || y0_blk0_size == 0)
  {
    return;
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(dynamic,1)
    for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
      {
        for (int x = x0_blk0; x <= x0_blk0 + x0_blk0_size - 1; x += 1)
        {
          for (int y = y0_blk0; y <= y0_blk0 + y0_blk0_size - 1; y += 1)
          {
            #pragma omp simd aligned(damp,u,vp:32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              float r8 = 1.0/dt;
              float r7 = 1.0/(dt*dt);
              float r6 = 1.0/(vp[x + 4][y + 4][z + 4]*vp[x + 4][y + 4][z + 4]);
              u[t2][x + 4][y + 4][z + 4] = (r6*(-r7*(-2.0F*u[t0][x + 4][y + 4][z + 4] + u[t1][x + 4][y + 4][z + 4])) + r8*(damp[x + 1][y + 1][z + 1]*u[t0][x + 4][y + 4][z + 4]) - 3.70370379e-4F*(u[t0][x + 2][y + 4][z + 4] + u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6] + u[t0][x + 4][y + 6][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 5.92592607e-3F*(u[t0][x + 3][y + 4][z + 4] + u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5] + u[t0][x + 4][y + 5][z + 4] + u[t0][x + 5][y + 4][z + 4]) - 3.33333341e-2F*u[t0][x + 4][y + 4][z + 4])/(r6*r7 + r8*damp[x + 1][y + 1][z + 1]);
            }
          }
        }
      }
    }
  }
}
