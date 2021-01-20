#define _POSIX_C_SOURCE 200809L
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

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
} ;


int Kernel(struct dataobj *restrict block_sizes_vec, struct dataobj *restrict damp_vec, const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict nnz_sp_source_mask_vec, struct dataobj *restrict save_src_vec, struct dataobj *restrict source_id_vec, struct dataobj *restrict source_mask_vec, struct dataobj *restrict sp_source_mask_vec, struct dataobj *restrict usol_vec, struct dataobj *restrict vp_vec, const int sp_zi_m, const int time_M, const int time_m, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads_nonaffine, struct profiler * timers)
{
  int (*restrict block_sizes) __attribute__ ((aligned (64))) = (int (*)) block_sizes_vec->data;
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  int (*restrict nnz_sp_source_mask)[nnz_sp_source_mask_vec->size[1]] __attribute__ ((aligned (64))) = (int (*)[nnz_sp_source_mask_vec->size[1]]) nnz_sp_source_mask_vec->data;
  float (*restrict save_src)[save_src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[save_src_vec->size[1]]) save_src_vec->data;
  int (*restrict source_id)[source_id_vec->size[1]][source_id_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[source_id_vec->size[1]][source_id_vec->size[2]]) source_id_vec->data;
  float (*restrict source_mask)[source_mask_vec->size[1]][source_mask_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[source_mask_vec->size[1]][source_mask_vec->size[2]]) source_mask_vec->data;
  int (*restrict sp_source_mask)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]] __attribute__ ((aligned (64))) = (int (*)[sp_source_mask_vec->size[1]][sp_source_mask_vec->size[2]]) sp_source_mask_vec->data;
  float (*restrict usol)[usol_vec->size[1]][usol_vec->size[2]][usol_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[usol_vec->size[1]][usol_vec->size[2]][usol_vec->size[3]]) usol_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  int xb_size = block_sizes[0];
  int yb_size = block_sizes[1];
  int x0_blk0_size = block_sizes[2];
  int y0_blk0_size = block_sizes[3];
  for (int time = time_m, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 2)%(3), t2 = (time + 1)%(3))
  {
    /* Begin section0 */
    START_TIMER(section0)
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(x_M - x_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int x = x_m; x <= x_M; x += 1)
      {
        #pragma omp simd aligned(damp,nnz_sp_source_mask,save_src,source_id,source_mask,sp_source_mask,usol,vp:32)
        for (int y = y_m; y <= y_M; y += 1)
        {
          for (int z = z_m; z <= z_M; z += 1)
          {
            float r9 = -2.5F*usol[t0][x + 4][y + 4][z + 4];
            float r8 = 1.0/dt;
            float r7 = 1.0/(dt*dt);
            float r6 = 1.0/(vp[x + 4][y + 4][z + 4]*vp[x + 4][y + 4][z + 4]);
            usol[t2][x + 4][y + 4][z + 4] = (r6*(-r7*(-2.0F*usol[t0][x + 4][y + 4][z + 4] + usol[t1][x + 4][y + 4][z + 4])) + r8*(damp[x + 1][y + 1][z + 1]*usol[t0][x + 4][y + 4][z + 4]) + (r9 - 8.33333333e-2F*(usol[t0][x + 4][y + 4][z + 2] + usol[t0][x + 4][y + 4][z + 6]) + 1.33333333F*(usol[t0][x + 4][y + 4][z + 3] + usol[t0][x + 4][y + 4][z + 5]))/((h_z*h_z)) + (r9 - 8.33333333e-2F*(usol[t0][x + 4][y + 2][z + 4] + usol[t0][x + 4][y + 6][z + 4]) + 1.33333333F*(usol[t0][x + 4][y + 3][z + 4] + usol[t0][x + 4][y + 5][z + 4]))/((h_y*h_y)) + (r9 - 8.33333333e-2F*(usol[t0][x + 2][y + 4][z + 4] + usol[t0][x + 6][y + 4][z + 4]) + 1.33333333F*(usol[t0][x + 3][y + 4][z + 4] + usol[t0][x + 5][y + 4][z + 4]))/((h_x*h_x)))/(r6*r7 + r8*damp[x + 1][y + 1][z + 1]);
          }
          int sp_zi_M = nnz_sp_source_mask[x][y] - 1;
          for (int sp_zi = sp_zi_m; sp_zi <= sp_zi_M; sp_zi += 1)
          {
            int zind = sp_source_mask[x][y][sp_zi];
            float r0 = save_src[time][source_id[x][y][zind]]*source_mask[x][y][zind];
            usol[t2][x + 4][y + 4][zind + 4] += r0;
          }
        }
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */
  }

  return 0;
}
/* Backdoor edit at Wed Jan 20 16:52:34 2021*/ 
