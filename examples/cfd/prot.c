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

void bf0(const float h_x, const float h_y, const float h_z, struct dataobj *restrict u_vec, const int t0, const int t1, const int x0_blk0_M, const int x0_blk0_m, const int x0_blk0_size, const int y0_blk0_M, const int y0_blk0_m, const int y0_blk0_size, const int z_M, const int z_m, const int nthreads);

int Kernel(const float h_x, const float h_y, const float h_z, const float o_x, const float o_y, const float o_z, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers, const int x0_blk0_size, const int y0_blk0_size, const int nthreads, const int nthreads_nonaffine)
{
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  for (int time = time_m, t0 = (time)%(3), t1 = (time + 1)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 1)%(3))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    bf0(h_x,h_y,h_z,u_vec,t0,t1,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,x0_blk0_size,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,y0_blk0_size,z_M,z_m,nthreads);
    bf0(h_x,h_y,h_z,u_vec,t0,t1,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,x0_blk0_size,y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,(y_M - y_m + 1)%(y0_blk0_size),z_M,z_m,nthreads);
    bf0(h_x,h_y,h_z,u_vec,t0,t1,x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,(x_M - x_m + 1)%(x0_blk0_size),y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,y0_blk0_size,z_M,z_m,nthreads);
    bf0(h_x,h_y,h_z,u_vec,t0,t1,x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,(x_M - x_m + 1)%(x0_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,(y_M - y_m + 1)%(y0_blk0_size),z_M,z_m,nthreads);
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_src_M - p_src_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        int ii_src_0 = (int)(floor((-o_x + src_coords[p_src][0])/h_x));
        int ii_src_1 = (int)(floor((-o_y + src_coords[p_src][1])/h_y));
        int ii_src_2 = (int)(floor((-o_z + src_coords[p_src][2])/h_z));
        int ii_src_3 = (int)(floor((-o_z + src_coords[p_src][2])/h_z)) + 1;
        int ii_src_4 = (int)(floor((-o_y + src_coords[p_src][1])/h_y)) + 1;
        int ii_src_5 = (int)(floor((-o_x + src_coords[p_src][0])/h_x)) + 1;
        float px = (float)(-h_x*(int)(floor((-o_x + src_coords[p_src][0])/h_x)) - o_x + src_coords[p_src][0]);
        float py = (float)(-h_y*(int)(floor((-o_y + src_coords[p_src][1])/h_y)) - o_y + src_coords[p_src][1]);
        float pz = (float)(-h_z*(int)(floor((-o_z + src_coords[p_src][2])/h_z)) - o_z + src_coords[p_src][2]);
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r3 = 1.0e-4F*(-px*py*pz/(h_x*h_y*h_z) + px*py/(h_x*h_y) + px*pz/(h_x*h_z) - px/h_x + py*pz/(h_y*h_z) - py/h_y - pz/h_z + 1)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] += r3;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r4 = 1.0e-4F*(px*py*pz/(h_x*h_y*h_z) - px*pz/(h_x*h_z) - py*pz/(h_y*h_z) + pz/h_z)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] += r4;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r5 = 1.0e-4F*(px*py*pz/(h_x*h_y*h_z) - px*py/(h_x*h_y) - py*pz/(h_y*h_z) + py/h_y)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] += r5;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r6 = 1.0e-4F*(-px*py*pz/(h_x*h_y*h_z) + py*pz/(h_y*h_z))*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] += r6;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r7 = 1.0e-4F*(px*py*pz/(h_x*h_y*h_z) - px*py/(h_x*h_y) - px*pz/(h_x*h_z) + px/h_x)*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] += r7;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r8 = 1.0e-4F*(-px*py*pz/(h_x*h_y*h_z) + px*pz/(h_x*h_z))*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] += r8;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r9 = 1.0e-4F*(-px*py*pz/(h_x*h_y*h_z) + px*py/(h_x*h_y))*src[time][p_src];
          #pragma omp atomic update
          u[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] += r9;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r10 = 1.0e-4F*px*py*pz*src[time][p_src]/(h_x*h_y*h_z);
          #pragma omp atomic update
          u[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] += r10;
        }
      }
    }
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
  }
  return 0;
}

void bf0(const float h_x, const float h_y, const float h_z, struct dataobj *restrict u_vec, const int t0, const int t1, const int x0_blk0_M, const int x0_blk0_m, const int x0_blk0_size, const int y0_blk0_M, const int y0_blk0_m, const int y0_blk0_size, const int z_M, const int z_m, const int nthreads)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  if (x0_blk0_size == 0 || y0_blk0_size == 0)
  {
    return;
  }
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(2) schedule(dynamic,1)
    for (int x0_blk0 = x0_blk0_m; x0_blk0 <= x0_blk0_M; x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y0_blk0_m; y0_blk0 <= y0_blk0_M; y0_blk0 += y0_blk0_size)
      {
        for (int x = x0_blk0; x <= x0_blk0 + x0_blk0_size - 1; x += 1)
        {
          for (int y = y0_blk0; y <= y0_blk0 + y0_blk0_size - 1; y += 1)
          {
            #pragma omp simd aligned(u:32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              float r2 = 1.0/(h_z*h_z);
              float r1 = 1.0/(h_y*h_y);
              float r0 = 1.0/(h_x*h_x);
              u[t1][x + 8][y + 8][z + 8] = r0*(-1.78571429e-3F*(u[t0][x + 4][y + 8][z + 8] + u[t0][x + 12][y + 8][z + 8]) + 2.53968254e-2F*(u[t0][x + 5][y + 8][z + 8] + u[t0][x + 11][y + 8][z + 8]) - 2.0e-1F*(u[t0][x + 6][y + 8][z + 8] + u[t0][x + 10][y + 8][z + 8]) + 1.6F*(u[t0][x + 7][y + 8][z + 8] + u[t0][x + 9][y + 8][z + 8]) - 2.84722222F*u[t0][x + 8][y + 8][z + 8]) + r1*(-1.78571429e-3F*(u[t0][x + 8][y + 4][z + 8] + u[t0][x + 8][y + 12][z + 8]) + 2.53968254e-2F*(u[t0][x + 8][y + 5][z + 8] + u[t0][x + 8][y + 11][z + 8]) - 2.0e-1F*(u[t0][x + 8][y + 6][z + 8] + u[t0][x + 8][y + 10][z + 8]) + 1.6F*(u[t0][x + 8][y + 7][z + 8] + u[t0][x + 8][y + 9][z + 8]) - 2.84722222F*u[t0][x + 8][y + 8][z + 8]) + r2*(-1.78571429e-3F*(u[t0][x + 8][y + 8][z + 4] + u[t0][x + 8][y + 8][z + 12]) + 2.53968254e-2F*(u[t0][x + 8][y + 8][z + 5] + u[t0][x + 8][y + 8][z + 11]) - 2.0e-1F*(u[t0][x + 8][y + 8][z + 6] + u[t0][x + 8][y + 8][z + 10]) + 1.6F*(u[t0][x + 8][y + 8][z + 7] + u[t0][x + 8][y + 8][z + 9]) - 2.84722222F*u[t0][x + 8][y + 8][z + 8]);
            }
          }
        }
      }
    }
  }
}
