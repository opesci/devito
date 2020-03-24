#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

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
  double section2;
} ;

void bf0(const float dt, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int t0, const int t1, const int t2, const int x0_blk0_M, const int x0_blk0_m, int x0_blk0_size, const int y0_blk0_M, const int y0_blk0_m, int y0_blk0_size, const int z_M, const int z_m);

int Forward(const float dt, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers, int x0_blk0_size, int y0_blk0_size)
{
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  x0_blk0_size = 4;
  y0_blk0_size = 4;
  
  for (int time = time_m, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    printf("---  time=%d --- \n", time);
    printf("--- bf0  --- \n");

    bf0(dt,u_vec,vp_vec,t0,t1,t2,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,x0_blk0_size,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,y0_blk0_size,z_M,z_m);
    printf("--- bf1  --- \n");

    bf0(dt,u_vec,vp_vec,t0,t1,t2,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,x0_blk0_size,y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,(y_M - y_m + 1)%(y0_blk0_size),z_M,z_m);
    printf("--- bf2  --- \n");

    bf0(dt,u_vec,vp_vec,t0,t1,t2,x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,(x_M - x_m + 1)%(x0_blk0_size),y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,y0_blk0_size,z_M,z_m);
    printf("--- bf3  --- \n");

    bf0(dt,u_vec,vp_vec,t0,t1,t2,x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,(x_M - x_m + 1)%(x0_blk0_size),y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,(y_M - y_m + 1)%(y0_blk0_size),z_M,z_m);
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      int ii_src_0 = (int)(floor(-6.66667e-2*o_x + 6.66667e-2*src_coords[p_src][0]));
      int ii_src_1 = (int)(floor(-6.66667e-2*o_y + 6.66667e-2*src_coords[p_src][1]));
      int ii_src_2 = (int)(floor(-6.66667e-2*o_z + 6.66667e-2*src_coords[p_src][2]));
      int ii_src_3 = (int)(floor(-6.66667e-2*o_z + 6.66667e-2*src_coords[p_src][2])) + 1;
      int ii_src_4 = (int)(floor(-6.66667e-2*o_y + 6.66667e-2*src_coords[p_src][1])) + 1;
      int ii_src_5 = (int)(floor(-6.66667e-2*o_x + 6.66667e-2*src_coords[p_src][0])) + 1;
      float px = (float)(-o_x - 1.5e+1F*(int)(floor(-6.66667e-2F*o_x + 6.66667e-2F*src_coords[p_src][0])) + src_coords[p_src][0]);
      float py = (float)(-o_y - 1.5e+1F*(int)(floor(-6.66667e-2F*o_y + 6.66667e-2F*src_coords[p_src][1])) + src_coords[p_src][1]);
      float pz = (float)(-o_z - 1.5e+1F*(int)(floor(-6.66667e-2F*o_z + 6.66667e-2F*src_coords[p_src][2])) + src_coords[p_src][2]);
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
      {
        float r0 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py + 4.44445e-3F*px*pz - 6.66667e-2F*px + 4.44445e-3F*py*pz - 6.66667e-2F*py - 6.66667e-2F*pz + 1)*src[time][p_src];
        u[t1][ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] += r0;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
      {
        float r1 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*pz - 4.44445e-3F*py*pz + 6.66667e-2F*pz)*src[time][p_src];
        u[t1][ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] += r1;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r2 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*py*pz + 6.66667e-2F*py)*src[time][p_src];
        u[t1][ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] += r2;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r3 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4]*vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*py*pz)*src[time][p_src];
        u[t1][ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] += r3;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r4 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4]*vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*px*pz + 6.66667e-2F*px)*src[time][p_src];
        u[t1][ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] += r4;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r5 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4]*vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*pz)*src[time][p_src];
        u[t1][ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] += r5;
      }
      if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r6 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4]*vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py)*src[time][p_src];
        u[t1][ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] += r6;
      }
      if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r7 = 2.96296e-4F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4]*vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4])*src[time][p_src];
        u[t1][ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] += r7;
      }
    }
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec-start_section2.tv_sec)+(double)(end_section2.tv_usec-start_section2.tv_usec)/1000000;
  }
  return 0;
}

void bf0(const float dt, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int t0, const int t1, const int t2, const int x0_blk0_M, const int x0_blk0_m, int x0_blk0_size, const int y0_blk0_M, const int y0_blk0_m, int y0_blk0_size, const int z_M, const int z_m)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;
  for (int x0_blk0 = x0_blk0_m; x0_blk0 <= x0_blk0_M; x0_blk0 += x0_blk0_size)
  {
    for (int y0_blk0 = y0_blk0_m; y0_blk0 <= y0_blk0_M; y0_blk0 += y0_blk0_size)
    {
      for (int x = x0_blk0; x <= x0_blk0 + x0_blk0_size - 1; x += 1)
      {
        for (int y = y0_blk0; y <= y0_blk0 + y0_blk0_size - 1; y += 1)
        {
          printf("t1=%d , x0_blk0_m = %d, x = %d, y= %d \n", t1, x0_blk0_m, x, y );
          #pragma omp simd aligned(u,vp:32)
          for (int z = z_m; z <= z_M; z += 1)
          {
            u[t1][x + 4][y + 4][z + 4] = (vp[x + 4][y + 4][z + 4]*vp[x + 4][y + 4][z + 4])*((dt*dt)*(-3.70370379e-4F*(u[t0][x + 2][y + 4][z + 4] + u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6] + u[t0][x + 4][y + 6][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 5.92592607e-3F*(u[t0][x + 3][y + 4][z + 4] + u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5] + u[t0][x + 4][y + 5][z + 4] + u[t0][x + 5][y + 4][z + 4]) - 3.33333341e-2F*u[t0][x + 4][y + 4][z + 4]) + (2*u[t0][x + 4][y + 4][z + 4] - u[t2][x + 4][y + 4][z + 4])/((vp[x + 4][y + 4][z + 4]*vp[x + 4][y + 4][z + 4])));
          }
        }
      }
    }
  }
}
/* Backdoor edit at Mon Mar 16 16:02:36 2020*/ 
/* Backdoor edit at Mon Mar 16 16:08:07 2020*/ 
/* Backdoor edit at Mon Mar 16 16:08:22 2020*/ 
/* Backdoor edit at Mon Mar 16 16:09:42 2020*/ 
/* Backdoor edit at Mon Mar 16 16:17:28 2020*/ 
/* Backdoor edit at Mon Mar 16 16:22:01 2020*/ 
/* Backdoor edit at Mon Mar 16 16:24:05 2020*/ 
