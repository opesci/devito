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
  double section2;
} ;

void bf0(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r17_vec, float *restrict r18_vec, float *restrict r19_vec, float *restrict r20_vec, float *restrict r21_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x0_blk0_size, const int x_size, const int y0_blk0_size, const int y_size, const int z_size, const int t0, const int t1, const int t2, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, float * *restrict r61_vec, float * *restrict r62_vec);

int ForwardTTI(struct dataobj *restrict damp_vec, struct dataobj *restrict delta_vec, const float dt, struct dataobj *restrict epsilon_vec, const float o_x, const float o_y, const float o_z, struct dataobj *restrict phi_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict theta_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x0_blk0_size, const int x_M, const int x_m, const int x_size, const int y0_blk0_size, const int y_M, const int y_m, const int y_size, const int z_M, const int z_m, const int z_size, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers, const int nthreads, const int nthreads_nonaffine)
{
  float (*restrict delta)[delta_vec->size[1]][delta_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[delta_vec->size[1]][delta_vec->size[2]]) delta_vec->data;
  float (*restrict phi)[phi_vec->size[1]][phi_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[phi_vec->size[1]][phi_vec->size[2]]) phi_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict theta)[theta_vec->size[1]][theta_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[theta_vec->size[1]][theta_vec->size[2]]) theta_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]]) v_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  float (*r17)[y_size + 1][z_size + 1];
  posix_memalign((void**)&r17, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float (*r18)[y_size + 1][z_size + 1];
  posix_memalign((void**)&r18, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float (*r19)[y_size + 1][z_size + 1];
  posix_memalign((void**)&r19, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float (*r20)[y_size + 1][z_size + 1];
  posix_memalign((void**)&r20, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float (*r21)[y_size + 1][z_size + 1];
  posix_memalign((void**)&r21, 64, sizeof(float[x_size + 1][y_size + 1][z_size + 1]));
  float **r61;
  posix_memalign((void**)&r61, 64, sizeof(float*)*nthreads);
  float **r62;
  posix_memalign((void**)&r62, 64, sizeof(float*)*nthreads);
  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    posix_memalign((void**)&r61[tid], 64, sizeof(float[x0_blk0_size + 1][y0_blk0_size + 1][z_size + 1]));
    posix_memalign((void**)&r62[tid], 64, sizeof(float[x0_blk0_size + 1][y0_blk0_size + 1][z_size + 1]));
  }

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  struct timeval start_section0, end_section0;
  gettimeofday(&start_section0, NULL);
  /* Begin section0 */
  #pragma omp parallel num_threads(nthreads)
  {
    #pragma omp for collapse(1) schedule(static,1)
    for (int x = x_m - 1; x <= x_M; x += 1)
    {
      for (int y = y_m - 1; y <= y_M; y += 1)
      {
        #pragma omp simd aligned(delta,phi,theta:32)
        for (int z = z_m - 1; z <= z_M; z += 1)
        {
          r17[x + 1][y + 1][z + 1] = sqrt(2*delta[x + 4][y + 4][z + 4] + 1);
          r18[x + 1][y + 1][z + 1] = cos(theta[x + 4][y + 4][z + 4]);
          r19[x + 1][y + 1][z + 1] = sin(phi[x + 4][y + 4][z + 4]);
          r20[x + 1][y + 1][z + 1] = sin(theta[x + 4][y + 4][z + 4]);
          r21[x + 1][y + 1][z + 1] = cos(phi[x + 4][y + 4][z + 4]);
        }
      }
    }
  }
  /* End section0 */
  gettimeofday(&end_section0, NULL);
  timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
  for (int time = time_m, t1 = (time + 2)%(3), t0 = (time)%(3), t2 = (time + 1)%(3); time <= time_M; time += 1, t1 = (time + 2)%(3), t0 = (time)%(3), t2 = (time + 1)%(3))
  {
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    bf0(damp_vec,dt,epsilon_vec,(float *)r17,(float *)r18,(float *)r19,(float *)r20,(float *)r21,u_vec,v_vec,vp_vec,x0_blk0_size,x_size,y0_blk0_size,y_size,z_size,t0,t1,t2,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads,(float * *)r61,(float * *)r62);
    bf0(damp_vec,dt,epsilon_vec,(float *)r17,(float *)r18,(float *)r19,(float *)r20,(float *)r21,u_vec,v_vec,vp_vec,x0_blk0_size,x_size,(y_M - y_m + 1)%(y0_blk0_size),y_size,z_size,t0,t1,t2,x_M - (x_M - x_m + 1)%(x0_blk0_size),x_m,y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads,(float * *)r61,(float * *)r62);
    bf0(damp_vec,dt,epsilon_vec,(float *)r17,(float *)r18,(float *)r19,(float *)r20,(float *)r21,u_vec,v_vec,vp_vec,(x_M - x_m + 1)%(x0_blk0_size),x_size,y0_blk0_size,y_size,z_size,t0,t1,t2,x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,y_M - (y_M - y_m + 1)%(y0_blk0_size),y_m,z_M,z_m,nthreads,(float * *)r61,(float * *)r62);
    bf0(damp_vec,dt,epsilon_vec,(float *)r17,(float *)r18,(float *)r19,(float *)r20,(float *)r21,u_vec,v_vec,vp_vec,(x_M - x_m + 1)%(x0_blk0_size),x_size,(y_M - y_m + 1)%(y0_blk0_size),y_size,z_size,t0,t1,t2,x_M,x_M - (x_M - x_m + 1)%(x0_blk0_size) + 1,y_M,y_M - (y_M - y_m + 1)%(y0_blk0_size) + 1,z_M,z_m,nthreads,(float * *)r61,(float * *)r62);
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    /* Begin section2 */
    #pragma omp parallel num_threads(nthreads_nonaffine)
    {
      int chunk_size = (int)(fmax(1, (1.0F/3.0F)*(p_src_M - p_src_m + 1)/nthreads_nonaffine));
      #pragma omp for collapse(1) schedule(dynamic,chunk_size)
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        float posx = -o_x + src_coords[p_src][0];
        float posy = -o_y + src_coords[p_src][1];
        float posz = -o_z + src_coords[p_src][2];
        int ii_src_0 = (int)(floor(1.0e-1*posx));
        int ii_src_1 = (int)(floor(1.0e-1*posy));
        int ii_src_2 = (int)(floor(1.0e-1*posz));
        int ii_src_3 = (int)(floor(1.0e-1*posz)) + 1;
        int ii_src_4 = (int)(floor(1.0e-1*posy)) + 1;
        int ii_src_5 = (int)(floor(1.0e-1*posx)) + 1;
        float px = (float)(posx - 1.0e+1F*(int)(floor(1.0e-1F*posx)));
        float py = (float)(posy - 1.0e+1F*(int)(floor(1.0e-1F*posy)));
        float pz = (float)(posz - 1.0e+1F*(int)(floor(1.0e-1F*posz)));
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r22 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] += r22;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r23 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] += r23;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r24 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] += r24;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r25 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4]*vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4])*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] += r25;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r26 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4]*vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] += r26;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r27 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4]*vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] += r27;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r28 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4]*vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] += r28;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r29 = 1.0e-3F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4]*vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4])*src[time][p_src];
          #pragma omp atomic update
          u[t2][ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] += r29;
        }
        posx = -o_x + src_coords[p_src][0];
        posy = -o_y + src_coords[p_src][1];
        posz = -o_z + src_coords[p_src][2];
        ii_src_0 = (int)(floor(1.0e-1*posx));
        ii_src_1 = (int)(floor(1.0e-1*posy));
        ii_src_2 = (int)(floor(1.0e-1*posz));
        ii_src_3 = (int)(floor(1.0e-1*posz)) + 1;
        ii_src_4 = (int)(floor(1.0e-1*posy)) + 1;
        ii_src_5 = (int)(floor(1.0e-1*posx)) + 1;
        px = (float)(posx - 1.0e+1F*(int)(floor(1.0e-1F*posx)));
        py = (float)(posy - 1.0e+1F*(int)(floor(1.0e-1F*posy)));
        pz = (float)(posz - 1.0e+1F*(int)(floor(1.0e-1F*posz)));
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r30 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py + 1.0e-2F*px*pz - 1.0e-1F*px + 1.0e-2F*py*pz - 1.0e-1F*py - 1.0e-1F*pz + 1)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] += r30;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r31 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4]*vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4])*(1.0e-3F*px*py*pz - 1.0e-2F*px*pz - 1.0e-2F*py*pz + 1.0e-1F*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] += r31;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r32 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4]*vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*py*pz + 1.0e-1F*py)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] += r32;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r33 = (dt*dt)*(vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4]*vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4])*(-1.0e-3F*px*py*pz + 1.0e-2F*py*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] += r33;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r34 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4]*vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4])*(1.0e-3F*px*py*pz - 1.0e-2F*px*py - 1.0e-2F*px*pz + 1.0e-1F*px)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] += r34;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r35 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4]*vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*pz)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] += r35;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r36 = (dt*dt)*(vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4]*vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4])*(-1.0e-3F*px*py*pz + 1.0e-2F*px*py)*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] += r36;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r37 = 1.0e-3F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4]*vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4])*src[time][p_src];
          #pragma omp atomic update
          v[t2][ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] += r37;
        }
      }
    }
    /* End section2 */
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec-start_section2.tv_sec)+(double)(end_section2.tv_usec-start_section2.tv_usec)/1000000;
  }

  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    free(r61[tid]);
    free(r62[tid]);
  }
  free(r17);
  free(r18);
  free(r19);
  free(r20);
  free(r21);
  free(r61);
  free(r62);
  return 0;
}

void bf0(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict epsilon_vec, float *restrict r17_vec, float *restrict r18_vec, float *restrict r19_vec, float *restrict r20_vec, float *restrict r21_vec, struct dataobj *restrict u_vec, struct dataobj *restrict v_vec, struct dataobj *restrict vp_vec, const int x0_blk0_size, const int x_size, const int y0_blk0_size, const int y_size, const int z_size, const int t0, const int t1, const int t2, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, float * *restrict r61_vec, float * *restrict r62_vec)
{
  float (*restrict damp)[damp_vec->size[1]][damp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]][damp_vec->size[2]]) damp_vec->data;
  float (*restrict epsilon)[epsilon_vec->size[1]][epsilon_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[epsilon_vec->size[1]][epsilon_vec->size[2]]) epsilon_vec->data;
  float (*restrict r17)[y_size + 1][z_size + 1] __attribute__ ((aligned (64))) = (float (*)[y_size + 1][z_size + 1]) r17_vec;
  float (*restrict r18)[y_size + 1][z_size + 1] __attribute__ ((aligned (64))) = (float (*)[y_size + 1][z_size + 1]) r18_vec;
  float (*restrict r19)[y_size + 1][z_size + 1] __attribute__ ((aligned (64))) = (float (*)[y_size + 1][z_size + 1]) r19_vec;
  float (*restrict r20)[y_size + 1][z_size + 1] __attribute__ ((aligned (64))) = (float (*)[y_size + 1][z_size + 1]) r20_vec;
  float (*restrict r21)[y_size + 1][z_size + 1] __attribute__ ((aligned (64))) = (float (*)[y_size + 1][z_size + 1]) r21_vec;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict v)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[v_vec->size[1]][v_vec->size[2]][v_vec->size[3]]) v_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;
  float **r61 = (float**) r61_vec;
  float **r62 = (float**) r62_vec;

  if (x0_blk0_size == 0)
  {
    return;
  }
  #pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    float (*restrict r48)[y0_blk0_size + 1][z_size + 1] __attribute__ ((aligned (64))) = (float (*)[y0_blk0_size + 1][z_size + 1]) r61[tid];
    float (*restrict r49)[y0_blk0_size + 1][z_size + 1] __attribute__ ((aligned (64))) = (float (*)[y0_blk0_size + 1][z_size + 1]) r62[tid];
    #pragma omp for collapse(1) schedule(dynamic,1)
    for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
      {
        for (int x = x0_blk0 - 1, xs = 0; x <= x0_blk0 + x0_blk0_size - 1; x += 1, xs += 1)
        {
          for (int y = y0_blk0 - 1, ys = 0; y <= y0_blk0 + y0_blk0_size - 1; y += 1, ys += 1)
          {
            #pragma omp simd aligned(u,v:32)
            for (int z = z_m - 1; z <= z_M; z += 1)
            {
              float r53 = -u[t0][x + 4][y + 4][z + 4];
              r48[xs][ys][z + 1] = 1.0e-1F*(-(r53 + u[t0][x + 4][y + 4][z + 5])*r18[x + 1][y + 1][z + 1] - (r53 + u[t0][x + 4][y + 5][z + 4])*r19[x + 1][y + 1][z + 1]*r20[x + 1][y + 1][z + 1] - (r53 + u[t0][x + 5][y + 4][z + 4])*r20[x + 1][y + 1][z + 1]*r21[x + 1][y + 1][z + 1]);
              float r54 = -v[t0][x + 4][y + 4][z + 4];
              r49[xs][ys][z + 1] = 1.0e-1F*(-(r54 + v[t0][x + 4][y + 4][z + 5])*r18[x + 1][y + 1][z + 1] - (r54 + v[t0][x + 4][y + 5][z + 4])*r19[x + 1][y + 1][z + 1]*r20[x + 1][y + 1][z + 1] - (r54 + v[t0][x + 5][y + 4][z + 4])*r20[x + 1][y + 1][z + 1]*r21[x + 1][y + 1][z + 1]);
            }
          }
        }
        for (int x = x0_blk0, xs = 0; x <= x0_blk0 + x0_blk0_size - 1; x += 1, xs += 1)
        {
          for (int y = y0_blk0, ys = 0; y <= y0_blk0 + y0_blk0_size - 1; y += 1, ys += 1)
          {
            #pragma omp simd aligned(damp,epsilon,u,v,vp:32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              float r60 = 1.0/dt;
              float r59 = 1.0/(dt*dt);
              float r58 = r18[x + 1][y + 1][z]*r49[xs + 1][ys + 1][z] - r18[x + 1][y + 1][z + 1]*r49[xs + 1][ys + 1][z + 1] + r19[x + 1][y][z + 1]*r20[x + 1][y][z + 1]*r49[xs + 1][ys][z + 1] - r19[x + 1][y + 1][z + 1]*r20[x + 1][y + 1][z + 1]*r49[xs + 1][ys + 1][z + 1] + r20[x][y + 1][z + 1]*r21[x][y + 1][z + 1]*r49[xs][ys + 1][z + 1] - r20[x + 1][y + 1][z + 1]*r21[x + 1][y + 1][z + 1]*r49[xs + 1][ys + 1][z + 1];
              float r57 = 1.0/(vp[x + 4][y + 4][z + 4]*vp[x + 4][y + 4][z + 4]);
              float r56 = 1.0e-1F*(-r18[x + 1][y + 1][z]*r48[xs + 1][ys + 1][z] + r18[x + 1][y + 1][z + 1]*r48[xs + 1][ys + 1][z + 1] - r19[x + 1][y][z + 1]*r20[x + 1][y][z + 1]*r48[xs + 1][ys][z + 1] + r19[x + 1][y + 1][z + 1]*r20[x + 1][y + 1][z + 1]*r48[xs + 1][ys + 1][z + 1] - r20[x][y + 1][z + 1]*r21[x][y + 1][z + 1]*r48[xs][ys + 1][z + 1] + r20[x + 1][y + 1][z + 1]*r21[x + 1][y + 1][z + 1]*r48[xs + 1][ys + 1][z + 1]) - 8.33333315e-4F*(u[t0][x + 2][y + 4][z + 4] + u[t0][x + 4][y + 2][z + 4] + u[t0][x + 4][y + 4][z + 2] + u[t0][x + 4][y + 4][z + 6] + u[t0][x + 4][y + 6][z + 4] + u[t0][x + 6][y + 4][z + 4]) + 1.3333333e-2F*(u[t0][x + 3][y + 4][z + 4] + u[t0][x + 4][y + 3][z + 4] + u[t0][x + 4][y + 4][z + 3] + u[t0][x + 4][y + 4][z + 5] + u[t0][x + 4][y + 5][z + 4] + u[t0][x + 5][y + 4][z + 4]) - 7.49999983e-2F*u[t0][x + 4][y + 4][z + 4];
              float r55 = 1.0/(r57*r59 + r60*damp[x + 1][y + 1][z + 1]);
              float r46 = r59*(-2.0F*u[t0][x + 4][y + 4][z + 4] + u[t1][x + 4][y + 4][z + 4]);
              float r47 = r59*(-2.0F*v[t0][x + 4][y + 4][z + 4] + v[t1][x + 4][y + 4][z + 4]);
              u[t2][x + 4][y + 4][z + 4] = r55*((-r46)*r57 + r56*(2*epsilon[x + 4][y + 4][z + 4] + 1) + 1.0e-1F*r58*r17[x + 1][y + 1][z + 1] + r60*(damp[x + 1][y + 1][z + 1]*u[t0][x + 4][y + 4][z + 4]));
              v[t2][x + 4][y + 4][z + 4] = r55*((-r47)*r57 + r56*r17[x + 1][y + 1][z + 1] + 1.0e-1F*r58 + r60*(damp[x + 1][y + 1][z + 1]*v[t0][x + 4][y + 4][z + 4]));
            }
          }
        }
      }
    }

  }
}
/* Backdoor edit at Tue Sep 22 18:44:51 2020*/ 
