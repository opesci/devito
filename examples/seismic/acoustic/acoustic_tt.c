#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include <stdio.h>
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#define min(a, b) (((a) < (b)) ? (a) : (b))
#define max(a, b) (((a) > (b)) ? (a) : (b))

int *malloc3d_and_init_int_cont(int dim1, int dim2, int dim3, int value)
{

  //Save data for 3d array in a contiguous way.
  int *p = (int *)malloc(dim1 * dim2 * dim3 * sizeof(int));

  for (int t = 0; t < dim1; t++)
  {
    for (int i = 0; i < dim2; i++)
    {
      for (int j = 0; j < dim3; j++)
      {
        int id = t * dim2 * dim3 + i * dim3 + j;
        p[id] = value;
      }
    }
  }
  return p;
}

// Function to allocate memory for u[t][x][y]
int ***malloc_3d_int(int dim1, int dim2, int dim3)
{
  //printf("\nAllocating 3d...");
  int ***matrix3d;

  matrix3d = (int ***)malloc(dim1 * sizeof(int **));
  for (int idim1 = 0; idim1 < dim1; idim1++)
  {
    matrix3d[idim1] = (int **)malloc(dim2 * sizeof(int *));
    for (int idim2 = 0; idim2 < dim2; idim2++)
    {
      matrix3d[idim1][idim2] = (int *)malloc(dim3 * sizeof(int));
    }
  }
  //printf("...DONE\n");
  return matrix3d;
}

// Function to allocate memory for u[t][x][y]
void initialize3_int(int timestamps, int nrows, int ncols, int ***A_init, int value)
{
  // Initialize the matrix
  //printf("\nInit 3d...");
  for (int ti = 0; ti < timestamps; ti++)
  {
    for (int xi = 0; xi < nrows; xi++)
    {
      for (int yi = 0; yi < ncols; yi++)
      {
        A_init[ti][xi][yi] = value; //2 * xi - 1.4 * yi + 0.5 * fabs(xi - yi);
      }
    }
  }
  //printf("...DONE\n");
}

// Function to allocate memory for u[t][x][y]
void initialize2(int nrows, int ncols, float **A_init, float value)
{
  // Initialize the matrix
  //printf("\nInit 3d...");
  int xi = 0;
  int yi = 0;

  for (xi = 0; xi < nrows; xi += 1)
  {
    for (yi = 0; yi < ncols; yi += 1)
    {
      A_init[xi][yi] = value; //2 * xi - 1.4 * yi + 0.5 * fabs(xi - yi);
    }
  }
  //printf("...DONE\n");
}

int **malloc_2d_int(int dim1, int dim2)
{
  //  printf("\nAllocating 3d...");
  int **matrix2d;

  matrix2d = (int **)malloc(dim1 * sizeof(int *));

  for (int idim1 = 0; idim1 < dim1; idim1++)
  {
    matrix2d[idim1] = (int *)malloc(dim2 * sizeof(int));
  }
  //printf("...DONE\n");
  return matrix2d;
}

float **malloc_2d_float(int dim1, int dim2)
{
  //  printf("\nAllocating 3d...");
  float **matrix2d;

  matrix2d = (float **)malloc(dim1 * sizeof(float *));

  for (int idim1 = 0; idim1 < dim1; idim1++)
  {
    matrix2d[idim1] = (float *)malloc(dim2 * sizeof(float));
  }
  //printf("...DONE\n");
  return matrix2d;
}

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

void bf0(const float dt, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int t_blk, const int t_blk_size, int x0_blk0_size, const int y0_blk0_M, const int y0_blk0_m, int y0_blk0_size, const int z_M, const int z_m, int x_M, int x_m, int y_M, int y_m, int sf, int time_M, int time_m, int **sparse_source_mask_NNZ, int ***sparse_source_mask, int ***source_mask, int ***source_id, float **save_src);

int Forward(const float dt, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler *timers, int x0_blk0_size, int y0_blk0_size)
{
  printf("\n ---- Starting time-tiling kernel  ----\n");
  float(*restrict rec)[rec_vec->size[1]] __attribute__((aligned(64))) = (float(*)[rec_vec->size[1]])rec_vec->data;
  float(*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__((aligned(64))) = (float(*)[rec_coords_vec->size[1]])rec_coords_vec->data;
  float(*restrict src)[src_vec->size[1]] __attribute__((aligned(64))) = (float(*)[src_vec->size[1]])src_vec->data;
  float(*restrict src_coords)[src_coords_vec->size[1]] __attribute__((aligned(64))) = (float(*)[src_coords_vec->size[1]])src_coords_vec->data;
  float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
  float(*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[vp_vec->size[1]][vp_vec->size[2]])vp_vec->data;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /*Allocate for inspection data structures */

  int id = 0;

  int ***source_id; //Grid 1
  source_id = malloc_3d_int(u_vec->size[1], u_vec->size[2], u_vec->size[3]);
  initialize3_int(u_vec->size[1], u_vec->size[2], u_vec->size[3], source_id, 0);

  int ***source_mask; //Grid 1
  source_mask = malloc_3d_int(u_vec->size[1], u_vec->size[2], u_vec->size[3]);
  initialize3_int(u_vec->size[1], u_vec->size[2], u_vec->size[3], source_mask, 0);

  for (int time = time_m, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3); time <= time_M; time += 1, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3))
  {
    /* Begin section1 */
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      int ii_src_0 = (int)(floor(-6.66667e-2 * o_x + 6.66667e-2 * src_coords[p_src][0]));
      int ii_src_1 = (int)(floor(-6.66667e-2 * o_y + 6.66667e-2 * src_coords[p_src][1]));
      int ii_src_2 = (int)(floor(-6.66667e-2 * o_z + 6.66667e-2 * src_coords[p_src][2]));
      int ii_src_3 = (int)(floor(-6.66667e-2 * o_z + 6.66667e-2 * src_coords[p_src][2])) + 1;
      int ii_src_4 = (int)(floor(-6.66667e-2 * o_y + 6.66667e-2 * src_coords[p_src][1])) + 1;
      int ii_src_5 = (int)(floor(-6.66667e-2 * o_x + 6.66667e-2 * src_coords[p_src][0])) + 1;
      float px = (float)(-o_x - 1.5e+1F * (int)(floor(-6.66667e-2F * o_x + 6.66667e-2F * src_coords[p_src][0])) + src_coords[p_src][0]);
      float py = (float)(-o_y - 1.5e+1F * (int)(floor(-6.66667e-2F * o_y + 6.66667e-2F * src_coords[p_src][1])) + src_coords[p_src][1]);
      float pz = (float)(-o_z - 1.5e+1F * (int)(floor(-6.66667e-2F * o_z + 6.66667e-2F * src_coords[p_src][2])) + src_coords[p_src][2]);
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
      {
        if (source_mask[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] == 0)
        {
            //printf("\n In ");

          source_id[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] = id++;
          source_mask[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] = 1;
        }
      }
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
      {
        if (source_mask[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] == 0)
        {//printf("\n In ");
          source_id[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] = id++;
          source_mask[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] = 1;
        }
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        if (source_mask[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] == 0)
        {//printf("\n In ");
          source_id[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] = id++;
          source_mask[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] = 1;
        }
      }
      if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        if (source_mask[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] == 0)
        {//printf("\n In ");
          source_id[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] = id++;
          source_mask[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] = 1;
        }
      }
      if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        if (source_mask[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] == 0)
        {//printf("\n In ");
          source_id[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] = id++;
          source_mask[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] = 1;
        }
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        if (source_mask[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] == 0)
        {//printf("\n In ");
          source_id[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] = id++;
          source_mask[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] = 1;
        }
      }
      if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        if (source_mask[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] == 0)
        {//printf("\n In ");
          source_id[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] = id++;
          source_mask[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] = 1;
        }
      }
      if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        if (source_mask[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] == 0)
        {//printf("\n In ");
          source_id[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] = id++;
          source_mask[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] = 1;
        }
      }
    }
    /* End section1 */
  }


  //printf("\n Test source_mask--------------------------");
  for (int xi = 0; xi < u_vec->size[1]; xi++)
  {
    for (int yi = 0; yi < u_vec->size[2]; yi++)
    {
      for (int zi = 0; zi < u_vec->size[3]; zi++)
      {
        if (source_mask[xi][yi][zi] == 1)
        {
          //printf("\n src_mask is : %d, %d, %d, %d ", xi, yi, zi, source_mask[xi][yi][zi]);
          //printf("\n src_id is : %d, %d, %d, %d ", xi, yi, zi, source_id[xi][yi][zi]);

          //sparse_source_mask[xi][yi][spzi] = zi;
          //sparse_source_id[xi][yi][spzi] = source_id[xi][yi][zi];
          //sparse_source_mask_NNZ[xi][yi]++;

          //printf("\n src_mask is : [%d, %d, %d] = %d ", xi, yi, zi, source_mask[xi][yi][zi]);
          //printf("\n src_id is : [%d, %d, %d]", xi, yi, zi);

          //spzi++;
        }
      }
    }
  }
  printf("\n ---- Test source_mask--------------------------\n");


  int ***sparse_source_id; //Grid 1
  sparse_source_id = malloc_3d_int(u_vec->size[1], u_vec->size[2], id);
  initialize3_int(u_vec->size[1], u_vec->size[2], id, sparse_source_id, 0);

  int ***sparse_source_mask; //Grid 1
  sparse_source_mask = malloc_3d_int(u_vec->size[1], u_vec->size[2], id);
  initialize3_int(u_vec->size[1], u_vec->size[2], id, sparse_source_mask, 0);

  int nnz = 0;
  int spzi = 0;

  int **sparse_source_mask_NNZ;
  sparse_source_mask_NNZ = malloc_2d_int(u_vec->size[1], u_vec->size[2]);

  for (int xi = 0; xi < u_vec->size[1]; xi++)
  {
    for (int yi = 0; yi < u_vec->size[2]; yi++)
    {
      sparse_source_mask_NNZ[xi][yi] = 0;
      spzi = 0;
      for (int zi = 0; zi < u_vec->size[3]; zi++)
      {
        if (source_mask[xi][yi][zi] == 1)
        {
          //printf("\n src_mask is : %d, %d, %d, %d ", xi, yi, zi, source_mask[xi][yi][zi]);
          //printf("\n src_id is : %d, %d, %d, %d ", xi, yi, zi, source_id[xi][yi][zi]);

          sparse_source_mask[xi][yi][spzi] = zi;
          sparse_source_id[xi][yi][spzi] = source_id[xi][yi][zi];
          sparse_source_mask_NNZ[xi][yi]++;

          //printf("\n src_mask is : [%d, %d, %d] = %d ", xi, yi, spzi, sparse_source_mask[xi][yi][spzi]);
          //printf("\n src  _id is : [%d, %d, %d] = %d ", xi, yi, spzi, sparse_source_id[xi][yi][spzi]);

          spzi++;
        }
      }
    }
  }

  // Sparse structs are already built here

  float **save_src;
  save_src = malloc_2d_float(id, (time_M + 1) );
  initialize2(id, (time_M + 1), save_src, 0.0F);

  for (int time = time_m, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3); time <= time_M; time += 1, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3))
  {
    /* Begin section1 */
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      int ii_src_0 = (int)(floor(-6.66667e-2 * o_x + 6.66667e-2 * src_coords[p_src][0]));
      int ii_src_1 = (int)(floor(-6.66667e-2 * o_y + 6.66667e-2 * src_coords[p_src][1]));
      int ii_src_2 = (int)(floor(-6.66667e-2 * o_z + 6.66667e-2 * src_coords[p_src][2]));
      int ii_src_3 = (int)(floor(-6.66667e-2 * o_z + 6.66667e-2 * src_coords[p_src][2])) + 1;
      int ii_src_4 = (int)(floor(-6.66667e-2 * o_y + 6.66667e-2 * src_coords[p_src][1])) + 1;
      int ii_src_5 = (int)(floor(-6.66667e-2 * o_x + 6.66667e-2 * src_coords[p_src][0])) + 1;
      float px = (float)(-o_x - 1.5e+1F * (int)(floor(-6.66667e-2F * o_x + 6.66667e-2F * src_coords[p_src][0])) + src_coords[p_src][0]);
      float py = (float)(-o_y - 1.5e+1F * (int)(floor(-6.66667e-2F * o_y + 6.66667e-2F * src_coords[p_src][1])) + src_coords[p_src][1]);
      float pz = (float)(-o_z - 1.5e+1F * (int)(floor(-6.66667e-2F * o_z + 6.66667e-2F * src_coords[p_src][2])) + src_coords[p_src][2]);
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
      {
        float r0 = (dt * dt) * (vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] * vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4]) * (-2.96296e-4F * px * py * pz + 4.44445e-3F * px * py + 4.44445e-3F * px * pz - 6.66667e-2F * px + 4.44445e-3F * py * pz - 6.66667e-2F * py - 6.66667e-2F * pz + 1) * src[time][p_src];
        //u[t1][ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4] += r0;
        save_src[(source_id[ii_src_0 + 4][ii_src_1 + 4][ii_src_2 + 4])][time] += r0;
        //printf("\n src_addition is : [%d, %d, %d] = %f at time %d ", ii_src_0 + 4, ii_src_1 + 4, ii_src_2 + 4, r0, time);

      }
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
      {
        float r1 = (dt * dt) * (vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] * vp[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4]) * (2.96296e-4F * px * py * pz - 4.44445e-3F * px * pz - 4.44445e-3F * py * pz + 6.66667e-2F * pz) * src[time][p_src];
        save_src[(source_id[ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4])][time] += r1;
        //printf("\n src_addition is : [%d, %d, %d] = %f at time %d ", ii_src_0 + 4, ii_src_1 + 4, ii_src_3 + 4, r1, time);
        //u[t1][ii_src_0 + 4][ii_src_1 + 4][ii_src_3 + 4] += r1;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r2 = (dt * dt) * (vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] * vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4]) * (2.96296e-4F * px * py * pz - 4.44445e-3F * px * py - 4.44445e-3F * py * pz + 6.66667e-2F * py) * src[time][p_src];
        save_src[(source_id[ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4])][time] += r2;
        //u[t1][ii_src_0 + 4][ii_src_4 + 4][ii_src_2 + 4] += r2;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r3 = (dt * dt) * (vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] * vp[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4]) * (-2.96296e-4F * px * py * pz + 4.44445e-3F * py * pz) * src[time][p_src];
        save_src[(source_id[ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4])][time] += r3;
        //u[t1][ii_src_0 + 4][ii_src_4 + 4][ii_src_3 + 4] += r3;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r4 = (dt * dt) * (vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] * vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4]) * (2.96296e-4F * px * py * pz - 4.44445e-3F * px * py - 4.44445e-3F * px * pz + 6.66667e-2F * px) * src[time][p_src];
        save_src[(source_id[ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4])][time] += r4;
        //u[t1][ii_src_5 + 4][ii_src_1 + 4][ii_src_2 + 4] += r4;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r5 = (dt * dt) * (vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] * vp[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4]) * (-2.96296e-4F * px * py * pz + 4.44445e-3F * px * pz) * src[time][p_src];
        save_src[(source_id[ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4])][time] += r5;
        //u[t1][ii_src_5 + 4][ii_src_1 + 4][ii_src_3 + 4] += r5;
      }
      if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r6 = (dt * dt) * (vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] * vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4]) * (-2.96296e-4F * px * py * pz + 4.44445e-3F * px * py) * src[time][p_src];
        save_src[(source_id[ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4])][time] += r6;
        //u[t1][ii_src_5 + 4][ii_src_4 + 4][ii_src_2 + 4] += r6;
      }
      if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r7 = 2.96296e-4F * px * py * pz * (dt * dt) * (vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] * vp[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4]) * src[time][p_src];
        save_src[(source_id[ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4])][time] += r7;
        //u[t1][ii_src_5 + 4][ii_src_4 + 4][ii_src_3 + 4] += r7;
      }
    }
  }

  x0_blk0_size = 9;
  y0_blk0_size = 9;
  int sf = 4;
  int t_blk_size = 10204;

  //for (int time = time_m, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3); time <= time_M; time += 1, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3))
  for (int t_blk = time_m; t_blk <= sf * (time_M - time_m); t_blk += t_blk_size) // for each t block
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    //printf(" Change of time_block t_blk = %d --- \n", t_blk);
    //printf("--- bf0  --- \n");

    bf0(dt, u_vec, vp_vec, t_blk, t_blk_size, x0_blk0_size, y_M - (y_M - y_m + 1) % (y0_blk0_size), y_m, y0_blk0_size, z_M, z_m, x_M, x_m, y_M, y_m, sf, time_M, time_m, sparse_source_mask_NNZ, sparse_source_mask, source_mask, source_id, save_src);
    //printf("--- bf1  --- \n");

    //bf0(dt, u_vec, vp_vec, t_blk, t_blk_size, x_M - (x_M - x_m + 1) % (x0_blk0_size), x_m, x0_blk0_size, y_M, y_M - (y_M - y_m + 1) % (y0_blk0_size) + 1, (y_M - y_m + 1) % (y0_blk0_size), z_M, z_m, x_M, x_m, y_M, y_m, sf, time_M);
    //printf("--- bf2  --- \n");

    //bf0(dt, u_vec, vp_vec, t_blk, t_blk_size, x_M, x_M - (x_M - x_m + 1) % (x0_blk0_size) + 1, (x_M - x_m + 1) % (x0_blk0_size), y_M - (y_M - y_m + 1) % (y0_blk0_size), y_m, y0_blk0_size, z_M, z_m, x_M, x_m, y_M, y_m, sf, time_M);
    //printf("--- bf3  --- \n");

    //bf0(dt, u_vec, vp_vec, t_blk, t_blk_size, x_M, x_M - (x_M - x_m + 1) % (x0_blk0_size) + 1, (x_M - x_m + 1) % (x0_blk0_size), y_M, y_M - (y_M - y_m + 1) % (y0_blk0_size) + 1, (y_M - y_m + 1) % (y0_blk0_size), z_M, z_m, x_M, x_m, y_M, y_m, sf, time_M);
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec - start_section0.tv_sec) + (double)(end_section0.tv_usec - start_section0.tv_usec) / 1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);

    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec - start_section1.tv_sec) + (double)(end_section1.tv_usec - start_section1.tv_usec) / 1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec - start_section2.tv_sec) + (double)(end_section2.tv_usec - start_section2.tv_usec) / 1000000;
  }
  return 0;
}

void bf0(const float dt, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int t_blk, const int t_blk_size, int x0_blk0_size, const int y0_blk0_M, const int y0_blk0_m, int y0_blk0_size, const int z_M, const int z_m, int x_M, int x_m, int y_M, int y_m, int sf, int time_M, int time_m, int **sparse_source_mask_NNZ, int ***sparse_source_mask, int ***source_mask, int ***source_id, float **save_src)
{
  float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
  float(*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__((aligned(64))) = (float(*)[vp_vec->size[1]][vp_vec->size[2]])vp_vec->data;

  //printf("From x: %d to %d \n", x_m, (x_M + sf * (time_M - time_m)));
  //printf("From y: %d to %d \n", y_m, (y_M + sf * (time_M - time_m)));

  for (int x0_blk0 = x_m; x0_blk0 <= (x_M + sf * (time_M - time_m)); x0_blk0 += x0_blk0_size)
  {
    for (int y0_blk0 = y_m; y0_blk0 <= (y_M + sf * (time_M - time_m)); y0_blk0 += y0_blk0_size)
    {
      for (int time = t_blk, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3); time <= 1 + min(t_blk + t_blk_size, sf * (time_M - time_m)); time += sf, t0 = (time) % (3), t1 = (time + 1) % (3), t2 = (time + 2) % (3))
      {
        int tw = (( time / sf)%(time_M - time_m + 1));

        //printf("Timesteps = %d \n", time_M - time_m +1 );
        //printf("time is: %d \n", time);
        //printf("tw sources is: %d \n", tw);
        //printf("New T time= %d : %d \n", t_blk, min(t_blk + t_blk_size, sf * (time_M - time_m)));
        for (int x = max((x_m + time), x0_blk0); x <= min((x_M + time), (x0_blk0 + x0_blk0_size - 1)); x++)
        //for (int xi = max((x_m + titer2), xouter); xi < min((x_M + titer2), (xouter + x_blk_size)); xi++)
        {
          for (int y = max((y_m + time), y0_blk0); y <= min((y_M + time), (y0_blk0 + y0_blk0_size - 1)); y++)
          {
            //printf("time= %d , x = %d, y= %d \n", time, x - time, y - time);
#pragma omp simd aligned(u, vp : 32)
            for (int z = z_m; z <= z_M; z += 1)
            {
              u[t1][x - time + 4][y - time + 4][z + 4] = (vp[x - time + 4][y - time + 4][z + 4] * vp[x - time + 4][y - time + 4][z + 4]) * ((dt * dt) * (-3.70370379e-4F * (u[t0][x - time + 2][y - time + 4][z + 4] + u[t0][x - time + 4][y - time + 2][z + 4] + u[t0][x - time + 4][y - time + 4][z + 2] + u[t0][x - time + 4][y - time + 4][z + 6] + u[t0][x - time + 4][y - time + 6][z + 4] + u[t0][x - time + 6][y - time + 4][z + 4]) + 5.92592607e-3F * (u[t0][x - time + 3][y - time + 4][z + 4] + u[t0][x - time + 4][y - time + 3][z + 4] + u[t0][x - time + 4][y - time + 4][z + 3] + u[t0][x - time + 4][y - time + 4][z + 5] + u[t0][x - time + 4][y - time + 5][z + 4] + u[t0][x - time + 5][y - time + 4][z + 4]) - 3.33333341e-2F * u[t0][x - time + 4][y - time + 4][z + 4]) + (2 * u[t0][x - time + 4][y - time + 4][z + 4] - u[t2][x - time + 4][y - time + 4][z + 4]) / ((vp[x - time + 4][y - time + 4][z + 4] * vp[x - time + 4][y - time + 4][z + 4])));
            }
            //#pragma omp simd aligned(u : 32)
            for (int spzi = 0; spzi < sparse_source_mask_NNZ[x + 4 - time][y + 4 - time]; spzi++) // Inner block loop
            {
              //printf("spzi is %d \n", spzi);
              
              int zind = sparse_source_mask[x + 4 - time][y + 4 - time][spzi];
              //printf("\n zind is : %d", sparse_source_mask[x + 4 - time][y + 4 - time][spzi]);

              u[t1][x + 4 - time][y + 4 - time][zind] += source_mask[x + 4 - time][y + 4 - time][zind] * save_src[(source_id[x + 4 - time][y + 4 - time][zind])][ tw + 1 ];
              //printf("\n Inject source in grid[%d][%d][%d], src = %f, ", x + 4 - time, y + 4 - time, zind, save_src[(source_id[x + 4 - time][y + 4 - time][zind])][tw + 1]);
              //printf("\n update : %d, %d, %d in %d", x - titer2, yi - titer2, zi ,titer2);
            }
          }
        }
      }
    }
  }
  printf("\n ---- Ending time-tiling kernel  ----\n");
}
/* Backdoor edit at Mon Mar 16 16:02:36 2020*/
/* Backdoor edit at Mon Mar 16 16:08:07 2020*/
/* Backdoor edit at Mon Mar 16 16:08:22 2020*/
/* Backdoor edit at Mon Mar 16 16:09:42 2020*/
/* Backdoor edit at Mon Mar 16 16:17:28 2020*/
/* Backdoor edit at Mon Mar 16 16:22:01 2020*/
/* Backdoor edit at Mon Mar 16 16:24:05 2020*/
