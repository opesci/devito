#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"

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
  //  printf("\nAllocating 3d...");
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
};

void bf0(const float h_x, const float h_y, const float h_z, struct dataobj *restrict u_vec, const int x0_blk0_M, const int x0_blk0_m, const int x0_blk0_size, const int y0_blk0_M, const int y0_blk0_m, const int y0_blk0_size, const int z_M, const int z_m, const int nthreads, int **sparse_source_mask_NNZ, int ***sparse_source_mask, int ***source_mask, int ***source_id, float **save_src, int sf, int time_M, int t_blk, int t_blk_size);

int Kernel(const float h_x, const float h_y, const float h_z, const float o_x, const float o_y, const float o_z, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler *timers, const int x0_blk0_size, const int y0_blk0_size, const int nthreads, const int nthreads_nonaffine)
{
  float(*restrict src)[src_vec->size[1]] __attribute__((aligned(64))) = (float(*)[src_vec->size[1]])src_vec->data;
  float(*restrict src_coords)[src_coords_vec->size[1]] __attribute__((aligned(64))) = (float(*)[src_coords_vec->size[1]])src_coords_vec->data;
  float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
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

  // Inspection start
  for (int time = time_m; time <= time_M; time += 1)
  {
    {
      int chunk_size = (int)(fmax(1, (1.0F / 3.0F) * (p_src_M - p_src_m + 1) / nthreads_nonaffine));
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        int ii_src_0 = (int)(floor((-o_x + src_coords[p_src][0]) / h_x));
        int ii_src_1 = (int)(floor((-o_y + src_coords[p_src][1]) / h_y));
        int ii_src_2 = (int)(floor((-o_z + src_coords[p_src][2]) / h_z));
        int ii_src_3 = (int)(floor((-o_z + src_coords[p_src][2]) / h_z)) + 1;
        int ii_src_4 = (int)(floor((-o_y + src_coords[p_src][1]) / h_y)) + 1;
        int ii_src_5 = (int)(floor((-o_x + src_coords[p_src][0]) / h_x)) + 1;

        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          if (source_mask[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] == 0)
          {
            source_id[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] = id++;
            source_mask[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] = 1;
          }
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          if (source_mask[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] == 0)
          {
            source_id[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] = id++;
            source_mask[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] = 1;
          }
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          if (source_mask[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] == 0)
          {
            source_id[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] = id++;
            source_mask[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] = 1;
          }
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          if (source_mask[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] == 0)
          {
            source_id[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] = id++;
            source_mask[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] = 1;
          }
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          if (source_mask[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] == 0)
          {
            source_id[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] = id++;
            source_mask[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] = 1;
          }
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          if (source_mask[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] == 0)
          {
            source_id[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] = id++;
            source_mask[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] = 1;
          }
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          if (source_mask[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] == 0)
          {
            source_id[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] = id++;
            source_mask[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] = 1;
          }
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          if (source_mask[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] == 0)
          {
            source_id[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] = id++;
            source_mask[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] = 1;
          }
        }
      }
    }
  }

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

  for (int xi = x_m; xi < x_M; xi++)
  {
    for (int yi = y_m; yi < y_M; yi++)
    {
      sparse_source_mask_NNZ[xi][yi] = 0;
      spzi = 0;
      for (int zi = z_m; zi < z_M; zi++)
      {
        if (source_mask[xi][yi][zi] == 1)
        {
          //printf("\n src_mask is : %d, %d, %d, %d ", xi, yi, zi, source_mask[xi][yi][zi]);
          //printf("\n src_id is : %d, %d, %d, %d ", xi, yi, zi, source_id[xi][yi][zi]);

          sparse_source_mask[xi][yi][spzi] = zi;
          sparse_source_id[xi][yi][spzi] = source_id[xi][yi][zi];
          sparse_source_mask_NNZ[xi][yi]++;

          //printf("\n src_mask is : %d, %d, %d, %d ", xi, yi, spzi, sparse_source_mask[xi][yi][spzi]);
          //printf("\n src_id is : %d, %d, %d, %d ", xi, yi, spzi, sparse_source_id[xi][yi][spzi]);

          spzi++;
        }
      }
    }
  }

  // Sparse structs are already built here

  float **save_src;
  save_src = malloc_2d_float(id, time_M);
  initialize2(id, time_M, save_src, 0.0F);

  for (int time = time_m; time <= time_M; time += 1)
  {
    /* Begin section1 */
    {
      int chunk_size = (int)(fmax(1, (1.0F / 3.0F) * (p_src_M - p_src_m + 1) / nthreads_nonaffine));
      for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
      {
        int ii_src_0 = (int)(floor((-o_x + src_coords[p_src][0]) / h_x));
        int ii_src_1 = (int)(floor((-o_y + src_coords[p_src][1]) / h_y));
        int ii_src_2 = (int)(floor((-o_z + src_coords[p_src][2]) / h_z));
        int ii_src_3 = (int)(floor((-o_z + src_coords[p_src][2]) / h_z)) + 1;
        int ii_src_4 = (int)(floor((-o_y + src_coords[p_src][1]) / h_y)) + 1;
        int ii_src_5 = (int)(floor((-o_x + src_coords[p_src][0]) / h_x)) + 1;
        float px = (float)(-h_x * (int)(floor((-o_x + src_coords[p_src][0]) / h_x)) - o_x + src_coords[p_src][0]);
        float py = (float)(-h_y * (int)(floor((-o_y + src_coords[p_src][1]) / h_y)) - o_y + src_coords[p_src][1]);
        float pz = (float)(-h_z * (int)(floor((-o_z + src_coords[p_src][2]) / h_z)) - o_z + src_coords[p_src][2]);
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
        {
          float r3 = 1.0e-4F * (-px * py * pz / (h_x * h_y * h_z) + px * py / (h_x * h_y) + px * pz / (h_x * h_z) - px / h_x + py * pz / (h_y * h_z) - py / h_y - pz / h_z + 1) * src[time][p_src];
          //u[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] += r3;
          save_src[(source_id[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8])][time] += r3;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
        {
          float r4 = 1.0e-4F * (px * py * pz / (h_x * h_y * h_z) - px * pz / (h_x * h_z) - py * pz / (h_y * h_z) + pz / h_z) * src[time][p_src];
          //u[t1][ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] += r4;
          save_src[(source_id[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8])][time] += r4;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r5 = 1.0e-4F * (px * py * pz / (h_x * h_y * h_z) - px * py / (h_x * h_y) - py * pz / (h_y * h_z) + py / h_y) * src[time][p_src];
          //u[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] += r5;
          save_src[(source_id[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8])][time] += r5;
        }
        if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
        {
          float r6 = 1.0e-4F * (-px * py * pz / (h_x * h_y * h_z) + py * pz / (h_y * h_z)) * src[time][p_src];
          //u[t1][ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] += r6;
          save_src[(source_id[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8])][time] += r6;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r7 = 1.0e-4F * (px * py * pz / (h_x * h_y * h_z) - px * py / (h_x * h_y) - px * pz / (h_x * h_z) + px / h_x) * src[time][p_src];
          //u[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] += r7;
          save_src[(source_id[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8])][time] += r7;
        }
        if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r8 = 1.0e-4F * (-px * py * pz / (h_x * h_y * h_z) + px * pz / (h_x * h_z)) * src[time][p_src];
          //u[t1][ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] += r8;
          save_src[(source_id[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8])][time] += r8;
        }
        if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r9 = 1.0e-4F * (-px * py * pz / (h_x * h_y * h_z) + px * py / (h_x * h_y)) * src[time][p_src];
          //u[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] += r9;
          save_src[(source_id[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8])][time] += r9;
        }
        if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
        {
          float r10 = 1.0e-4F * px * py * pz * src[time][p_src] / (h_x * h_y * h_z);
          //u[t1][ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] += r10;
          save_src[(source_id[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8])][time] += r10;
        }
      }
    }
    /* End section1 */
  }

  int sf = 4;
  int t_blk_size = 16;

  for (int t_blk = time_m; t_blk < sf * time_M; t_blk += t_blk_size) // for each t block
  {
    //for (int time = time_m, t0 = (time) % (3), t1 = (time + 1) % (3); time <= time_M; time += 1, t0 = (time) % (3), t1 = (time + 1) % (3))
    //{
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    bf0(h_x, h_y, h_z, u_vec, x_M - (x_M - x_m + 1) % (x0_blk0_size), x_m, x0_blk0_size, y_M - (y_M - y_m + 1) % (y0_blk0_size), y_m, y0_blk0_size, z_M, z_m, nthreads, sparse_source_mask_NNZ, sparse_source_mask, source_mask, source_id, save_src, sf, time_M, t_blk, t_blk_size);
    bf0(h_x, h_y, h_z, u_vec, x_M - (x_M - x_m + 1) % (x0_blk0_size), x_m, x0_blk0_size, y_M, y_M - (y_M - y_m + 1) % (y0_blk0_size) + 1, (y_M - y_m + 1) % (y0_blk0_size), z_M, z_m, nthreads, sparse_source_mask_NNZ, sparse_source_mask, source_mask, source_id, save_src, sf, time_M, t_blk, t_blk_size);
    bf0(h_x, h_y, h_z, u_vec, x_M, x_M - (x_M - x_m + 1) % (x0_blk0_size) + 1, (x_M - x_m + 1) % (x0_blk0_size), y_M - (y_M - y_m + 1) % (y0_blk0_size), y_m, y0_blk0_size, z_M, z_m, nthreads, sparse_source_mask_NNZ, sparse_source_mask, source_mask, source_id, save_src, sf, time_M, t_blk, t_blk_size);
    bf0(h_x, h_y, h_z, u_vec, x_M, x_M - (x_M - x_m + 1) % (x0_blk0_size) + 1, (x_M - x_m + 1) % (x0_blk0_size), y_M, y_M - (y_M - y_m + 1) % (y0_blk0_size) + 1, (y_M - y_m + 1) % (y0_blk0_size), z_M, z_m, nthreads, sparse_source_mask_NNZ, sparse_source_mask, source_mask, source_id, save_src, sf, time_M, t_blk, t_blk_size);
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec - start_section0.tv_sec) + (double)(end_section0.tv_usec - start_section0.tv_usec) / 1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    /*  */
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec - start_section1.tv_sec) + (double)(end_section1.tv_usec - start_section1.tv_usec) / 1000000;
  }
  return 0;
}

void bf0(const float h_x, const float h_y, const float h_z, struct dataobj *restrict u_vec, const int x0_blk0_M, const int x0_blk0_m, const int x0_blk0_size, const int y0_blk0_M, const int y0_blk0_m, const int y0_blk0_size, const int z_M, const int z_m, const int nthreads, int **sparse_source_mask_NNZ, int ***sparse_source_mask, int ***source_mask, int ***source_id, float **save_src, int sf, int time_M, int t_blk, int t_blk_size)
{
  float(*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__((aligned(64))) = (float(*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]])u_vec->data;
  if (x0_blk0_size == 0)
  {
    return;
  }
  int zind = 0;
  //#pragma omp parallel num_threads(nthreads)
  {
#pragma omp for collapse(1) schedule(dynamic, 1)
    for (int x0_blk0 = x0_blk0_m; x0_blk0 <= (x0_blk0_M + sf * time_M); x0_blk0 += x0_blk0_size)
    {
      for (int y0_blk0 = y0_blk0_m; y0_blk0 <= (y0_blk0_M + sf * time_M); y0_blk0 += y0_blk0_size)
      {
        for (int time = t_blk, t0 = (time) % (3), t1 = (time + 1) % (3); time <= min(t_blk + t_blk_size, sf * time_M); time += sf, t0 = (time) % (3), t1 = (time + 1) % (3))
        {
          /* code */

          for (int x = x0_blk0; x <= x0_blk0 + x0_blk0_size - 1; x += 1)
          {
            for (int y = y0_blk0; y <= y0_blk0 + y0_blk0_size - 1; y += 1)
            {
#pragma omp simd aligned(u : 32)
              for (int z = z_m; z <= z_M; z += 1)
              {
                float r2 = 1.0 / (h_z * h_z);
                float r1 = 1.0 / (h_y * h_y);
                float r0 = 1.0 / (h_x * h_x);
                u[t1][x + 8 - time][y + 8 - time][z + 8] = r0 * (-1.78571429e-3F * (u[t0][x + 4 - time][y + 8 - time][z + 8] + u[t0][x + 12 - time][y + 8 - time][z + 8]) + 2.53968254e-2F * (u[t0][x + 5 - time][y + 8 - time][z + 8] + u[t0][x + 11 - time][y + 8 - time][z + 8]) - 2.0e-1F * (u[t0][x + 6 - time][y + 8 - time][z + 8] + u[t0][x + 10 - time][y + 8 - time][z + 8]) + 1.6F * (u[t0][x + 7 - time][y + 8 - time][z + 8] + u[t0][x + 9 - time][y + 8 - time][z + 8]) - 2.84722222F * u[t0][x + 8 - time][y + 8 - time][z + 8]) + r1 * (-1.78571429e-3F * (u[t0][x + 8 - time][y + 4 - time][z + 8] + u[t0][x + 8 - time][y + 12 - time][z + 8]) + 2.53968254e-2F * (u[t0][x + 8 - time][y + 5 - time][z + 8] + u[t0][x + 8 - time][y + 11 - time][z + 8]) - 2.0e-1F * (u[t0][x + 8 - time][y + 6 - time][z + 8] + u[t0][x + 8 - time][y + 10 - time][z + 8]) + 1.6F * (u[t0][x + 8 - time][y + 7 - time][z + 8] + u[t0][x + 8 - time][y + 9 - time][z + 8]) - 2.84722222F * u[t0][x + 8 - time][y + 8 - time][z + 8]) + r2 * (-1.78571429e-3F * (u[t0][x + 8 - time][y + 8 - time][z + 4] + u[t0][x + 8 - time][y + 8 - time][z + 12]) + 2.53968254e-2F * (u[t0][x + 8 - time][y + 8 - time][z + 5] + u[t0][x + 8 - time][y + 8 - time][z + 11]) - 2.0e-1F * (u[t0][x + 8 - time][y + 8 - time][z + 6] + u[t0][x + 8 - time][y + 8 - time][z + 10]) + 1.6F * (u[t0][x + 8 - time][y + 8 - time][z + 7] + u[t0][x + 8 - time][y + 8 - time][z + 9]) - 2.84722222F * u[t0][x + 8 - time][y + 8 - time][z + 8]);
              }
#pragma omp simd aligned(u : 32)
              for (int spzi = 0; spzi < sparse_source_mask_NNZ[x - time][y - time]; spzi++) // Inner block loop
              {
                zind = sparse_source_mask[x - time][y - time][spzi];
                //printf("\n zind is : %d", sparse_source_mask[xi - titer2][yi - titer2][spzi]);
                u[t1][x + 8 - time][y + 8 - time][zind + 8] += source_mask[x - time][y - time][zind] * save_src[(source_id[x - time][y - time][zind])][time];
                //printf("\n Inject source in grid[%d][%d][%d], src = %f, ", xi-titer2, yi-titer2, zind, save_src[(source_id[xi - titer2][yi - titer2][zind])][titer2]);
                //printf("\n update : %d, %d, %d in %d", xi - titer2, yi - titer2, zi ,titer2);
              }
            }
          }
        }
      }
    }
  }
}
