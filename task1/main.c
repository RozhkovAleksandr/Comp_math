// #include "/usr/local/opt/libomp/include/omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "math.h"

#define EPSILON 0.1
#define N 3000
#define NB 128

void free_uf(double **u, double **f)
{
    for (int i = 0; i < N; i++)
    {
        free(u[i]);
        free(f[i]);
    }
    free(u);
    free(f);
}

void create_matrix(double **u, double **f)
{
    int x, y;

    for (x = 0; x < N; x++)
    {
        for (y = 0; y < N; y++)
        {
            if (x == 0)
            {
                u[x][y] = (100 - 200 * y);
            }
            else if (x == 1)
                u[x][y] = (-100 + 200 * y);
            else if (y == 0)
                u[x][y] = (100 - 200 * x);
            else if (y == 1)
                u[x][y] = (-100 + 200 * x);

            else
            {
                u[x][y] = (double)(rand() % 100);
            }
            // u[x][y] = 6000 * x + 9000 * y;
            f[x][y] = 0.0;
        }
    }
}

int algo(double **u, double **f)
{
    double dmax;
    int iter = 0;
    int block_size = (N - 2) / NB;
    double *dm = calloc(block_size, sizeof(*dm));
    double h = 1.0 / (N + 1);
    do
    {
        iter++;
        dmax = 0.0;
        for (int nx = 0; nx < block_size; nx++)
        {
            dm[nx] = 0;
            double d;
            int i, j;
#pragma omp parallel for shared(nx, dm) private(i, j, d)
            for (i = 0; i < nx + 1; i++)
            {
                j = nx - i;
                int x0 = 1 + i * NB;
                int xmax = fmin(x0 + NB, N - 1);
                int y0 = 1 + j * NB;
                int ymax = fmin(y0 + NB, N - 1);
                double dm_block = 0;

                for (int x = x0; x < xmax; x++)
                {
                    for (int y = y0; y < ymax; y++)
                    {
                        double temp = u[x][y];
                        u[x][y] = 0.25 * (u[x - 1][y] + u[x + 1][y] + u[x][y - 1] + u[x][y + 1] - h * h * f[x][y]);
                        double temp_d = fabs(temp - u[x][y]);
                        if (dm_block < temp_d)
                        {
                            dm_block = temp_d;
                        }
                    }
                }

                if (dm[i] < d)
                {
                    dm[i] = d;
                }
            }
        }

        for (int nx = block_size - 2; nx >= 0; nx--)
        {
            int i, j;
            double d;
#pragma omp parallel for shared(nx, dm) private(i, j, d)
            for (int i = block_size - nx - 1; i < block_size; i++)
            {
                int j = block_size + ((block_size - 2) - nx) - i;
                int x0 = 1 + i * NB;
                int xmax = fmin(x0 + NB, N - 1);
                int y0 = 1 + j * NB;
                int ymax = fmin(y0 + NB, N - 1);
                double dm_block = 0;

                for (int x = x0; x < xmax; x++)
                {
                    for (int y = y0; y < ymax; y++)
                    {
                        double temp = u[x][y];
                        u[x][y] = 0.25 * (u[x - 1][y] + u[x + 1][y] + u[x][y - 1] + u[x][y + 1] - h * h * f[x][y]);
                        double temp_d = fabs(temp - u[x][y]);
                        if (dm_block < temp_d)
                        {
                            dm_block = temp_d;
                        }
                    }
                }

                if (dm[i] < dm_block)
                {
                    dm[i] = dm_block;
                }
            }
        }
        int i;
        for (i = 1; i <= block_size; i++)
        {
            if (dmax < dm[i])
                dmax = dm[i];
        }
    } while (dmax > EPSILON);

    return iter;
}

int main()
{
    double t1, t2;
    int threads = 8;
    omp_set_num_threads(threads);
    t1 = omp_get_wtime();

    double **u = (double **)malloc(N * sizeof(double *));
    double **f = (double **)malloc(N * sizeof(double *));
    int i, j;
    for (i = 0; i < N; i++)
    {
        u[i] = (double *)malloc(N * sizeof(double));
        f[i] = (double *)malloc(N * sizeof(double));
    }
    create_matrix(u, f);

    int iter = algo(u, f);

    t2 = omp_get_wtime();
    printf("threads = %d;    ", threads);
    printf("size = %d;   ", N);
    printf("time = %f;   ", t2 - t1);
    printf("iter = %d.", iter);

    free_uf(u, f);
}
