#include <iostream>
#include <cmath>
#include <mpi.h>

using namespace std;

double func(double x)
{
    // Define the function to integrate here
    return sin(x); // Example: sin(x)
}

void partition(double A[], double a, double b, int n)
{
    double h = (b - a) / n;
    for (int i = 0; i <= n; ++i)
    {
        A[i] = a + i * h;
    }
}

double BoolesMethod(double a, double b, int n)
{
    double h = (b - a) / n;
    int k = n / 4;
    double *A = new double[n + 1];
    partition(A, a, b, n);
    double local_sum = 0.0;

    // Parallel computation with MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_k = k / size;
    int start = rank * local_k;
    int end = (rank == size - 1) ? k : start + local_k;

    for (int i = start; i < end; ++i)
    {
        local_sum += 7 * func(A[4 * i]) +
                     32 * func(A[4 * i + 1]) +
                     12 * func(A[4 * i + 2]) +
                     32 * func(A[4 * i + 3]) +
                     7 * func(A[4 * i + 4]);
    }

    double global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    delete[] A; // Free allocated memory

    if (rank == 0)
    {
        double Integration = 2 * h / 45 * global_sum;
        return Integration;
    }

    return 0.0; // Return 0 for all ranks except rank 0
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    double a = 0.0; // Lower limit
    double b = 2.0; // Upper limit
    int n = 100;    // Number of subintervals, should be a multiple of 4

    double result = BoolesMethod(a, b, n);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        cout << "Result of integration: " << result << endl;
    }

    MPI_Finalize();
    return 0;
}
