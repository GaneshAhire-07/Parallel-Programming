#include<mpi.h>
#include<iostream>
using namespace std;

int main(int argc,char** argv)
{
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	//cout << "Hello World" << endl;
	cout << "I am being printed on processor " << rank << " out of " << size << " total processors" << endl;
	MPI_Finalize();
	
	return 0;
}