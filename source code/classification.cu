#include "dtw_h.h"

using namespace std;


int testSize = 128;
int trainSize = 1024;
const int blockSize = 1024;
const int window_size = 1024;



int main( int argc, char** argv)
{
		
	int i,j,cur;
	float tmp;
	double endtime;
	clock_t init, final;
	int dist = 0;
	int testSize;

	//choosing test file size
	if( argc >= 4 )	{
		testSize = atoi(argv[3]);
	}
	
	infoDev(0);


	unsigned long long int trainBytes = trainSize * 2 * window_size * sizeof(float);
	unsigned long long int testBytes = testSize * window_size * sizeof(float);
	
	float* h_train = (float*) malloc (trainBytes);
	int * trainLabels = (int *) malloc(trainSize*sizeof(int));
	float* h_test = (float*) malloc (testBytes);
	int * testLabels = (int *) malloc(testSize*sizeof(int));


	FILE *trainFile;
    trainFile=fopen(argv[1],"r");
    if(trainFile==NULL) printf("The file %s cannot be opened!\n", argv[1]);

    //reading training file
    for(i = 0; i < trainSize; i++) 
    {
        fscanf(trainFile,"%f",&tmp);
        trainLabels[i] = (int)tmp;
		for (j=0; j<window_size; j++)
		{
			fscanf(trainFile,"%f",&tmp);
			h_train[(2*i)*window_size+j] = tmp;
			h_train[(2*i+1)*window_size+j] = tmp;
		}        
    }
    fclose(trainFile);
    

	FILE *testFile;
    testFile=fopen(argv[2],"r");
    if(testFile==NULL) printf("The file %s cannot be opened!\n", argv[1]);    

    //reading testing file
    for(i = 0; i < testSize; i++)
    {
        fscanf(testFile,"%f",&tmp);
		testLabels[i] = (int)tmp;        
		for (j=0; j<window_size; j++)
		{
			fscanf(testFile,"%f",&tmp);
			h_test[i*window_size+j] = tmp;
		}
    }
    fclose(testFile);

    //picking a distance from the command line
	if( argc >= 5 )
		dist = atoi(argv[4]);
	

	float* h_Out = (float*) malloc (trainSize*window_size*sizeof(float));
	float* d_A = 0;
	cudaMalloc((void**)&d_A, trainBytes);
	cudaMemcpy(d_A, h_train, trainBytes, cudaMemcpyHostToDevice);


	float* d_Out = 0;
	float* d_query = 0;
	cudaMalloc((void**)&d_query,window_size*sizeof(float));
	cudaMalloc((void**)&d_Out, trainSize*window_size*sizeof(float));

	dim3 grid(trainSize*window_size/blockSize,1);
	dim3 threads(blockSize,1);

	// printf("blocks: %d\n", trainSize*window_size/blockSize);
	// printf("thread: %d\n", blockSize);

    infoDev(1);
    printf("\n****************Light Curve Classification****************\n\n");
	printf("Path Training set: %s, length: %ld, n_attrs: %d byte_size: %f\n",argv[1],trainSize,1024,1024*sizeof(float)*(float)trainSize);
	printf("Path Testing set: %s, length: %ld, n_attrs: %d, byte_size: %f\n",argv[2],testSize,1024,1024*sizeof(float)*(float)testSize);
    printf("Grid_size_x: %d, number_of_threads_x: %d \n", grid.x,threads.x);
    printf("Grid_size_y: %d, number_of_threads_y: %d \n\n", grid.y,threads.y);

	float * d_Out_copy = (float *) malloc(trainSize*window_size*sizeof(float));

	init = clock();

	for( j = 0 ; j < trainSize ; j++ )
		cur = 0;
	int err = 0 , errNR = 0 , minI = -1 , minINR  = -1;

	while( cur < testSize ) 
	{

		cudaMemcpy(d_query, h_test + window_size*cur , window_size*sizeof(float), cudaMemcpyHostToDevice);

		if( dist == 1){
			printf("Distance measure used: rDTW\n");
			rDTW <<<grid, threads>>> (d_A, d_query, d_Out, window_size); //rotation invariant
		}
		else{
			printf("Distance measure used: rED\n");
			rED <<<grid, threads>>> (d_A, d_query, d_Out, window_size);
		}

		checkCUDAError("DTW Kernel");
		cudaThreadSynchronize();

		cudaMemcpy(h_Out, d_Out, trainSize*window_size*sizeof(float) , cudaMemcpyDeviceToHost);


		float  min = 9999999999.99;
		minI = -1;
		float minNR = 99999999999.99;
		minINR = -1; 
		i = 0;
		for( j = 0 ; j < trainSize ; j++ )
		{
			if ( minNR > h_Out[j*window_size] )
			{
				minNR = h_Out[j*window_size];
				minINR = j;
			}
			for( i = 0 ; i < window_size ; i++ )
			{
				int t = j*window_size+i;
				if ( min > h_Out[t] )
				{
					min = h_Out[t];
					minI = j;
				}
			}
		}
		if( trainLabels[minI] != testLabels[cur] )
			err++;

		if( trainLabels[minINR] != testLabels[cur] )
			errNR++;

		printf("%d\t%d\tRI : %d\t%d\t%3.6f \t\t NRI : %d\t%d\t%3.6f\n",cur , testLabels[cur] , trainLabels[minI] ,  minI, min, trainLabels[minINR], minINR , minNR );
		cur++;
	}
	cudaFree(d_A);
	cudaFree(d_Out);
	cudaFree(d_query);
	
	free(trainLabels);
	free(testLabels);
	
	final = clock() - init;
	endtime = (double)final / ((double)CLOCKS_PER_SEC);
	printf("Total Time %f\n", endtime);
	printf("Rotation Invariant Accuracy is %f\n",(float)(testSize-err)*(100.0/testSize));
	printf("Regular Accuracy is %f\n",(float)(testSize-errNR)*(100.0/testSize));	

	return 0;
}	
