#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <iostream>

const int MAXN = 1 << 28;
double *a;

double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000 + t.tv_usec * 1e-3;
}

double sum_oridinary(int n) {
    double sum = 0.0;
	for (int i = 0; i < n; i++) // 逐个累加
	    sum += a[i];  
    return sum;
}
double sum_multilink(int n) {
	double sum1 = 0, sum2 = 0; 
	for(int i = 0;i < n;i += 2){
		sum1 += a[i];
		sum2 += a[i+1]; 
	}
	return sum1 + sum2;
}
double sum_unroll(int n) {
    double sum = 0.0;
	for (int i = 0; i < n; i += 8) {
	    sum += a[i] + a[i+1] + a[i+2] + a[i+3] + a[i+4] + a[i+5] + a[i+6] + a[i+7]; 
	}
    return sum;
}

int main() {
    a = new double[MAXN];

    // 测量运行时间
    double start_time, end_time;
    for(int n = 11;n <= 28;++ n){

    	printf("scale n : %d\n", n);
    	
   	 	start_time = get_time();  
    	sum_oridinary(1<<n);//普通 
	    end_time = get_time();  
	    printf("ordinary_time : %.6f ms\n", end_time - start_time);
	    
   	 	start_time = get_time();  
    	sum_multilink(1<<n);//多链路 
	    end_time = get_time();  
	    printf("multilink_time : %.6f ms\n", end_time - start_time);
	    
   	 	start_time = get_time();  
    	sum_unroll(1<<n);//循环展开 
	    end_time = get_time();  
	    printf("unroll_time : %.6f ms\n", end_time - start_time);
	}
    delete[] a;
    return 0;
}
