#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <iostream>

const int MAXN = 1 << 26;
double *a;

double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}

double sum_oridinary(int n) {
    double sum = 0.0;
	for (int i = 0; i < n; i++) // 逐个累加
	    sum += a[i];  
    return sum;
}
double sum_recursive(int l, int r) {
    if (l == r) 
        return a[l];  
    int mid = (l + r) / 2;
    return sum_recursive(l, mid) + sum_recursive(mid + 1, r);  // 合并结果
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

    // 初始化数组
    for (int i = 0; i < MAXN; i++) {
        a[i] = i * 0.01;
    }

    // 测量运行时间
    double start_time, end_time;
    for(int n = 8;n <= (1<<26);n <<= 1){
    	printf("scale n : %d\n", n);
    	
   	 	start_time = get_time();  
    	sum_oridinary(n);//普通 
	    end_time = get_time();  
	    printf("ordinary_time : %.6f ms\n", end_time - start_time);
	    
   	 	start_time = get_time();  
    	sum_recursive(0,n-1);//递归 
	    end_time = get_time();  
	    printf("recursive_time : %.6f ms\n", end_time - start_time);
	    
   	 	start_time = get_time();  
    	sum_unroll(n);//循环展开 
	    end_time = get_time();  
	    printf("unroll_time : %.6f ms\n", end_time - start_time);
	}
    delete[] a;
    return 0;
}
