#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <iostream>

const int MAXN = 10005;
double **A, *x, *y;

double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000 + t.tv_usec * 1e-3;
}

// 平凡算法：逐列访问矩阵元素
void mat_vec_naive(int n) {
    double lp = std::max(1.0,std::ceil(100000000.0 / n / n));
    double start_time = get_time();  // 获取开始时间
    for(int l = 0;l < lp;++ l){
        for (int j = 0; j < n; j++) {  // 外层循环遍历每一列
            y[j] = 0;  // 初始化结果
            for (int i = 0; i < n; i++) {  // 内层循环遍历每一行
                y[j] += A[i][j] * x[i];  // 计算内积
            }
        }
    }
    double end_time = get_time();  // 获取结束时间
    printf("ordinary_time : %.6f ms\n", (end_time - start_time) / lp);
}

// 优化算法：逐行访问矩阵元素
void mat_vec_optimized(int n) {
    double lp = std::max(1.0,100000000.0 / n / n);
    double start_time = get_time();  // 获取开始时间
    for(int l = 0;l < lp;++ l){
    	for (int j = 0; j < n; j++)
    		y[j] = 0;
	    for (int i = 0; i < n; i++) {
	        double xi = x[i]; 
	        for (int j = 0; j < n; j++) {
	            y[j] += A[i][j] * xi;  // 累加结果
	        }
	    }
	}
    double end_time = get_time();  // 获取结束时间
    printf("optimized_time : %.6f ms\n", (end_time - start_time) / lp);
}

int main() {
    A = new double*[MAXN];
    for (int i = 0; i < MAXN; i++) {
        A[i] = new double[MAXN];
    }
    x = new double[MAXN];
    y = new double[MAXN];
    // 初始化数据
    for (int i = 0; i < MAXN; i++) {
        x[i] = i * 0.01;
        for (int j = 0; j < MAXN; j++) {
            A[i][j] = i + j;
        }
    }
    // 测量运行时间
    for(int n = 10;n <= 100;n += 10){
    	printf("scale n : %d\n", n);
    	mat_vec_naive(n);
    	mat_vec_optimized(n);
	}
    for(int n = 100;n <= 1000;n += 100){
    	printf("scale n : %d\n", n);
    	mat_vec_naive(n);
    	mat_vec_optimized(n);
	}
    for(int n = 1000;n <= 10000;n += 1000){
    	printf("scale n : %d\n", n);
    	mat_vec_naive(n);
    	mat_vec_optimized(n);
	}
	
    for (int i = 0; i < MAXN; i++) {
        delete[] A[i];
    }
    delete[] A;
    delete[] x;
    delete[] y;
    return 0;
}
