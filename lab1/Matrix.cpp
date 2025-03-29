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

// ƽ���㷨�����з��ʾ���Ԫ��
void mat_vec_naive(int n) {
    double lp = std::max(1.0,std::ceil(100000000.0 / n / n));
    double start_time = get_time();  // ��ȡ��ʼʱ��
    for(int l = 0;l < lp;++ l){
        for (int j = 0; j < n; j++) {  // ���ѭ������ÿһ��
            y[j] = 0;  // ��ʼ�����
            for (int i = 0; i < n; i++) {  // �ڲ�ѭ������ÿһ��
                y[j] += A[i][j] * x[i];  // �����ڻ�
            }
        }
    }
    double end_time = get_time();  // ��ȡ����ʱ��
    printf("ordinary_time : %.6f ms\n", (end_time - start_time) / lp);
}

// �Ż��㷨�����з��ʾ���Ԫ��
void mat_vec_optimized(int n) {
    double lp = std::max(1.0,100000000.0 / n / n);
    double start_time = get_time();  // ��ȡ��ʼʱ��
    for(int l = 0;l < lp;++ l){
    	for (int j = 0; j < n; j++)
    		y[j] = 0;
	    for (int i = 0; i < n; i++) {
	        double xi = x[i]; 
	        for (int j = 0; j < n; j++) {
	            y[j] += A[i][j] * xi;  // �ۼӽ��
	        }
	    }
	}
    double end_time = get_time();  // ��ȡ����ʱ��
    printf("optimized_time : %.6f ms\n", (end_time - start_time) / lp);
}

int main() {
    A = new double*[MAXN];
    for (int i = 0; i < MAXN; i++) {
        A[i] = new double[MAXN];
    }
    x = new double[MAXN];
    y = new double[MAXN];
    // ��ʼ������
    for (int i = 0; i < MAXN; i++) {
        x[i] = i * 0.01;
        for (int j = 0; j < MAXN; j++) {
            A[i][j] = i + j;
        }
    }
    // ��������ʱ��
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
