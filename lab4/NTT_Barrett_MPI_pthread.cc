#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <sys/time.h>
#include <omp.h>
#include <pthread.h>
#include <mpi.h>
// 可以自行添加需要的头文件

typedef long long LL;
const int G = 3;

void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

class Barrett {
private:
    uint64_t p;
    __uint128_t mu;  // 用于精度计算

public:
    // 构造函数，预处理 mu = floor(2^64 / p)
    Barrett(uint64_t m) : p(m) {
        mu = (__uint128_t(1) << 64) / m;
    }

    // 计算 a mod p，要求 a < p^2
    uint64_t reduce(uint64_t a) const {
        __uint128_t x = a;
        __uint128_t q = (x * mu) >> 64;     // q ≈ a / p
        uint64_t ret = a - q * p;
        return (ret >= p ? ret - p : ret);
    }

    // 模乘：返回 (a * b) mod p
    uint64_t mul(uint64_t a, uint64_t b) const {
        return reduce((__uint128_t)a * b);
    }
};
std::vector<Barrett> barrett;

//光速防爆乘法
__int128 gsc(__int128 x,__int128 y,__int128 p)
{
	__int128 ret = x*y - (__int128)((long double)x/p*y+0.5)*p;
	return (ret%p+p)%p;
} 
int rev[300000],PHI,len,l;
//快速幂
int qpow(int x,int y,int p)
{
	int ret = 1;
	while(y){if(y & 1) ret = 1ll * ret * x % p;x = 1ll * x * x % p;y >>= 1;}
	return ret;
}
//扩展欧几里得算法
void exgcd(int a,int b,int &x,int &y)
{
    if(!b){x = 1,y = 0;}
    else{
        exgcd(b,a%b,y,x);
        y -= a/b*x;
    }
}
//利用扩展欧几里得算法求逆元
int getinv(int x,int p)
{
    int inv,y;
    exgcd(x,p,inv,y);
    return (inv+p) % p;
}

typedef struct{
    int *a;
    int i, len, p, w, tid, thread_num, pid;
}ThreadArg;

void* ntt_thread(void *arg) {
    ThreadArg* t = (ThreadArg*)arg;
    int *a = t->a;
    int i = t->i, p = t->p, w = t->w;
    int s = i << 1;
    int tid = t->tid, thread_num = t->thread_num;
    int pid = t->pid;

    for (int j = tid * s; j < t->len; j += thread_num * s) {
        int mi = 1;
        for (int k = 0; k < i; ++k, mi = barrett[pid].mul(mi,w)) {
            int X = a[j + k];
            int Y = barrett[pid].mul(mi,a[i + j + k]);
            a[j + k] = barrett[pid].reduce(X + Y);
            a[i + j + k] = barrett[pid].reduce(X - Y + p);
        }
    }
    return NULL;
}

// 四个模数 479*2^21+1,7*2^26+1, 997*2^20+1, 119*2^23+1，原根均为3
const int MOD[4] = {1004535809, 469762049, 1045430273, 998244353}; 
void NTT(int *a,int opt,int p, int pid, int thread_num = 4)
{
    //蝴蝶变换
	for(int i = 0;i < len;++ i) if(i < rev[i]) std::swap(a[i],a[rev[i]]);
    //NTT主体
    int GINV = getinv(G,p);
    std::vector<pthread_t> threads(thread_num);
    std::vector<ThreadArg> args(thread_num);
    // pthread_t threads[64];
    // ThreadArg args[64];
	for(int i = 1;i < len;i <<= 1)
	{
		int w = qpow(opt == 1 ? G : GINV,(p-1) / (i << 1),p);
        for (int t = 0; t < thread_num; ++t) {
            args[t] = {a, i, len, p, w, t, thread_num, pid};
            pthread_create(&threads[t], NULL, ntt_thread, &args[t]);
        }

        for (int t = 0; t < thread_num; ++t) {
            pthread_join(threads[t], NULL);
        }
	}
	if(opt == -1) {
	    int invlen = getinv(len,p);
        // for(int i = 0;i < len;++ i) a[i] = 1ll * a[i] * invlen % p;
        for(int i = 0;i < len;++ i) a[i] = barrett[pid].mul(a[i],invlen);
    }
}
void poly_multiply(int *a, int *b, int *ab, int n, int p,int pid){
    len = 1, l = -1;
	while(len < n+n-1) len <<= 1,l++;
	for(int i = 0;i < len;++ i) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << l);
    NTT(a,1,p,pid);
    NTT(b,1,p,pid);
    // for(int i = 0;i < len;++ i) ab[i] = 1ll * a[i] * b[i] % p;
    for(int i = 0;i < len;++ i) ab[i] = barrett[pid].mul(a[i],b[i]);
    NTT(ab,-1,p,pid);
}
int ans[4][300000]; // 存储每个模数下的结果
int A[4][300000], B[4][300000]; // 存储每个模数下的输入多项式
// 中国剩余定理的多项式乘法
void CRT_poly_multiply(int *a, int *b, int *ab, int n, int p, int rank, int size){
    len = 1, l = -1;
	while(len < n+n-1) len <<= 1,l++;
	for(int i = 0;i < len;++ i) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << l);
    for(int j = 0; j < len; ++j){
        if(j < n){
            A[rank][j] = barrett[rank].reduce(a[j]);
            B[rank][j] = barrett[rank].reduce(b[j]);
        }
        else{
            A[rank][j] = 0;
            B[rank][j] = 0;
        }
    }
    poly_multiply(A[rank], B[rank], ans[rank], n, MOD[rank],rank);
    //发送结果到主进程
    if(rank > 0)
        MPI_Send(ans[rank], len, MPI_INT, 0, rank, MPI_COMM_WORLD);
    //接收其他进程的结果
    if(rank == 0){
        for(int i = 1; i < 4; ++i){
            MPI_Recv(ans[i], len, MPI_INT, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    //等待发送和接收完成
    MPI_Barrier(MPI_COMM_WORLD);
    //主进程负责CRT合并
    if(rank == 0){
        __int128 M = 1ll * MOD[0] * MOD[1];
        M *= MOD[2];  M *= MOD[3];
        int inv[4];
        for(int i = 0; i < 4; ++i)
            inv[i] = getinv((M / MOD[i]) % MOD[i], MOD[i]);
        for(int i = 0;i < len;++ i){
            __int128 ret = 0;
            for(int j = 0;j < 4;++ j){
                // ret = (ret + inv * (M / MOD[j]) % M * ans[j][i]) % M;
                ret = (ret + gsc(1ll * inv[j] * ans[j][i], M / MOD[j], M)) % M;
            }
            ab[i] = ret % p;
        }
    }
}

int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 1337006139375617(并非263882790666241)
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    for(int i = 0;i < 4;++ i)
        barrett.emplace_back(Barrett(MOD[i]));
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        std::cout << "test id: " << i << "\n";
        long double ans = 0;
        int n_, p_;
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        if (rank == 0)
            fRead(a, b, &n_, &p_, i);
        // broadcast !
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(a, n_, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, n_, MPI_INT, 0, MPI_COMM_WORLD);
        memset(ab, 0, sizeof(ab));
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        // auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        // poly_multiply(a, b, ab, n_, p_);
        CRT_poly_multiply(a, b, ab, n_, p_, rank, size);
        // auto End = std::chrono::high_resolution_clock::now();
        double end_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0)
            ans += (end_time - start_time) * 1000;
        if (rank == 0){
            fCheck(ab, n_, i);
            std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (ms) "<<std::endl;
            // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
            // 禁止使用 cout 一次性输出大量文件内容
            fWrite(ab, n_, i);
        }
    }
    MPI_Finalize();
    return 0;
}
