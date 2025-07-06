#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>

typedef long long LL;
const int G = 3;

// ---------------- 输入输出函数保持不变 ----------------
void fRead(int *a, int *b, int *n, int *p, int input_id) {
    std::string str1 = "../data/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) fin >> a[i];
    for (int i = 0; i < *n; i++) fin >> b[i];
}

void fCheck(int *ab, int n, int input_id) {
    std::string str1 = "../data/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++) {
        int x; fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
}

void fWrite(int *ab, int n, int input_id) {
    std::string str1 = "../files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++) fout << ab[i] << '\n';
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

// ---------------- 工具函数 ----------------
__host__ __device__ inline int qpow(int x, int y, int p) {
    int ret = 1;
    while (y) {
        if (y & 1) ret = (LL)ret * x % p;
        x = (LL)x * x % p;
        y >>= 1;
    }
    return ret;
}

__host__ int getinv(int x, int p) {
    int a = x, b = p, u = 1, v = 0;
    while (b) {
        int t = a / b;
        a -= t * b; std::swap(a, b);
        u -= t * v; std::swap(u, v);
    }
    return (u + p) % p;
}

__global__ void bit_reverse_copy(int *a, int *rev, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len && tid < rev[tid]) {
        int tmp = a[tid];
        a[tid] = a[rev[tid]];
        a[rev[tid]] = tmp;
    }
}

__global__ void ntt_stage_shared(int *a, int len, int step, const int *roots, int p) {
    extern __shared__ int shared_roots[]; // 动态分配共享内存

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = step << 1;
    int thread_in_block = threadIdx.x;

    // 所在线程块中的第 threadIdx.x 个线程处理的位置
    int start = (tid / step) * m + (tid % step);

    // 加载 step 个旋转因子到共享内存（仅前 step 个线程做这件事）
    if (thread_in_block < step)
        shared_roots[thread_in_block] = roots[step + thread_in_block];

    __syncthreads(); // 确保共享内存填充完毕

    if (start + step < len) {
        int u = a[start];
        int v = (LL)a[start + step] * shared_roots[tid % step] % p;
        a[start] = (u + v) % p;
        a[start + step] = (u - v + p) % p;
    }
}

__host__ __device__ inline unsigned __int128 mul128(unsigned long long a, unsigned long long b) {
    return (unsigned __int128)a * b;
}

__host__ __device__ inline int barrett_reduce(unsigned long long a, unsigned long long p, unsigned long long m) {
    unsigned long long q = (unsigned __int128)a * m >> 64;
    unsigned long long r = a - q * p;
    return r < p ? r : r - p;
}

__global__ void pointwise_mul_barrett(int *a, int *b, int *c, int len, int p, unsigned long long m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        unsigned long long prod = (unsigned long long)a[i] * b[i];
        c[i] = barrett_reduce(prod, p, m);
    }
}


void poly_multiply_cuda(int *a, int *b, int *ab, int n, int p,int pid) {
    cudaEvent_t start, stop;//计时器
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);//开始计时

    int len = 1, loglen = -1;
    while (len < 2 * n - 1) len <<= 1, loglen++;

    int *ta = new int[len]();
    int *tb = new int[len]();
    memcpy(ta, a, n * sizeof(int));
    memcpy(tb, b, n * sizeof(int));

    int *rev = new int[len];
    for (int i = 0; i < len; i++)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << loglen);

    int *root = new int[len<<1];
    for (int i = 1; i < len; i <<= 1) {
        int wn = qpow(G, (p - 1) / (i << 1), p);
        root[i] = 1;
        for (int j = 1; j < i; j++)
            root[i + j] = barrett[pid].mul(root[i + j - 1], wn);
    }

    int Ginv = getinv(G, p);
    int *inv_root = new int[len << 1];
    for (int i = 1; i < len; i <<= 1) {
        int wn = qpow(Ginv, (p - 1) / (i << 1), p);
        inv_root[i] = 1;
        for (int j = 1; j < i; j++)
            inv_root[i + j] = barrett[pid].mul(inv_root[i + j - 1], wn);
    }

    int *da, *db, *dc, *drev, *droot, *d_inv_root;
    cudaMalloc(&da, len * sizeof(int));
    cudaMalloc(&db, len * sizeof(int));
    cudaMalloc(&dc, len * sizeof(int));
    cudaMalloc(&drev, len * sizeof(int));
    cudaMalloc(&droot, (len<<1) * sizeof(int));
    cudaMalloc(&d_inv_root, (len << 1) * sizeof(int));

    cudaMemcpy((int4*)da, (int4*)ta, len/4 * sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy((int4*)db, (int4*)tb, len/4 * sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy((int4*)drev, (int4*)rev, len/4 * sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy((int4*)droot, (int4*)root, (len<<1)/4 * sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy((int4*)d_inv_root, (int4*)inv_root, (len << 1)/4 * sizeof(int4), cudaMemcpyHostToDevice);


    int block = 512, grid = (len + block - 1) / block;

    bit_reverse_copy<<<grid, block>>>(da, drev, len);
    for (int s = 1; s < len; s <<= 1)
        ntt_stage_shared<<<grid, block, s * sizeof(int)>>>(da, len, s, droot, p);

    bit_reverse_copy<<<grid, block>>>(db, drev, len);
    for (int s = 1; s < len; s <<= 1)
        ntt_stage_shared<<<grid, block, s * sizeof(int)>>>(db, len, s, droot, p);

    unsigned __int128 m = 1;
    m <<= 64; m /= p;
    pointwise_mul_barrett<<<grid, block>>>(da, db, dc, len, p, m);

    bit_reverse_copy<<<grid, block>>>(dc, drev, len);
    for (int s = 1; s < len; s <<= 1)
        ntt_stage_shared<<<grid, block, s * sizeof(int)>>>(dc, len, s, d_inv_root, p);

    int inv_len = getinv(len, p);
    cudaMemcpy(ab, dc, len * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < len; i++)
    //     ab[i] = (LL)ab[i] * inv_len % p;
    for(int i = 0;i < len;++ i)
        ab[i] = barrett[pid].mul(ab[i],inv_len);

    delete[] ta; delete[] tb; delete[] rev; delete[] root; delete[] inv_root;
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(drev); cudaFree(droot); cudaFree(d_inv_root);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);//停止计时
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout<<"average latency for n = "<<n<<" p = "<<p<<" : "<<elapsedTime<<" (ms) "<<std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int a[300000], b[300000], ab[300000];

int main() {
    int test_begin = 0, test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        std::cout << "test id: " << i << "\n";
        long double ans = 0;
        int n_, p_;
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        barrett.emplace_back(Barrett(p_));
        // auto Start = std::chrono::high_resolution_clock::now();
        poly_multiply_cuda(a, b, ab, n_, p_, i-test_begin);
        // cudaDeviceSynchronize();  // 等待 GPU 计算完成
        // auto End = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        // ans += elapsed.count();
        fCheck(ab, n_, i);
        // std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (ms) "<<std::endl;
        fWrite(ab, n_, i);
    }
    return 0;
}