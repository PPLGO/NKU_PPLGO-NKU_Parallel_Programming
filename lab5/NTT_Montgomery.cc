#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
// 可以自行添加需要的头文件

typedef long long LL;
const int G = 3;

void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
    std::string str1 = "../data/";
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
    std::string str1 = "../data/";
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
    std::string str1 = "../files/";
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

LL R, Rinv, p_inv, mod, R2;

int rev[300000],PHI,len,l;
int qpow(int x, int y, int p) {
    int ret = 1;
    while (y) {
        if (y & 1) ret = 1LL * ret * x % p;
        x = 1LL * x * x % p;
        y >>= 1;
    }
    return ret;
}

int getinv(int a, int p) {
    int b = p, u = 1, v = 0;
    while (b) {
        int t = a / b;
        a -= t * b; std::swap(a, b);
        u -= t * v; std::swap(u, v);
    }
    return (u + p) % p;
}

inline int montgomery_reduce(LL t) {
    int m = ((t & (R - 1)) * p_inv) & (R - 1);
    int u = (t + 1LL * m * mod) >> l;
    if (u >= mod) u -= mod;
    return u;
}

inline int montgomery_mul(int a, int b) {
    return montgomery_reduce(1LL * a * b);
}

void NTT(int *a, int opt) {
    for (int i = 0; i < len; ++i) if (i < rev[i]) std::swap(a[i], a[rev[i]]);
    
    for (int i = 1; i < len; i <<= 1) {
        int wn = qpow(opt == 1 ? G : getinv(G, mod), (mod - 1) / (i << 1), mod);
        wn = montgomery_mul(wn, R2); // 转入Montgomery域
        for (int j = 0; j < len; j += (i << 1)) {
            int w = R; // w = 1 in Montgomery domain
            for (int k = 0; k < i; ++k) {
                int u = a[j + k];
                int v = montgomery_mul(a[j + k + i], w);
                a[j + k] = u + v;
                if (a[j + k] >= mod) a[j + k] -= mod;
                a[j + k + i] = u - v;
                if (a[j + k + i] < 0) a[j + k + i] += mod;
                w = montgomery_mul(w, wn);
            }
        }
    }

    if (opt == -1) {
        int inv_len = getinv(len, mod);
        inv_len = montgomery_mul(inv_len, R2); // 转入Montgomery域
        for (int i = 0; i < len; ++i) a[i] = montgomery_mul(a[i], inv_len);
    }
}
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    mod = p;
    R = 1ll << 32; // Must be larger than mod
    Rinv = getinv(R % p, p);
    p_inv = (R * Rinv - 1) / p;
    int invR = getinv(R, p);

    R2 = (1LL * R * R) % p; // 用于进入 Montgomery 域

    len = 1; l = -1;
    while (len < 2 * n - 1) len <<= 1, l++;
    for (int i = 0; i < len; ++i)
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << l);

    for (int i = 0; i < len; ++i) {
        a[i] = (i < n ? montgomery_mul(a[i], R2) : 0);
        b[i] = (i < n ? montgomery_mul(b[i], R2) : 0);
    }

    NTT(a, 1);
    NTT(b, 1);

    for (int i = 0; i < len; ++i)
        ab[i] = montgomery_mul(a[i], b[i]);

    NTT(ab, -1);

    for (int i = 0; i < len; ++i)
        ab[i] = montgomery_reduce(ab[i]); // 返回普通整数域
}

int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        std::cout << "test id: " << i << "\n";
        long double ans = 0;
        int n_, p_;
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (ms) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }
    return 0;
}
