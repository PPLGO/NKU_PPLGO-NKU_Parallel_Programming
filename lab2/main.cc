#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <arm_neon.h>
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

int rev[300000],PHI,len,l;
uint32_t p_inv,rmodp,r2modp;
//利用费马小定理求逆元 
LL qpow(LL x,int y,LL p)
{
	LL ret = 1;
	while(y){if(y & 1) ret = ret * x % p;x = x * x % p;y >>= 1;}
	return ret;
}
inline uint32_t REDC(uint64_t T,uint32_t p){
    // m = (T mod R * N') mod R
    uint64_t m = ((T & 0xFFFFFFFFull) * p_inv)& 0xFFFFFFFFull;
    // t = (T + m * p) / R
    uint64_t t = (T + m * p) >> 32;
    // 如果 t >= p, 则减去 p
    if(t >= p) t -= p;
    return t;
}
inline uint32_t to_mont(uint64_t x,uint32_t p) {
    return REDC(x * r2modp, p);
}
inline uint32_t MontMul(uint64_t a, uint64_t b,uint32_t p) {
    return REDC(a * b, p);
}
inline uint32x4_t vREDC(uint64x2_t T_vec_low, uint64x2_t T_vec_high, uint32_t p){
    // m = (T % R) * N' % R = T_low * p_inv
    uint32x4_t T_low = vcombine_u32(vmovn_u64(T_vec_low), vmovn_u64(T_vec_high));
    uint32x4_t m = vmulq_n_u32(T_low, p_inv);

    // m * N 
    uint64x2_t mN_low = vmull_n_u32(vget_low_u32(m), p);
    uint64x2_t mN_high = vmull_n_u32(vget_high_u32(m), p);

    // T + m*N
    uint64x2_t sum64_low = vaddq_u64(T_vec_low, mN_low);
    uint64x2_t sum64_high = vaddq_u64(T_vec_high, mN_high);

    // t = (T + m*N) >> 32
    uint32x2_t t_parts_low = vshrn_n_u64(sum64_low, 32);
    uint32x2_t t_parts_high = vshrn_n_u64(sum64_high, 32);
    uint32x4_t t = vcombine_u32(t_parts_low, t_parts_high);

    // result = t - p if t >= p
    uint32x4_t p_vec = vdupq_n_u32(p);
    uint32x4_t cmp = vcgeq_u32(t, p_vec);
    uint32x4_t ret = vsubq_u32(t, vandq_u32(p_vec, cmp));

    return ret;    
}
inline uint32x4_t montgomery_mulmod_neon(uint32x4_t a, uint32x4_t b, uint32_t p) {
    uint64x2_t ab_low_u64 = vmull_u32(vget_low_u32(a), vget_low_u32(b));
    uint64x2_t ab_high_u64 = vmull_u32(vget_high_u32(a), vget_high_u32(b));
    return vREDC(ab_low_u64, ab_high_u64, p);
}
void NTT(int *a,int opt,uint32_t p)
{
    //蝴蝶变换
	for(int i = 0;i < len;++ i) if(i < rev[i]) std::swap(a[i],a[rev[i]]);
    //NTT主体
    int GINV = qpow(G,p-2,p);
    int rmodp = (1ll << 32) % p, rinv = qpow((1ll << 32)%p, p-2, p);
	for(int i = 1;i < len;i <<= 1)
	{
		int w = qpow(opt == 1 ? G : GINV,(p-1) / (i << 1),p);
        uint32_t mont_w = to_mont(w, p);
		for(int j = 0,s = i << 1;j < len;j += s)
        {
            //不够长
            if(i < 4){
                int mi = 1;
                for(int k = 0;k < i;++ k,mi = 1ll * mi * w % p)
                {
                    int X = a[j+k],Y = 1ll * mi * a[i+j+k] % p;
                    a[j+k] = (X + Y) % p;
                    a[i+j+k] = (X + p - Y) % p;
                }
            }
            else{
                // 计算对应的 mi
                uint32_t mi[4] = {1, (uint32_t)w, (uint32_t)(1ll*w*w%p), (uint32_t)(1ll*w*w%p*w%p)};
                int w4 = 1ll * mi[3] * w % p;
                for(int k = 0; k < 4; ++k) {
                    mi[k] = to_mont(mi[k], p);
                }
                uint32x4_t mi_base = vdupq_n_u32(to_mont(w4, p));
                uint32x4_t mi_vec = vld1q_u32(mi);
                for(int k = 0; k < i; k += 4)
                {
                    // 加载 a[j + k] 和 a[j + k + i]
                    uint32x4_t A = vld1q_u32((uint32_t*)(a + j + k));
                    uint32x4_t B = vld1q_u32((uint32_t*)(a + j + k + i));
            
                    // Y = mi * B
                    uint32x4_t Y = montgomery_mulmod_neon(B, mi_vec, p);            
            
                    // 执行加减
                    uint32x4_t P = vdupq_n_u32(p);
                    uint32x4_t U = vaddq_u32(A, Y);            // X + Y
                    uint32x4_t V = vaddq_u32(A, P); 
                    V = vsubq_u32(V, Y);                 // X + P - Y
            
                    // 取模（简易版，确保结果在 [0, p) 范围）
                    uint32x4_t Cmp = vcgeq_u32(U, P);
                    U = vsubq_u32(U, vandq_u32(P, Cmp)); // if (U >= p) U -= p;
                    
                    Cmp = vcgeq_u32(V, P);
                    V = vsubq_u32(V, vandq_u32(P, Cmp)); // if (V >= p) V -= p;
            
                    // 存回结果
                    vst1q_u32((uint32_t*)(a + j + k), U);
                    vst1q_u32((uint32_t*)(a + j + k + i), V);

                    mi_vec = montgomery_mulmod_neon(mi_vec, mi_base, p);              
                }
		    }
	    }
    }
	int mont_invlen = to_mont(qpow(len,p-2,p), p); 
	if(opt == -1) {
        for(int i = 0;i < len;++ i) a[i] = MontMul(a[i], mont_invlen, p);
        for(int i = 0;i < len;++ i) a[i] = REDC(a[i], p); //转换回原始域
    }
}
void poly_multiply(int *a, int *b, int *ab, int n, int p){
    len = 1, l = -1;
	while(len < n+n-1) len <<= 1,l++;
	for(int i = 0;i < len;++ i) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << l);

    p_inv = (1ll<<32) - qpow(p, (1ll<<31)-1, 1ll<<32);
    rmodp = (1ll << 32) % p;
    r2modp = 1ll * rmodp * rmodp % p;
    for(int i = 0;i < n;++ i) a[i] = to_mont(a[i], p),b[i] = to_mont(b[i], p);
    NTT(a,1,p);
    NTT(b,1,p);
    for(int i = 0;i < len;++ i) ab[i] = MontMul(a[i], b[i], p);
    NTT(ab,-1,p);
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
        // std::cout << "test id: " << i << "\n";
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
