#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <pthread.h>
// 可以自行添加需要的头文件

typedef long long LL;
const int G = 3;

void fRead(LL *a, LL *b, int *n, LL *p, int input_id){
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

void fCheck(LL *ab, int n, int input_id){
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
        LL x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(LL *ab, int n, int input_id){
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
//光速防爆乘法
__int128 gsc(__int128 x,__int128 y,__int128 p)
{
	__int128 ret = x*y - (__int128)((long double)x/p*y+0.5)*p;
	return (ret%p+p)%p;
} 
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
void NTT(int *a,int opt,int p)
{
    //蝴蝶变换
	for(int i = 0;i < len;++ i) if(i < rev[i]) std::swap(a[i],a[rev[i]]);
    //NTT主体
    int GINV = getinv(G,p);
	for(int i = 1;i < len;i <<= 1)
	{
		int w = qpow(opt == 1 ? G : GINV,(p-1) / (i << 1),p);
		for(int j = 0,s = i << 1;j < len;j += s)
		{
			int mi = 1;
			for(int k = 0;k < i;++ k,mi = 1ll * mi * w % p)
			{
				int X = a[j+k],Y = 1ll * mi * a[i+j+k] % p;
				a[j+k] = (X + Y) % p;
				a[i+j+k] = (X - Y + p) % p;
			}
		}
	}
	if(opt == -1) {
	    int invlen = getinv(len,p);
        for(int i = 0;i < len;++ i) a[i] = 1ll * a[i] * invlen % p;
    }
}

typedef struct{
    int *a, *b, *ab;
    int n, p, tid, thread_num;
}ThreadArg;

// 四个模数 479*2^21+1,7*2^26+1, 997*2^20+1, 119*2^23+1，原根均为3
const int MOD[4] = {1004535809, 469762049, 1045430273, 998244353}; 
int ans[4][300000]; // 存储每个模数下的结果
int A[4][300000], B[4][300000]; // 存储每个模数下的输入多项式
__int128 tmp[300000]; // 用于存储临时结果

void* poly_multiply_thread(void *arg){
    //poly_mul
    ThreadArg* t = (ThreadArg*)arg;
    int *a = t->a,*b = t->b,*ab = t->ab,tid = t->tid, thread_num = t->thread_num;
    int n = t->n, p = t->p;
    NTT(a,1,p);
    NTT(b,1,p);
    for(int i = 0;i < len;++ i) ab[i] = 1ll * a[i] * b[i] % p;
    NTT(ab,-1,p);
    //CRT
    __int128 M = 1;
    for(int i = 0;i < thread_num;++ i)
        M *= MOD[i];
    __int128 Mi = M / MOD[tid];
    int inv = getinv(Mi % MOD[tid], MOD[tid]);
    for(int i = 0; i < len; ++i) {
        tmp[i] = (tmp[i] + gsc(1ll * inv * ans[tid][i], Mi, M)) % M;
    }
    return NULL;
}  

// 中国剩余定理的多项式乘法
void CRT_poly_multiply(LL *a, LL *b, LL *ab, int n, LL p,int thread_num = 4){
    len = 1, l = -1;
	while(len < n+n-1) len <<= 1,l++;
	for(int i = 0;i < len;++ i) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << l);
    pthread_t threads[64];
    ThreadArg args[64];
    for(int i = 0;i < len;++ i)
        tmp[i] = 0; // 初始化临时结果数组
    for(int i = 0; i < thread_num; ++i){
        for(int j = 0; j < len; ++j){
            if(j < n){
                A[i][j] = a[j] % MOD[i];
                B[i][j] = b[j] % MOD[i];
            }
            else{
                A[i][j] = 0;
                B[i][j] = 0;
            }
        }
    }
    for(int i = 0; i < thread_num; ++i){
        args[i] = {A[i], B[i], ans[i], n, MOD[i], i, thread_num};
        pthread_create(&threads[i], NULL, poly_multiply_thread, &args[i]);
    }
    for(int i = 0; i < thread_num; ++i){
        pthread_join(threads[i], NULL);
    }
    //CRT主体
    for(int i = 0;i < len;++ i)
        ab[i] = tmp[i] % p;
}

LL a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
    //28,358,116,740,142,657,441,886,131,716,097
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 1337006139375617(并非263882790666241)
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 4;
    for(int i = test_begin; i <= test_end; ++i){
        std::cout << "test id: " << i << "\n";
        long double ans = 0;
        int n_; LL p_;
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        // poly_multiply(a, b, ab, n_, p_);
        CRT_poly_multiply(a, b, ab, n_, p_);
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