#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <math.h>
#include <x86intrin.h> // SIMD命令を使用するためのヘッダファイル

#include "global-p.h"
#include "struct.h"
#include "chash-p.c"

#define SEPARABLE 0
#define MATRIX_SIZE K * 2
#define SHM_KEY 128
#define MAX_SIZE N

// 行列要素の型（例：int）
typedef int matrix_element;

short g[K + 1] = {0};

// ランダム多項式の生成
static void
ginit(void)
{
    int j, count = 0, k = 0;
    short gg[K + 1] = {0};

    printf("in ginit\n");

    g[K] = 1;          // xor128();
    g[0] = rand() % N; // or N
    k = rand() % (K - 1);
    if (k > 0)
    {
        while (count < k)
        {
            printf("in whule\n");
            j = rand() % (K);
            if (j < K && j > 0 && g[j] == 0)
            {
                g[j] = rand() % N; // or N;
                count++;
            }
        }
    }

    for (j = 0; j < K + 1; j++)
        gg[j] = g[K - j];

    memcpy(g, gg, sizeof(g));
}

// OP型からベクトル型への変換
vec o2v(OP f)
{
    vec a = {0};
    int i;

    for (i = 0; i < K * E; i++)
    {
        if (f.t[i].a > 0 && f.t[i].n < K * E)
            a.x[f.t[i].n] = f.t[i].a;
    }

    return a;
}

// ベクトル型からOP型への変換
OP v2o(vec a)
{
    int i, j = 0;
    OP f = {0};

    // #pragma omp parallel for
    for (i = 0; i < K * E; i++)
    {
        if (a.x[i] > 0)
        {
            f.t[j].n = i;
            f.t[j++].a = a.x[i];
        }
    }

    return f;
}

short oinv(short a, short n)
{
    short i;

    if (a == 0 || a % N == 0)
        return 0;
    if (a < 0)
        a += N;
    // if (a == 1)
    //     return 1;
    for (i = 1; i < n; i++)
    {
        if ((i * a) % N == 1)
            return i;
    }
    printf("no return %d\n", a);
    exit(1);
}

// 多項式の次数(default)
int deg(vec a)
{
    int i, n = 0, flg = 0;

    // #pragma omp parallel for
    for (i = 0; i < DEG; i++)
    {
        if (a.x[i] > 0)
        {
            n = i;
            flg = 1;
        }
    }
    if (flg == 0)
        return 0;

    return n;
}

// 多項式を表示する(default)
void printpol(vec a)
{
    int i, n;

    n = deg(a);

    // printf ("baka\n");
    //  assert(("baka\n", n >= 0));

    for (i = n; i > -1; i--)
    {
        if (a.x[i] > 0)
        {
            printf("%u*", a.x[i]);
            // if (i > 0)
            printf("x^%d", i);
            // if (i > 0)
            printf("+");
        }
    }
    //  printf("\n");

    return;
}

vec kof2(short c, vec f)
{
    int i, k;
    vec b = {0}, h = {0};

    c = oinv(c, N);
    printf("c=%d\n", c);
    // exit(1);
    b = f; // o2v(f);
    k = deg(b);
    printpol(b);
    printf(" =b debugi\n");
    for (i = 0; i < k + 1; i++)
    {
        h.x[i] = (c * b.x[i]) % N;
    }
    // g = v2o(h);
    printpol(h);
    printf(" =h in oinv2\n");
    return h;
}

vec vadd(vec a, vec b)
{
    int i;
    vec c = {0};

    // printf("deg=%d %d\n",deg(a),deg(b));

    for (i = 0; i < DEG; i++)
        c.x[i] = (a.x[i] + b.x[i]) % N;

    return c;
}

vec lsft(vec a)
{
    vec b = {0};
    int o = deg(a);

    for (int i = 0; i < o + 1; i++)
    {
        b.x[i + 1] = a.x[i];
    }
    // b.x[K*2]=0;

    return b;
}

vec rsft(vec a)
{
    vec b = {0};
    int o = deg(a);

    for (int i = 0; i < o + 1; i++)
        b.x[i] = a.x[i + 1];
    // b.x[0]=0;

    return b;
}

int mul = 0, mul2 = 0;
vec vmul(vec a, vec b)
{
    int i, j, k, l;
    vec c = {0};

    k = deg(a);
    l = deg(b);

    i = 0;
    while (i < k + 1)
    {
        for (j = 0; j < l + 1; j++)
        {
            if (a.x[i] > 0)
                c.x[i + j] = (c.x[i + j] + a.x[i] * b.x[j]) % N;
        }
        i++;
    }

    return c;
}

short vb[K * 2][N] = {0};
short gt[K * 2][K * 2] = {0};

void van(int kk)
{
    int i, j;

    printf("van der\n");

    for (i = 0; i < N; i++)
    {
        mat[i][0] = vb[0][i] = 1;
        printf("%d,", vb[0][i]);
    }
    printf("\n");

    // #pragma omp parallel for private(i, j)
    for (i = 1; i < kk; i++)
    {
        for (j = 0; j < N; j++)
        {
            vb[i][j] = mltn(i, j);
            printf("g%d,", vb[i][j]);
            mat[j][i] = vb[i][j];
        }
        printf("\n");
    }
}

void ogt(int kk)
{
    int i, j;

    // #pragma omp parallel for private(i, j)
    for (i = 0; i < kk; i++)
    {
        for (j = 0; j < kk - i; j++)
        {
            gt[i][j + i] = g[j];
        }
    }
    for (i = 0; i < kk; i++)
    {
        for (j = 0; j < kk; j++)
            printf("h%d,", gt[i][j]);
        printf("\n");
    }
    // exit(1);
}

// 配列の値を係数として多項式に設定する
vec setpol(short f[], int n)
{
    vec g;
    vec v = {0};
    int i;

    for (i = 0; i < n; i++)
    {
        v.x[n - 1 - i] = f[i];
    }

    g = (v);

    return g;
}

vec mkpol()
{
    int i, j, k, flg, ii = 0;
    vec w = {0};

    do
    {
        // fail = 0;
        j = 0;
        k = 0;
        flg = 0;
        // l = 0;
        memset(g, 0, sizeof(g));
        // memset(ta, 0, sizeof(ta));
        memset(w.x, 0, sizeof(w));
        ginit();
        ii++;
        if (ii > 100)
        {
            printf("erro=%d\n", ii);
            exit(1);
        }

        for (i = 0; i < K; i++)
        {
            if (g[K - 1] > 0)
                flg = 1;
            if (i % 2 == 1 && g[i] > 0 && i < K)
                k++;
        }

        // 偶数項だけにならないようにする
        if ((k > 0 && flg == 0) || (k > 1 && flg == 1))
        // if(k>0)
        {
            w = setpol(g, K + 1);
            j = 1;
            // if(isquad(w)==-1)
            // exit(1);
        }
        // exit(1);

    } while (j == 0);

    printpol((w));
    printf(" ==g\n");
    // exit(1);

    return w;
}

short v2a(oterm a)
{
    int j;

    if (a.a == 0)
        return 0;

    // printf("aa=%d\n",a.a);
    for (j = 0; j < M; j++)
    {
        if (j == a.a && a.a > 0)
        {
            // printf("j==%d\n",j);
            return j - 1;
        }
    }
    return 0;
}

void printsage(vec a)
{
    int i, j;
    oterm b;

    printf("poly=");
    for (i = 0; i < DEG; i++)
    {
        if (a.x[i] > 0)
        {
            b.a = a.x[i];
            b.n = i;
            j = v2a(b);
            printf("%d*X**%d+", b.a, i); // for GF(2^m)
        }
    }
}

// 多項式の代入値
short trace(vec f, short x)
{
    short u = 0;
    vec v = (f);
    int d = deg((v)) + 1;

    for (int i = 0; i < d; i++)
    {
        if (v.x[i] > 0)
            u = (u + (v.x[i] * mltn(i, x))) % N;
    }

    return u;
}

// リーディグタームを抽出(default)
oterm vLT(vec f)
{
    int i;
    oterm t = {0};

    // k = deg (o2v (f));
    for (i = 0; i < DEG; i++)
    {
        // printf("a=%d %d\n",f.t[i].a,f.t[i].n);
        if (f.x[i] > 0)
        {
            t.n = i;
            t.a = f.x[i];
        }
    }

    return t;
}

short inv(short a, short n)
{
    short d = n;
    short x = 0;
    short s = 1;
    while (a != 0)
    {
        short q = d / a;
        short r = d % a;
        d = a;
        a = r;
        short t = x - q * s;
        x = s;
        s = t;
    }
    short gcd = d; // $\gcd(a, n)$

    return ((x + n) % (n / d));
}

// aに何をかけたらbになるか
short equ(short a, short b)
{
    // for(short i=0;i<N;i++)
    if (b == 0)
        return 0;
    if (a == 1)
        return b;

    return (inv(a, N) * b) % N;
}

// 多項式を単行式で割る
oterm vLTdiv(vec f, oterm t)
{
    oterm tt = {0}, s = {
                        0};

    tt = vLT(f);
    if (tt.n < t.n)
    {
        s.n = 0;
        s.a = 0;
    }
    else if (tt.n == t.n)
    {
        s.n = 0;
        s.a = equ(t.a, tt.a);
    }
    else if (tt.n > t.n)
    {
        s.n = tt.n - t.n;
        s.a = equ(t.a, tt.a);
        // printf("%u\n",s.a);
    }
    else if (t.n == 0 && t.a > 0)
    {
        s.a = (tt.a * inv(t.a, N)) % N;
        s.n = tt.n;
    }

    return s;
}

// 多項式を項ずつ掛ける
vec vterml(vec f, oterm t)
{
    // f = conv(f);
    // ssert(op_verify(f));
    int i;
    vec h = {0};

    // f=conv(f);
    // k = deg (o2v(f));

    for (i = 0; i < DEG; i++)
    {
        // h.t[i].n = f.t[i].n + t.n;
        if (f.x[i] > 0)
            h.x[i + t.n] = (f.x[i] * t.a) % N;
    }

    // h = conv(h);
    //  assert(op_verify(h));
    return h;
}

// 20200816:正規化したいところだがうまく行かない
// 多項式の足し算
vec vsub(vec a, vec b)
{
    vec c = {0};
    // int i, j, k, l = 0;
    vec h = {0}, f2 = {0}, g2 = {0};

    for (int i = 0; i < DEG; i++)
    {
        if (a.x[i] >= b.x[i])
            c.x[i] = (a.x[i] - b.x[i]) % N;
        if (a.x[i] < b.x[i])
            c.x[i] = (N + a.x[i] - b.x[i]) % N;
    }

    return c;
}

int vm = 0;
// 多項式の剰余を取る
vec vmod(vec f, vec g)
{
    vec h = {0};
    oterm b = {0}, c = {0};

    if (deg(g) == 0)
        return g;
    vm++;
    // printf("vmod-bl=%d k=%d\n",deg(f),deg(g));
    if (vLT(f).n < vLT(g).n)
    {
        //    exit(1);
        return f;
    }

    b = vLT(g);

    // printpol(f);
    // printf(" ==f\n");
    while (1)
    {
        // printf("@\n");
        c = vLTdiv(f, b);
        h = vterml(g, c);
        f = vsub(f, h);
        // printsage(g);
        if (deg((f)) == 0 || deg((h)) == 0 || c.n==0)
        {
            break;
        }
    }
    // printf("vmod-baka== %d %d\n",deg(f),deg(g));
    return f;
}

// int mul = 0, mul2 = 0;
vec vmul_2(vec a, vec b)
{
    int i, j, k, l;
    vec c = {0};
    if (deg(a) > 128 && deg(b) > 128)
        mul++;
    mul2++;

    k = deg(a);
    l = deg(b);

    for (i = 0; i < k + 1; i++)
    {
        for (j = 0; j < l + 1; j++)
        // if (a.x[i] > 0)
        {
            c.x[i + j] += (a.x[i] * b.x[j]) % N;
            c.x[i + j] %= N;
        }
    }

    return c;
}

// 多項式のべき乗
vec opow(vec f, int n)
{
    // int i;
    vec g = {0};

    g = f;

    for (int i = 1; i < n; i++)
        g = vmul(g, f);

    return g;
}

vec table[128]={0};
vec vpowmod(vec f, vec mod, int n)
{
    vec ret = {0};

    ret.x[0] = 1;
    while (n > 0)
    {
        if (n % 2 == 1){
        ret = vmod(vmul(ret, f), mod); // n の最下位bitが 1 ならば x^(2^i) をかける
        }
        f = vmod(vmul(f, f), mod);
        n >>= 1; // n を1bit 左にずらす
    }
    return ret;
}

// gcd
vec ogcd(vec xx, vec yy)
{
    vec tt = {0}, tmp, h = {0};
    // ee.x[K] = 1;

    h.x[0] = 1;
    // h.x[0] = 0;
    if (deg((xx)) < deg((yy)))
    {
        tmp = xx;
        xx = yy;
        yy = tmp;
    }
    // tt = vmod(xx, yy);
    tt = vmod(xx, yy);
    while (deg(tt) > 0)
    {
        // printf("Oh!\n");
        xx = yy;
        yy = tt;
        if (deg(yy) > 0)
        {
            tt = vmod(xx, yy);
        }
        if (vLT(tt).a == 0)
            return yy;
    }
    if (vLT(yy).a == 0)
    {
        return tt;
    }
    else
    {
        return h;
    }
    //  return yy;
}

short diag(MTX a, int n)
{
    return (a.x[n][n] * a.x[n + 1][n + 1] - a.x[n][n + 1] * a.x[n + 1][n]) % N;
}

MTX mkmat(vec f, vec g)
{
    MTX a = {0};

    /*
    a.x[0][0] = 2; a.x[0][1] = 1; a.x[0][2] = 5; a.x[0][3] = 3;
    a.x[1][0] = 3; a.x[1][1] = 0; a.x[1][2] = 1; a.x[1][3] = 6;
    a.x[2][0] = 1; a.x[2][1] = 4; a.x[2][2] = 3; a.x[2][3] = 3;
    a.x[3][0] = 8; a.x[3][1] = 2; a.x[3][2] = 0; a.x[3][3] = 1;
    */
    int n = deg(f), m = deg(g);

    /*
    f.x[0]=16;
    f.x[1]=0;
    f.x[2]=4;
    f.x[3]=4;
    f.x[4]=1;
    g.x[0]=8;
    g.x[1]=9;
    g.x[2]=10;
    g.x[3]=9;
    printf("\n");
    */

    if (n < m)
    {
        for (int i = 0; i < n + 1; i++)
        {
            for (int j = 0; j < m + 1; j++)
            {
                a.x[i + j][i] = f.x[n - j];
            }
        }
        for (int i = 0; i < n + m; i++)
        {
            for (int j = 0; j < n + m; j++)
            {
                a.x[j + i][i + m] = g.x[m - j];
            }
        }
    }
    if (n >= m)
    {
        for (int i = 0; i < m + 1; i++)
        {
            for (int j = 0; j < n + 1; j++)
            {
                a.x[i + j][i] = f.x[n - j];
            }
        }
        for (int i = 0; i < n + m + 1; i++)
        {
            for (int j = 0; j < n + m + 1; j++)
            {
                a.x[j + i][i + m] = g.x[m - j];
            }
        }
    }

    return a;
}

int resl(vec f, vec g)
{
    MTX a = {0};
    short dia[N] = {0};

    int m = deg(f);
    int n = deg(g);

    a = mkmat(f, g);

    short tmp[N] = {0};
    int i, j, k, t;

    for (i = 0; i < m + n - 1; i++)
    {

        for (k = i; k < m + n - 1; k++)
        { // m+n
            t = a.x[k + 1][i];
            for (int j = i; j < m + n; j++)
            {
                tmp[j] = a.x[k + 1][j] - (a.x[i][j] * equ(a.x[i][i], a.x[k + 1][i])) % N; // equ(a,b)は素体上でaに何をかけたらbになるか、その答えを返します
            }
            for (int j = 0; j < n + m; j++)
            {
                a.x[k + 1][j] = tmp[j];
                if (a.x[k + 1][j] < 0)
                    a.x[k + 1][j] = N + a.x[k + 1][j];
            }
        }
    }

    for (int i = 0; i < n + m; i++)
        dia[i] = a.x[i][i];
    int y = diag(a, n + m - 2);

    for (int i = 0; i < n + m - 2; i++)
    {
        y = (y * dia[i]) % N;
        if (dia[i] == 0)
            return -1;
    }
    /*
    if(y!=det){
        printf("no=%d %d\n",det,y);
        exit(1);
    }
    */

    if (y > 0)
        return 0;
    if (y == 0)
        return -1;

    printf("no return in resl\n");
    exit(1);
}

int cnty = 0;
vec vpp(vec f, vec mod, int n)
{
    int i;
    vec s = {0};
    // t = f;
    s = f;
    printf("@\n");
    // 繰り返し２乗法
    for (i = 1; i < n; i++)
    {
        s = vmod(vmul_2(s, f), mod);
    }

    return s;
}

// GCD for decode
vec vgcd(vec xx, vec yy)
{
    vec tt;

    while (deg(yy) > 0)
    {
        tt = vmod(xx, yy);
        xx = yy;
        yy = tt;
    }
    if (yy.x[0] > 0)
        tt = kof2(yy.x[0], xx);
    printpol((yy));
    printf(" =========yy\n");
    printpol((tt));
    printf(" =========tt\n");

    return tt;
}

// 行列の逆行列を計算する関数
vec inverseMatrix(MTX A, MTX A_inv, int start_row, int end_row)
{
    int i, j, k;
    short temp;

    // 単位行列を初期化
    for (i = 0; i < K / 2; i++)
    {
        for (j = 0; j < K / 2 + 1; j++)
        {
            A_inv.x[i][j] = (i == j) ? 1 : 0;
        }
    }

    // ガウス・ジョルダン法による逆行列の計算
    for (k = start_row; k < end_row; k++)
    {
        temp = A.x[k][k];
        for (j = 0; j < K / 2 + 1; j++)
        {
            A.x[k][j] = A.x[k][j] * oinv(temp, N);
            A_inv.x[k][j] = A_inv.x[k][j] * oinv(temp, N) % N;
        }
        for (i = start_row; i < end_row; i++)
        {
            if (i != k)
            {
                temp = A.x[i][k];
                for (j = 0; j < K / 2 + 1; j++)
                {
                    A.x[i][j] -= (A.x[k][j] * temp) % N;
                    A_inv.x[i][j] -= (A_inv.x[k][j] * temp) % N;
                }
            }
        }
    }
    vec x = {0};
    for (i = 0; i < K / 2; i++)
    {
        if (N > A.x[i][K / 2])
        {
            x.x[K / 2 - i] = (N - A.x[i][K / 2]) % N;
        }
        else
        {
            x.x[K / 2 - i] = A.x[i][K / 2] % N;
        }
    }

    x.x[0] = 1;

    vec vv = {0};
    vec pol = {0};
    pol = setpol(x.x, K / 2 + 1);
    printpol((pol));
    printf(" ==key\n");
    for (i = 0; i < N; i++)
    {
        // v.x[i] = 0;
        if (trace(pol, i) % N == 0)
        {
            printf("error position=%d\n", i);
            vv.x[i] = i + 1;
        }
    }
    for (i = 0; i < K / 2; i++)
    {
        for (j = 0; j < K / 2 + 1; j++)
            printf("%d,", A.x[i][j]);
        printf("\n");
    }
    return vv;
    // exit(1);
}

// #define NN 16
vec sol(MTX a, int start, int end)
{
    int p, d;
    int i, j, k;
    vec v = {0};

    for (i = start; i < end; i++)
    {
        p = a.x[i][i];

        for (j = 0; j < (K / 2 + 1); j++)
        {
            a.x[i][j] = (a.x[i][j] * inv(p, N)) % N;
        }

        for (j = 0; j < K / 2; j++)
        {
            if (i != j)
            {
                d = a.x[j][i];

                for (k = i; k < (K / 2 + 1); k++)
                {
                    if (a.x[j][k] > (d * a.x[i][k]) % N)
                    {
                        a.x[j][k] -= (d * a.x[i][k]) % N;
                    }
                    else
                    {
                        a.x[j][k] = (N + (a.x[j][k] - (d * a.x[i][k]) % N)) % N;
                    }
                }
            }
        }
    }
    vec x = {0};
    for (i = start; i < end; i++)
    {
        if (N > a.x[i][K / 2])
        {
            x.x[K / 2 - i] = (N - a.x[i][K / 2]) % N;
        }
        else
        {
            x.x[K / 2 - i] = a.x[i][K / 2] % N;
        }
    }

    x.x[0] = 1;

    vec vv = {0};
    vec pol = {0};
    pol = setpol(x.x, K / 2 + 1);
    printpol((pol));
    printf(" ==key\n");
    for (i = 0; i < N; i++)
    {
        // v.x[i] = 0;
        if (trace(pol, i) % N == 0)
        {
            printf("error position=%d\n", i);
            vv.x[i] = i + 1;
        }
    }

    return vv;
}

int is_equ(vec a, vec b)
{
    for (int i = 0; i < N * N; i++)
        if (a.x[i] != b.x[i])
            return -1;

    return 0;
}



void win(vec f,vec mod)
{
    // テーブルの生成
    //table[0].x[0]=1;
    table[0]=f;
    vec ret=f;
    for (int i = 2; i < 13;i++) {
        f = vmod(vmul(f,f),mod);
        table[i]=f;
    }

}

vec fpowmod(int n){
    vec ret={0};
    int c=0;

    while (n > 0)
    {
        if (n % 2 == 1)
            ret = vadd(ret, table[c]); // n の最下位bitが 1 ならば x^(2^i) をかける
        n >>= 1; // n を1bit 左にずらす
        c++;
        //printsage(ret);
        //printf("damn %d\n",c);
    }

return ret;
}

// GF(2^m) then set m in this function.
int ben_or(vec f)
{
    int n; //, pid;
    int count = 0;

    vec s = {0}, u = {0}, r = {0};
    vec v = {0}; //, ff=o2v(f);
    // if GF(8192) is 2^m and m==13 or if GF(4096) and m==12 if GF(16384) is testing
    // int m = E;
    //  m=12 as a for GF(4096)=2^12 defined @ gloal.h or here,for example m=4 and GF(16)
    for(int i=0;i<128;i++)
    table[i]=u;

    v.x[1] = 1;
    s = (v);
    // for (int i = 0; i < K / 2; i++)
    r = s;
    n = deg((f));

    if (vLT(f).n == 0)
    {
        printf("f==0\n");
        exit(1);
    }
    if (n == 0)
        return -1;

    // r(x)^{q^i} square pow mod
    for (int i = 0; i < K / 2; i++)
    {
        printf(":i=%d", i);
        int l=1;
        //for(int jj=0;jj<i+1;jj++){
        //l=13*l%(K+1);
        //printf("jj=%d %d\n",l,jj);
        //}
        //win(v,f);
        memset(r.x, 0, sizeof(r.x));
        //exit(1);
        v = vpowmod(v, f, N);

        //v=fpowmod(N);
        r = v;
        //r.x[l]=1;

        r = vsub(r, (s));

        u = vmod(r, f);

        int le = 0;
        if (deg(u) > 0)
        {
            // printsage(u);
            // printf(" you\n");
            // printsage(f);
            printf(" me\n");
            //le = resl(f, r);
            //if (le == 0)
            //    count++;
             u = ogcd(f, r);
            //if (le == -1)
             //   return -1;
        }
        else
        {
            return -1;
        }

        //if (le == -1){
            if(deg(u) > 0){
            {
                // flg[i]= -1;
                printf("ae\n");
                return -1;
            }
        }
    }
    // if(count==T-1)
    return 0;
}

vec mkd(vec w, int kk, int start, int end)
{
    int i, j, k, l, ii = 0;

    short tr[N] = {0};
    short ta[N] = {0};
    vec v = {0}, ff = {0}, tt = {0};
    short po[K + 1] = {1, 0, 1, 0, 5};
    // vec w={0};
    vec r = {0};
    
    ff.x[1]=1;
aa:

    // printf("\n");
    memset(mat, 0, sizeof(mat));
    // 既約性判定のためのBen-Orアルゴリズム。拡大体にも対応している。デフォルトでGF(8192)
    // 既約多項式しか使わない。

    l = 0;
    ii = 0;
    // irreducible gvecpa code (既役多項式が必要なら、ここのコメントを外すこと。)

    w = mkpol();
    win(ff,w);
    l = ben_or((w));
    while (l == -1)
        goto aa;
    printsage((w));
    printf("\n");
    //exit(1);
    //     printf("wwwwwww\n");
    //  exit(1);
    //  separable gvecpa code
    //  w = mkpol();
    r = (w);
    //  r=vmul(w,w);
    memset(ta, 0, sizeof(ta));
    // w = setpol(g, K + 1);
    printpol((r));
    printf(" =poly\n");
    // exit(1);

    // 多項式の値が0でないことを確認
    for (int i = start; i < M; i++)
    {
        ta[i] = trace(w, i);
        if (ta[i] == 0)
        {
            printf("eval 0 @ %d\n", i);
            // fail = 1;
            // exit(1);
            goto aa;
        }
    }
    for (int i = start; i < M; i++)
    {
        tr[i] = inv(ta[i], N);
        // printf("%d,", tr[i]);
    }
    memset(g, 0, sizeof(g));
    // g[0] = 1;

    // 多項式を固定したい場合コメントアウトする。
    printpol(r);
    printf("\n");
    printsage((r));
    printf("\n");
    printf("sagemath で既約性を検査してください！\n");
    memset(v.x, 0, sizeof(v.x));
    //  v=rev(w);
    van(kk);
    //  v=(w);
    ogt(kk);
    // exit(1);
    //  wait();

    // #pragma omp parallel for

    printf("\nすげ、オレもうイキそ・・・\n");
    // keygen(g);
    // exit(1);

    for (int j = start; j < K; j++)
    {
        for (int i = 0; i < M; i++)
        {
            ma[i][j] = (vb[j][i] * tr[i]) % N;
        }
    }

    for (int i = start; i < K; i++)
    {
        for (int j = 0; j < M; j++)
        {
            for (int k = 0; k < K; k++)
            {
                mat[j][i] = (mat[j][i] + (gt[k][i] * ma[j][k])) % N;
            }
            printf("c%d,", ma[j][i]);
        }
        printf("\n");
    }


    return (w);
}

void vv(int kk)
{
    int i, j;
    vec r = mkpol();
    short tr[N];
    short ta[N] = {0};

    printf("van der\n");

    for (i = 0; i < kk; i++)
    {
        for (j = 0; j < N; j++)
        {
            vb[i][j] = mltn(i, j);
        }
        // printf("\n");
    }

    int l = -1;
    vec pp = {0}, tt = {0};

aa:
    // exit(1);
    r = mkpol();

    for (i = 0; i < N; i++)
    {
        ta[i] = trace(r, i);
        if (ta[i] == 0)
        {
            printf("trace 0 @ %d\n", i);
            // fail = 1;
            goto aa;
        }
    }

    for (i = 0; i < N; i++)
    {
        tr[i] = inv(ta[i], N);
        // printf("%d,", tr[i]);
    }

    printf("\nすげ、オレもうイキそ・・・\n");
    // keygen(g);
    // exit(1);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < kk; j++)
        {
            mat[i][j] = (vb[j][i] * tr[i]) % N;
        }
    }
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < N; j++)
            printf("c%d,", mat[j][i]);
        printf("\n");
    }
}

void mkerr(short *z1, int num)
{
    int j, l;

    j = 0;

    memset(z1, 0, sizeof(2 * N));

    while (j < num)
    {
        l = rand() % (N - 1);
        // printf ("l=%d\n", l);
        if (0 == z1[l] && l > 0)
        {
            z1[l] = 2;
            // printf("l=%d\n", l);
            j++;
        }
    }
}

vec synd(short zz[], int kk)
{
    short syn[K] = {0}, s = 0;
    int i, j;
    vec f = {0};

    printf("in synd2\n");

    for (i = 0; i < kk; i++)
    {
        syn[i] = 0;
        s = 0;
        // #pragma omp parallel num_threads(16)
        for (j = 0; j < N; j++)
        {
            s = (s + (zz[j] * mat[j][i])) % N;
        }
        syn[i] = s;
        // printf ("syn%d,", syn[i]);
    }
    // printf ("\n");

    f = setpol(syn, kk);
    printpol((f));
    printf(" syn=============\n");
    //  exit(1);

    return f;
}

// chen探索
vec chen(vec f)
{
    vec e = {0};
    int i, n, x = 0, count = 0;
    short z;

    n = deg((f));
    for (x = 0; x < N; x++)
    {
        z = 0;
        for (i = 0; i < n + 1; i++)
        {
            if (f.x[i] > 0)
                z += (mltn(i, x) * f.x[i]) % N;
        }
        if (z % N == 0)
        {
            e.x[count] = x;
            count++;
            printf("change %d\n", (x));
        }
    }

    return e;
}

typedef struct
{
    vec f;
    vec g;
    vec h;
} ymo;

vec pmul(vec a, vec b)
{
    int i, j, k, l;
    vec c = {0};

    k = deg(a) + 1;
    l = deg(b) + 1;
    printf("k=%d,l=%d", k, l);
    // exit(1);
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < l; j++)
            if (a.x[i] > 0)
            {
                c.x[i + j] = (c.x[i + j] + a.x[i] * b.x[j]) % N;
                // printf("%d=c ",c.x[i+j]);
            }
        // printf("\n");
    }
    /*
    printf("\n");
    printpol(v2o(c));
    printf(" ==c\n");
    printpol(v2o(a));
    printf(" ==a\n");
    printpol(v2o(b));
    printf(" ==b\n");
    // exit(1);
    */
    return c;
}


// 行列の掛け算関数
void matrix_multiply(short A[MATRIX_SIZE][MATRIX_SIZE], short B[MATRIX_SIZE][MATRIX_SIZE], short *C, int start_row, int end_row)
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            short sum = 0.0;
            for (int k = 0; k < MATRIX_SIZE; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i * MATRIX_SIZE + j] = sum;
        }
    }
}

int main()
{
    int i;
    short s[K + 1] = {0}, z1[N] = {0};
    vec v = {0}, x = {0};
    vec f = {0};

    srand(clock());
    // mkg(K); // Goppa Code (EEA type)
    // van(K); // RS-Code generate
    // vv(K);           // Goppa Code's Parity Check (Berlekamp type)

    // resl(v,x);
    // exit(1);

    mkd(f, K, 0, K);

    while (1)
    {
        memset(z1, 0, sizeof(z1));
        // mkerr(z1, T);    // generate error vector
        for (int i = 0; i < T; i++)
            z1[i] = i + 1;
        f = synd(z1, K); // calc syndrome
        x = (f);         // transorm to vec
        // for(i=0;i<N;i++)
        // if(z1[i]>0)
        // printf("i=%d\n",i);

        MTX b = {0};

        for (i = 0; i < K; i++)
            v.x[K - 1 - i] = x.x[i];
        printpol(v);
        printf(" ==synpol\n");
        // exit(1);

        for (i = 0; i < K / 2; i++)
        {
            for (int j = 0; j < K / 2 + 1; j++)
            {
                b.x[i][j] = v.x[i + j];
                // printf("%d,",b.x[i][i+j]);
            }
            // printf("\n");
        }
        printf("\n");
        for (i = 0; i < K / 2; i++)
        {
            for (int j = 0; j < K / 2 + 1; j++)
                printf("e%d,", b.x[i][j]);
            printf("\n");
        }

        x = sol(b, 0, K / 2);
        for (i = 0; i < N; i++)
        {
            if (z1[i] != x.x[i] && z1[i] > 0 && x.x[i])
            {
                printf("baka=%d %d %d\n", i, z1[i], x.x[i]);
                // exit(1);
            }
        }

        int flg = 0;
        for (i = 0; i < N; i++)
        {
            if (z1[i] > 0 && x.x[i] > 0)
            {
                printf("(correcting ,original) = (%d ,%d)\n", x.x[i], z1[i]);
                flg++;
            }
        }
        if (flg == T)
            break;
        // printf("\n");
        break;
    }
    return 0;
}
