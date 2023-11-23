#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//符号のパラーメータの指定。通常[N,K,T]として、
//Nは符号の長さ、Kが符号の次元、Tは訂正エラー数
//を表す。ここではDは符号長にしている。
#define N 8191 // set small prime
#define M N // order of group
#define K (20) // degree of polynomial
#define E (14)   // bit size of prime
#define DEG N // set (K * E) < N
#define T (K / 2) // weight of error vector


unsigned short mat[N][K*E] = {0};
unsigned short ma[N][K*E] = {0};
