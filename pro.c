#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <math.h>
#include <x86intrin.h> // SIMD命令を使用するためのヘッダファイル

#define MATRIX_SIZE 16
#define SHM_KEY 1234
#define N 17


typedef struct {
    short x[N][N];
  int row; //行
  int col; //列
} MTX;


// 行列を表示する関数
void print_matrix(short A[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }
}

unsigned short oinv(unsigned short a, unsigned short n)
{
    unsigned short i;

    if (a == 0)
        return 0;
    // if (a == 1)
    //     return 1;
    for (i = 1; i < n; i++)
    {
        if ((i * a) % N == 1)
            return i;
    }
    printf("no return\n");
    exit(1);
}


// 行列の掛け算関数
void matrix_multiply(short A[MATRIX_SIZE][MATRIX_SIZE], short B[MATRIX_SIZE][MATRIX_SIZE], short *C, int start_row, int end_row) {
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            int sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += A[i][k] * B[k][j]%N;
            }
            C[i*MATRIX_SIZE+j] = sum%N;
        }
    }
}

/*
int matmul_simd(short matrixA[MATRIX_SIZE][MATRIX_SIZE],short matrixB[MATRIX_SIZE][MATRIX_SIZE],short *resultMatrix,int start_row,int end_row) {
    //short resultMatrix[MATRIX_SIZE][MATRIX_SIZE] = {0};

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            __m128d sum = _mm_setzero_pd();

            for (int k = 0; k < MATRIX_SIZE; k++) {
                __m128d a = _mm_set1_pd(matrixA[i][k]);
                __m128d b = _mm_loadu_pd(&matrixB[k][j]);
                sum = _mm_add_pd(sum, _mm_mul_pd(a, b));
            }

            _mm_storeu_pd(&resultMatrix[i*MATRIX_SIZE+j], sum);
        }
    }

    return 0;
}


void matrix_inverse_simd(short A[MATRIX_SIZE][MATRIX_SIZE], short result[MATRIX_SIZE][MATRIX_SIZE]) {
    short pivot[MATRIX_SIZE];
    for (int i = 0; i < MATRIX_SIZE; i++) {
        pivot[i] = -1.0;
    }

    // 1. データ型を__m128dに変更し、SIMDレジスタを使用する
    __m128d one = _mm_set1_pd(1.0);

    for (int col = 0; col < MATRIX_SIZE; col++) {
        int pivot_row = -1;
        short max_value = 0.0;

        for (int row = 0; row < MATRIX_SIZE; row++) {
            if (pivot[row] != -1.0) continue;

            short val = fabs(A[row][col]);
            if (val > max_value) {
                max_value = val;
                pivot_row = row;
            }
        }

        if (pivot_row == -1) {
            fprintf(stderr, "Matrix is singular.\n");
            return;
        }

        pivot[pivot_row] = col;

        // Scale the pivot row
        short pivot_value = A[pivot_row][col];
        for (int j = 0; j < MATRIX_SIZE; j++) {
            A[pivot_row][j] /= pivot_value;
            result[pivot_row][j] = A[pivot_row][j];
        }

        // Eliminate non-zero entries below the pivot
        for (int row = 0; row < MATRIX_SIZE; row++) {
            if (row == pivot_row) continue;

            // 2. SIMDを使用して計算
            __m128d scale = _mm_set1_pd(A[row][col]);
            for (int j = 0; j < MATRIX_SIZE; j += 2) {
                __m128d row_pivot = _mm_loadu_pd(result[pivot_row] + j);
                __m128d scaled = _mm_mul_pd(scale, row_pivot);
                __m128d row_target = _mm_loadu_pd(result[row] + j);
                row_target = _mm_sub_pd(row_target, scaled);
                _mm_storeu_pd(result[row] + j, row_target);
            }
        }
    }
}
*/


// 行列の逆行列を計算する関数
void inverseMatrix(short A[MATRIX_SIZE][MATRIX_SIZE], short A_inv[MATRIX_SIZE][MATRIX_SIZE],int start_row,int end_row) {
    int i, j, k;
    short temp;

    // 単位行列を初期化
    for (i = start_row; i < end_row; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            A_inv[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // ガウス・ジョルダン法による逆行列の計算
    for (k = start_row; k < end_row; k++) {
        temp = A[k][k];
        for (j = 0; j < MATRIX_SIZE; j++) {
            A[k][j] *= oinv(temp,N);
            A_inv[k][j] *= oinv(temp,N);
            A[k][j]%=N;
            A_inv[k][j]%=N;
        }
        for (i = 0; i < MATRIX_SIZE; i++) {
            if (i != k) {
                temp = A[i][k];
                for (j = 0; j < MATRIX_SIZE; j++) {
                    A[i][j] -= A[k][j] * temp%N;
                    A_inv[i][j] -= A_inv[k][j] * temp%N;
                    if(A[i][j]<0)
                    A[i][j]=N+A[i][j]%N;
                    if(A_inv[i][j]<0)
                    A_inv[i][j]=N+A_inv[i][j]%N;
                }
            }
        }
    }
}

// 行列の逆行列を計算する関数
void inverseMatrix2(short A[MATRIX_SIZE][MATRIX_SIZE], short A_inv[MATRIX_SIZE][MATRIX_SIZE]) {
    int i, j, k;
    short temp;

    // 単位行列を初期化
    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            A_inv[i][j] = (i == j) ? 1 : 0;
        }
    }

    // ガウス・ジョルダン法による逆行列の計算
    for (k = 0; k < MATRIX_SIZE; k++) {
        temp = A[k][k];
        for (j = 0; j < MATRIX_SIZE; j++) {
            A[k][j] /= temp;
            A_inv[k][j] /= temp;
        }
        for (i = 0; i < MATRIX_SIZE; i++) {
            if (i != k) {
                temp = A[i][k];
                for (j = 0; j < MATRIX_SIZE; j++) {
                    A[i][j] -= A[k][j] * temp;
                    A_inv[i][j] -= A_inv[k][j] * temp;
                    if(A[i][j]<0)
                    A[i][j]=N+A[i][j]%N;
                    if(A_inv[i][j]<0)
                    A_inv[i][j]=N+A_inv[i][j]%N;

                }
            }
        }
    }
}




int main() {
    short A[MATRIX_SIZE][MATRIX_SIZE];
    short A_inv[MATRIX_SIZE][MATRIX_SIZE];
    short C[MATRIX_SIZE][MATRIX_SIZE];
    short AA[MATRIX_SIZE][MATRIX_SIZE];

    srand(clock());
    // 行列 A を初期化
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            A[i][j] = (1 + random()) % N;
            //printf("%d,",A[i][j]);
            // 行列 A のコピーを作成
            AA[i][j]=A[i][j];
        }
        //printf("\n");
    }
    
    
    //
    //matrix_inverse_simd(A,A_inv);
    //inverseMatrix2(A,A_inv);

    // マルチプロセスで行列掛け算を並列化
    int num_processes = 1;
    int rows_per_process = MATRIX_SIZE / num_processes;

    int shmid = shmget(SHM_KEY, sizeof(short) * MATRIX_SIZE * MATRIX_SIZE, IPC_CREAT | 0666);
    if (shmid == -1) {
        perror("shmget");
        exit(1);
    }

    short *shared_C = (short *)shmat(shmid, NULL, 0);
    if (shared_C == (short *)-1) {
        perror("shmat");
        exit(1);
    }

    // 各プロセスで一部の行を計算
    for (int i = 0; i < num_processes; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            int start_row = i * rows_per_process;
            int end_row = (i + 1) * rows_per_process;

            inverseMatrix(A,A_inv,start_row,end_row);
            //print_matrix(A_inv);
            matrix_multiply(AA,A_inv,shared_C,start_row,end_row);
            //matrix_multiply(AA, A_inv, shared_C, start_row, end_row);

            // 結果を表示
            printf("Process %d: Rows %d to %d completed\n", i, start_row, end_row);

            exit(0);
        } else if (pid < 0) {
            perror("fork");
            exit(1);
        }
    }

    // 親プロセスが子プロセスの終了を待つ
    for (int i = 0; i < num_processes; i++) {
        int status;
        wait(&status);
    }

    // 結果を表示
    printf("Result Matrix:\n");
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            printf("%d ", shared_C[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    // 共有メモリを解放
    if (shmdt(shared_C) == -1) {
        perror("shmdt");
        exit(1);
    }
    if (shmctl(shmid, IPC_RMID, NULL) == -1) {
        perror("shmctl");
        exit(1);
    }

    return 0;
}
