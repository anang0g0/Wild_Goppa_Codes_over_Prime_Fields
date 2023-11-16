#include <stdio.h>

#define MAX_SIZE 10

// 関数のプロトタイプ宣言
double determinant(int n, double matrix[MAX_SIZE][MAX_SIZE]);

int main() {
    int n, i, j;

    printf("行列のサイズを入力してください (最大 %d): ", MAX_SIZE);
    scanf("%d", &n);

    if (n <= 0 || n > MAX_SIZE) {
        printf("無効なサイズです。\n");
        return 1;
    }

    double matrix[MAX_SIZE][MAX_SIZE];

    // 行列の要素を入力
    printf("行列の要素を入力してください:\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            printf("matrix[%d][%d]: ", i, j);
            scanf("%lf", &matrix[i][j]);
        }
    }

    // 行列式を計算して表示
    double det = determinant(n, matrix);
    printf("行列式の値: %lf\n", det);

    return 0;
}

// 行列式を計算する再帰的な関数
double determinant(int n, double matrix[MAX_SIZE][MAX_SIZE]) {
    if (n == 1) {
        return matrix[0][0];
    } else if (n == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    } else {
        double det = 0.0;
        int i, j, k;
        double submatrix[MAX_SIZE][MAX_SIZE];

        for (i = 0; i < n; i++) {
            // サブマトリックスを作成
            for (j = 1; j < n; j++) {
                for (k = 0; k < n; k++) {
                    if (k < i) {
                        submatrix[j - 1][k] = matrix[j][k];
                    } else if (k > i) {
                        submatrix[j - 1][k - 1] = matrix[j][k];
                    }
                }
            }

            // 再帰的に行列式を計算
            det += matrix[0][i] * determinant(n - 1, submatrix) * ((i % 2 == 0) ? 1 : -1);
        }

        return det;
    }
}
