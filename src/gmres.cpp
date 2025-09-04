#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <cassert>
#include <chrono>
#include "sparseMatrix.hpp"
#include "gmres.hpp"

// 引入OpenMP头文件
#include <omp.h>

using namespace std;

const int RESTART_TIMES = 20;  // 禁止修改
const double REL_RESID_LIMIT = 1e-6; // 禁止修改
const int ITERATION_LIMIT = 10000; // 禁止修改

void applyRotation(double &dx, double &dy, double &cs, double &sn)
{
    double temp = cs * dx + sn * dy;
    dy = (-sn) * dx + cs * dy;
    dx = temp;
}

void generateRotation(double &dx, double &dy, double &cs, double &sn)
{
    if (dx == double(0))
    {
        cs = double(0);
        sn = double(1);
    }
    else
    {
        double scale = fabs(dx) + fabs(dy);
        double norm = scale * std::sqrt(fabs(dx / scale) * fabs(dx / scale) +
                                        fabs(dy / scale) * fabs(dy / scale));
        double alpha = dx / fabs(dx);
        cs = fabs(dx) / norm;
        sn = alpha * dy / norm;
    }
}


void rotation2(uint Am, double *H, double *cs, double *sn, double *s, uint i)
{
    for (uint k = 0; k < i; k++)
    {
        applyRotation(H[k * Am + i], H[(k + 1) * Am + i], cs[k], sn[k]);
    }
    generateRotation(H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
    applyRotation(H[i * Am + i], H[(i + 1) * Am + i], cs[i], sn[i]);
    applyRotation(s[i], s[i + 1], cs[i], sn[i]);
}

double calculateNorm(const double* vec, uint N) {
    double sum = 0.0;
    // 使用OpenMP并行化，并用reduction安全地进行求和
    #pragma omp parallel for reduction(+:sum)
    for (uint i = 0; i < N; ++i) {
        sum += vec[i] * vec[i];
    }
    return std::sqrt(sum);
}

void spmv(const uint* rowPtr, const uint* colInd, const double* values,
          const double* x, double* y, uint numRows) {
    // 使用OpenMP并行化，每一行的计算是独立的
    #pragma omp parallel for
    for (uint i = 0; i < numRows; ++i) {
        double sum = 0.0;
        for (uint j = rowPtr[i]; j < rowPtr[i+1]; ++j) {
            sum += values[j] * x[colInd[j]];
        }
        y[i] = sum;
    }
}

double dotProduct(const double* x, const double* y, uint N) {
    double sum = 0.0;
    // 使用OpenMP并行化，并用reduction安全地进行求和
    #pragma omp parallel for reduction(+:sum)
    for (uint i = 0; i < N; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

void daxpy(double alpha, const double* x, double* y, uint N) {
    // 使用OpenMP并行化，y[i]的更新是独立的
    #pragma omp parallel for
    for (uint i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }
}

void dscal(double alpha, double* x, uint N) {
    // 使用OpenMP并行化，x[i]的更新是独立的
    #pragma omp parallel for
    for (uint i = 0; i < N; ++i) {
        x[i] *= alpha;
    }
}

void dcopy(const double* src, double* dst, uint N) {
    // 使用OpenMP并行化，dst[i]的更新是独立的
    #pragma omp parallel for
    for (uint i = 0; i < N; ++i) {
        dst[i] = src[i];
    }
}

void sovlerTri(int Am, int i, double *H, double *s)
{
    for (int j = i; j >= 0; j--)
    {
        s[j] /= H[Am * j + j];
        for (int k = j - 1; k >= 0; k--)
        {
            s[k] -= H[k * Am + j] * s[j];
        }
    }
}

RESULT gmres(SpM<double> *A_d, double *x_d, double *_b)
{
    const uint N = A_d->nrows;

    std::vector<double> r0(N);
    std::vector<double> V((RESTART_TIMES + 1) * N);
    std::vector<double> s(RESTART_TIMES + 1, 0.0);
    // H, cs, sn 保持在CPU端，因为它们很小，并行开销大
    std::vector<double> H((RESTART_TIMES + 1) * RESTART_TIMES);
    std::vector<double> cs(RESTART_TIMES);
    std::vector<double> sn(RESTART_TIMES);
    
    double beta;
    beta = calculateNorm(_b, N);            // 禁止修改
    double RESID_LIMIT = REL_RESID_LIMIT * beta;// 禁止修改
    double init_res = beta;                     // 禁止修改

    int i, j, k;
    double resid;
    int iteration = 0;

    auto start = std::chrono::high_resolution_clock::now(); //禁止修改

    do
    {
        spmv(A_d->rows, A_d->cols, A_d->vals, x_d, r0.data(), N);
        daxpy(-1.0, _b, r0.data(), N);
        beta = calculateNorm(r0.data(), N);
        dscal(-1.0 / beta, r0.data(), N);
        dcopy(r0.data(), V.data(), N);

        fill(s.begin(), s.end(), 0.0);
        s[0] = beta;
        resid = std::abs(beta);
        i = -1;

        if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) break;

        do
        {
            i++;
            iteration++;

            double* V_i = V.data() + i * N;
            double* V_i_plus_1 = V.data() + (i + 1) * N;

            spmv(A_d->rows, A_d->cols, A_d->vals, V_i, r0.data(), N);

            for (k = 0; k <= i; k++)
            {
                H[k * RESTART_TIMES + i] = dotProduct(r0.data(), V.data() + k * N, N);
                daxpy(-H[k * RESTART_TIMES + i], V.data() + N * k, r0.data(), N);
            }
            H[(i + 1) * RESTART_TIMES + i] = calculateNorm(r0.data(), N);

            dscal(1.0 / H[(i + 1) * RESTART_TIMES + i], r0.data(), N);
            dcopy(r0.data(), V_i_plus_1, N);

            rotation2(RESTART_TIMES, H.data(), cs.data(), sn.data(), s.data(), i);

            resid = std::abs(s[i + 1]);

            if (resid <= RESID_LIMIT || iteration >= ITERATION_LIMIT) break;

        } while (i + 1 < RESTART_TIMES && iteration <= ITERATION_LIMIT);

        sovlerTri(RESTART_TIMES, i, H.data(), s.data());

        for (j = 0; j <= i; j++)
        {
            daxpy(s[j], V.data() + j * N, x_d, N);
        }

    } while (resid > RESID_LIMIT && iteration <= ITERATION_LIMIT);

    auto stop = std::chrono::high_resolution_clock::now();                  //禁止修改
    std::chrono::duration<float, std::milli> duration = stop - start;       //禁止修改
    float test_time = duration.count();                                     //禁止修改

    return make_tuple(iteration, test_time, resid / init_res);              //禁止修改
}

// initialize函数中的计算也可以并行化以加速初始化
void initialize(SpM<double> *A, double *x, double *b)
{
    int N = A->nrows;

    for (int i = 0; i < N; i++)
    {
        x[i] = sin(i);
    }

    double beta = calculateNorm(x, N); // 已被并行化
    
    #pragma omp parallel for
    for (uint i = 0; i < N; i++)
    {
        x[i] /= beta;
    }

    spmv(A->rows, A->cols, A->vals, x, b, N); // 已被并行化

    #pragma omp parallel for
    for (uint i = 0; i < N; i++)
        x[i] = 0.0;
}