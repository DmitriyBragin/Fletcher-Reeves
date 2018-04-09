#pragma once
void getCofactor(double **A, double** temp, int p, int q, int n);
double determinant(double** A, int N, int n);
void adjoint(double** A, double** adj, int N);
bool inverse(double **A, double** inverse, int N);
