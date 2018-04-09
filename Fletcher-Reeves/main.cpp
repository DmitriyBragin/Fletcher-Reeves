#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <conio.h>
#include "MatrixWork.h"
#include <iostream>

//////////////////////////////////////////////////////////////
// LOCAL DEFINES
//////////////////////////////////////////////////////////////

#define EPSILON 1.e-6 

double Function(double x1, double x2, double x3)
{
	return 2 * x1 * x1 + 5 * x2 * x2 + 8 * x3 * x3 + 4 * x1 * x2 + 2 * x1 * x3 + 4 * x2 * x3;
}

void Gradient(double x[], double y[])
{
	y[0] = 4 * x[0] + 4 * x[1] + 2 * x[2];
	y[1] = 4 * x[0] + 10 * x[1] + 4 * x[2];
	y[2] = 2 * x[0] + 4 * x[1] + 16 * x[2];
}

void Hessian(double x[], double H[3][3])
{
	H[0][0] = 4;
	H[0][1] = H[1][0] = 4;
	H[0][2] = H[2][0] = 2;
	H[1][1] = 10;
	H[1][2] = H[2][1] = 4;
	H[2][2] = 16;
}

void Gradient2(double x[], double y[], int recalculate)
{
	double **H, **invH, grad[3], det, tmp[3][3];
	H = new double*[3];
	invH = new double*[3];

	for (int i = 0; i < 3; i++)
	{
		H[i] = new double[3];
		invH[i] = new double[3];
	}

	if (recalculate)
	{
		Hessian(x, tmp);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				H[i][j] = tmp[i][j];
			}
		}
		inverse(H, invH, 3);
	}

	Gradient(x, grad);
	y[0] = (invH[0][0] * grad[0] + invH[0][1] * grad[1] + invH[0][2] * grad[2]);
	y[1] = (invH[1][0] * grad[0] + invH[1][1] * grad[1] + invH[1][2] * grad[2]);
	y[2] = (invH[2][0] * grad[0] + invH[2][1] * grad[1] + invH[2][2] * grad[2]);
}

//////////////////////////////////////////////////////////////
// STEP NEWTON
//////////////////////////////////////////////////////////////

double Step_Newton(double x[], double y[], double *f, double norm)
{
	/* Pshenichniy power */
	double newtonStep = 1;
	double x_k_1[3];
	x_k_1[0] = x[0] - newtonStep * y[0];
	x_k_1[1] = x[1] - newtonStep * y[1];
	x_k_1[2] = x[2] - newtonStep * y[2];
	double fk1 = Function(x_k_1[0], x_k_1[1], x_k_1[2]);
	double fk2 = Function(x[0], x[1], x[2]);
	double mult;
	double grad[3];
	Gradient(x, grad);
	mult = -0.5 * (grad[0] * y[0] + grad[1] * y[1] + grad[2] * y[2]);
	while ((fk1 - fk2) > (mult * newtonStep))
	{
		newtonStep /= 2;
		x_k_1[0] = x[0] - newtonStep * y[0];
		x_k_1[1] = x[1] - newtonStep * y[1];
		x_k_1[2] = x[2] - newtonStep * y[2];
		fk1 = Function(x_k_1[0], x_k_1[1], x_k_1[2]);
		fk2 = Function(x[0], x[1], x[2]);
		Gradient(x, grad);
		mult = -0.5 * (grad[0] * y[0] + grad[1] * y[1] + grad[2] * y[2]);
	}

	x[0] -= newtonStep * y[0];
	x[1] -= newtonStep * y[1];
	x[2] -= newtonStep * y[2];
	
	*f = Function(x[0], x[1], x[2]);
	return newtonStep;
}

double* vector_dot_matrix(double* vec, double Matrix[3][3])
{
	double result[3];
	for (int i = 0; i < 3; i++)
	{
		result[i] = 0;
		for (int j = 0; j < 3; j++)
			result[i] += vec[j] * Matrix[j][i];
	}
	return result;
}

double* matrix_dot_vector(double Matrix[3][3], double* vec)
{
	double result[3];
	for (int i = 0; i < 3; i++)
	{
		result[i] = 0;
		for (int j = 0; j < 3; j++)
			result[i] += vec[j] * Matrix[i][j];
	}
	return result;
}



//////////////////////////////////////////////////////////////
// STEP FLETCHER-REEVES
//////////////////////////////////////////////////////////////
int    fp_loc_iter = 0;
double S_prev[3];
double stepFR = 0;
double Step_FletcherReeves(double x[], double y[], double *f, double norm)
{
	double S_cur[3];
	double hessian[3][3];
	Hessian(x, hessian);
	if (fp_loc_iter == 0)
	{
		S_cur[0] = S_prev[0] = -y[0];
		S_cur[1] = S_prev[1] = -y[1];
		S_cur[2] = S_prev[2] = -y[2];		
		double nominator = y[0] * S_cur[0] + y[1] * S_cur[1] + y[2] * S_cur[2];
		double *temp;
		temp = vector_dot_matrix(S_cur, hessian);
		double denominator = temp[0] * S_cur[0] + temp[1] * S_cur[1] + temp[2] * S_cur[2];
		stepFR = -nominator / denominator;
		fp_loc_iter++;
	}

	double nominator = 0;
	double denominator = 0;
	double *temp1;

	x[0] += stepFR * S_prev[0];
	x[1] += stepFR * S_prev[1];
	x[2] += stepFR * S_prev[2];

	/* Finding beta */
	double grad[3];
	Gradient(x, grad);
	temp1 = vector_dot_matrix(grad, hessian);
	nominator = temp1[0] * S_prev[0] + temp1[1] * S_prev[1] + temp1[2] * S_prev[2];
	temp1 = vector_dot_matrix(S_prev, hessian);
	denominator = temp1[0] * S_prev[0] + temp1[1] * S_prev[1] + temp1[2] * S_prev[2];
	double beta = nominator / denominator;
	std::cout << std::endl;
	std::cout << "S_cur:" << S_prev[0] << " " << S_prev[1] << " " << S_prev[2] << std::endl;
	/* Finding S_cur*/
	S_cur[0] = -grad[0] + beta * S_prev[0];
	S_cur[1] = -grad[1] + beta * S_prev[1];
	S_cur[2] = -grad[2] + beta * S_prev[2];

	/* Finding step for next step*/
	double nominator1 = grad[0] * S_cur[0] + grad[1] * S_cur[1] + grad[2] * S_cur[2];
	double *temp;
	temp = vector_dot_matrix(S_cur, hessian);
	double denominator1 = temp[0] * S_cur[0] + temp[1] * S_cur[1] + temp[2] * S_cur[2];
	stepFR = -nominator1 / denominator1;

	/* Setting current as prev*/
	S_prev[0] = S_cur[0];
	S_prev[1] = S_cur[1];
	S_prev[2] = S_cur[2];

	*f = Function(x[0], x[1], x[2]);
	return stepFR;
}


//////////////////////////////////////////////////////////////
// STEP FLETCHER-REEVES MODIFICATIONS
//////////////////////////////////////////////////////////////

double g(double x1, double x2, double x3)
{
	return 2 * x1 * x1 + 5 * x2 * x2 + 8 * x3 * x3 + 4 * x1 * x2 + 2 * x1 * x3 + 4 * x2 * x3 + sin(x1 * x1) * sin(x1 * x1);
}

void gradG(double x[], double y[])
{
	y[0] = 2 * (x[0] * sin(2 * x[0] * x[0]) + 2 * x[0] + 2 * x[1] + x[2]);
	y[1] = 4 * x[0] + 10 * x[1] + 4 * x[2];
	y[2] = 2 * x[0] + 4 * x[1] + 16 * x[2];
}

void hessianG(double x[], double H[3][3])
{
	H[0][0] = -8 * x[0] * x[0] * sin(x[0] * x[0]) * sin(x[0] * x[0]) 
		+ 8 * x[0] * x[0] * cos(x[0] * x[0]) * cos(x[0] * x[0])
		+ 4 * cos(x[0] * x[0]) * cos(x[0] * x[0]) * sin(x[0] * x[0]) * sin(x[0] * x[0]) + 4;
	H[0][1] = H[1][0] = 4;
	H[0][2] = H[2][0] = 2;
	H[1][1] = 10;
	H[1][2] = H[2][1] = 4;
	H[2][2] = 16;
}


double findAlpha(double x[], double y[], double *f, double S[])
{
	double alpha = (sqrt(5) - 1) / 2;
	double a = 0, b = 1;
	double l, m, fl, fm;

	l = a + (1 - alpha) * (b - a);
	m = a + b - l;
	fl = g(x[0] + S[0] * l, x[1] + S[1] * l, x[2] + S[2] * l);
	fm = g(x[0] + S[0] * m, x[1] + S[1] * m, x[2] + S[2] * m);

	while (fabs(m - l) > .00000001)
	{
		if (fl < fm)
		{
			b = m;
			m = l;
			fm = fl;
			l = a + (1 - alpha) * (b - a);
			fl = g(x[0] + S[0] * l, x[1] + S[1] * l, x[2] + S[2] * l);
		}
		else
		{
			a = l;
			l = m;
			fl = fm;
			m = a + alpha * (b - a);
			fm = g(x[0] + S[0] * m, x[1] + S[1] * m, x[2] + S[2] * m);
		}
	}
	return l;

}
int    mfp_loc_iter = 0;
double S_Mprev[3];
double stepMFR = 0;
double Step_ModificationFletcherReeves(double x[], double y[], double *f, double norm)
{
	double S_cur[3];
	double hessian[3][3];
	hessianG(x, hessian);
	if (mfp_loc_iter == 0)
	{
		S_cur[0] = S_Mprev[0] = -y[0];
		S_cur[1] = S_Mprev[1] = -y[1];
		S_cur[2] = S_Mprev[2] = -y[2];
		stepMFR = findAlpha(x, y, f, S_cur);
		mfp_loc_iter++;
	}

	double nominator = 0;
	double denominator = 0;
	double *temp1;


	x[0] += stepMFR * S_Mprev[0];
	x[1] += stepMFR * S_Mprev[1];
	x[2] += stepMFR * S_Mprev[2];

	/* Finding beta */
	double grad[3];
	gradG(x, grad);
	double delta[3];
	delta[0] = grad[0] - y[0];
	delta[1] = grad[1] - y[1];
	delta[2] = grad[2] - y[2];

	nominator = grad[0] * delta[0] + grad[1] * delta[1] + grad[2] * delta[2];
	denominator = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
	double beta = 0;
	if (mfp_loc_iter % 4 != 0)
	{
		beta = nominator / denominator;
		mfp_loc_iter++;
	}
	else
	{
		mfp_loc_iter = 0;
	}
	
	/* Finding S_cur*/
	S_cur[0] = -grad[0] + beta * S_Mprev[0];
	S_cur[1] = -grad[1] + beta * S_Mprev[1];
	S_cur[2] = -grad[2] + beta * S_Mprev[2];

	/* Finding step for next step*/
	stepMFR = findAlpha(x, y, f, S_cur);

	/* Setting current as prev*/
	S_Mprev[0] = S_cur[0];
	S_Mprev[1] = S_cur[1];
	S_Mprev[2] = S_cur[2];
	*f = g(x[0], x[1], x[2]);
	return stepMFR;
}
//////////////////////////////////////////////////////////////
// TEST METHODS ROUTINE
//////////////////////////////////////////////////////////////
int period;
int Test(double(*Step)(double[], double[], double *, double), int i)
{
	int    step = 0;
	double x[3] = { -1, -1, -1 }, x_prev[3], delta[3], old_delta[3], y[3],
		f = Function(x[0], x[1], x[2]), norm, mod, prev_mod;
	int iters = 0;
	static double solution[3];

	fp_loc_iter = 0;
	do
	{
		if(i != 6)
			Gradient(x, y);
		else
			gradG(x, y);

		norm = sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
		printf("x=(%06.4lf,%06.4lf,%06.4lf) y=(%06.4lf, %06.4lf,%06.4lf) norm=%08.6lf,"
			" f=%07.5lf", x[0], x[1], x[2], y[0], y[1], y[2], norm, f);
		if (norm <= EPSILON)
			break;
		if (i == 3)
			Gradient2(x, y, 1);

		memcpy(x_prev, x, 2 * sizeof(double));
		printf(" step=%6.5lf\n", Step(x, y, &f, norm));

		step++;
	} while (norm > EPSILON);

	if (i == 0)
		memcpy(solution, x, 2 * sizeof(double));
	printf("\nDone in %d steps\n", step);
//	getch();
	return step;
}


//////////////////////////////////////////////////////////////
// MAIN FUNCTION
//////////////////////////////////////////////////////////////
int main(void)
{
	/*
	* NOTE: IF YOU WANT TO USE THIS CODE, CHANGE THOSE THINGS:
	* FOR SQARE FORM MINIMIZATION AND NEWTON 
	* 1) Function() <- this is for standart square form (x^2 + y^2 + z^2 + xy + xz + yz), change your coefficents manually
	* 2) Gradient() <- change this for your gradient of square form
	* 3) Hessian() <- change for your Hessian matrix
	* FOR NON-SQARE MINIMIZATION
	* 1) g() <- new function
	* 2) gradG, HessianG - same, manually recalculate coefficients and put them in
	ENJOY, KEK
	*/
	int i;
	printf("\n2-dimensional minimization example\n");
	printf("\nFletcher-Reeves step:\n");
	Test(Step_FletcherReeves, 5);
	printf("\nModifications Fletcher-Reeves step:\n");
	Test(Step_ModificationFletcherReeves, 6);
	printf("\nSecond order gradient:\n");
	Test(Step_Newton, 3);
	return 0;
}