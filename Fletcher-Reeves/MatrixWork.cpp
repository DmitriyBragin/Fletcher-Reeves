// C++ program to find adjoint and inverse of a matrix
using namespace std;

// Function to get cofactor of A[p][q] in temp[][]. n is current
// dimension of A[][]
void getCofactor(double **A, double** temp, int p, int q, int n)
{
	int i = 0, j = 0;

	// Looping for each element of the matrix
	for (int row = 0; row < n; row++)
	{
		for (int col = 0; col < n; col++)
		{
			//  Copying into temporary matrix only those element
			//  which are not in given row and column
			if (row != p && col != q)
			{
				temp[i][j++] = A[row][col];

				// Row is filled, so increase row index and
				// reset col index
				if (j == n - 1)
				{
					j = 0;
					i++;
				}
			}
		}
	}
}

/* Recursive function for finding determinant of matrix.
n is current dimension of A[][]. */
double determinant(double** A, int N, int n)
{
	double D = 0; // Initialize result

			   //  Base case : if matrix contains single element
	if (n == 1)
		return A[0][0];

	double** temp = new double*[N]; // To store cofactors
	for (int i = 0; i < N; i++)
	{
		temp[i] = new double[N];
	}
	int sign = 1;  // To store sign multiplier

				   // Iterate for each element of first row
	for (int f = 0; f < n; f++)
	{
		// Getting Cofactor of A[0][f]
		getCofactor(A, temp, 0, f, n);
		D += sign * A[0][f] * determinant(temp, N, n - 1);

		// terms are to be added with alternate sign
		sign = -sign;
	}
	for (int i = 0; i < N; i++)
	{
		delete(temp[i]);
	}
	delete(temp);
	return D;
}

// Function to get adjoint of A[N][N] in adj[N][N].
void adjoint(double** A, double** adj, int N)
{
	if (N == 1)
	{
		adj[0][0] = 1;
		return;
	}

	// temp is used to store cofactors of A[][]
	int sign = 1;
	double **temp = new double*[N];
	for (int i = 0; i < N; i++)
	{
		temp[i] = new double[N];
	}

	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<N; j++)
		{
			// Get cofactor of A[i][j]
			getCofactor(A, temp, i, j, N);

			// sign of adj[j][i] positive if sum of row
			// and column indexes is even.
			sign = ((i + j) % 2 == 0) ? 1 : -1;

			// Interchanging rows and columns to get the
			// transpose of the cofactor matrix
			adj[j][i] = (sign)*(determinant(temp, N, N - 1));
		}
	}
	for (int i = 0; i < N; i++)
	{
		delete(temp[i]);
	}
	delete(temp);
}

// Function to calculate and store inverse, returns false if
// matrix is singular
bool inverse(double **A, double** inverse, int N)
{
	// Find determinant of A[][]
	double det = determinant(A, N, N);
	if (det == 0)
	{
		return false;
	}

	// Find adjoint
	double** adj = new double*[N];
	for (int i = 0; i < N; i++)
	{
		adj[i] = new double[N];
	}
	adjoint(A, adj, N);

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)"
	for (int i = 0; i<N; i++)
		for (int j = 0; j<N; j++)
			inverse[i][j] = adj[i][j] / (det);
	for (int i = 0; i < N; i++)
	{
		delete(adj[i]);
	}
	delete(adj);
	return true;
}