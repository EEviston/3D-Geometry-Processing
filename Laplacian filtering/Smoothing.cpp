#include "Smoothing.h"
#include "Curvature.h"

Eigen::MatrixXd Smoothing::ExplicitSmoothing(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces, int discretizationType, double lambda)
{
		int numVerts = cloud_verts.rows();

		// Construct Laplace
		Eigen::SparseMatrix<double> laplaceOp(numVerts, numVerts);
		if (discretizationType == 0)
			laplaceOp = Curvature::ConstructUniformLaplace(cloud_verts, cloud_faces);
		else if (discretizationType == 1)
			laplaceOp = Curvature::ComputeCotanDiscretization(cloud_verts, cloud_faces);

		// Explicit integration
		Eigen::MatrixXd x(numVerts, 3);
		Eigen::SparseMatrix<double> identity(numVerts, numVerts);
		identity.setIdentity();
		x = (identity + laplaceOp * lambda) * cloud_verts;

		return x;
}

Eigen::MatrixXd Smoothing::ImplicitSmoothing(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces, double lambda) 
{
	int numVerts = cloud_verts.rows();

	// Construct Lw
	Eigen::SparseMatrix<double> Lw(numVerts, numVerts);
	Lw = Helper::FindCotanWeights(cloud_verts, cloud_faces);

	// Construct M
	Eigen::SparseMatrix<double> M(numVerts, numVerts);
	M = Helper::FindDiagonalArea(cloud_verts, cloud_faces);

	// Ax = b
	Eigen::SparseMatrix<double> A(numVerts, numVerts);
	A = M - lambda*Lw;

	Eigen::MatrixXd b(numVerts, 3);
	b = M * cloud_verts;

	Eigen::MatrixXd x(numVerts, 3);
	// Solver
	ConjugateGradient<SparseMatrix<double>, Lower | Upper> cg;
	cg.compute(A);
	x = cg.solve(b).eval();
	std::cout << "isCompressed: " << A.isCompressed() << std::endl;
	std::cout << "#iterations:     " << cg.iterations() << std::endl;
	std::cout << "estimated error: " << cg.error() << std::endl;
	// update b, and solve again
	//new_verts = cg.solve(b).eval();

	return x;
}

// this is just to see which solver is quickest
Eigen::MatrixXd Smoothing::ImplicitSmoothingTest(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces, double lambda) 
{
	int numVerts = cloud_verts.rows();

	Eigen::SparseMatrix<double> M(numVerts, numVerts);
	Eigen::SparseMatrix<double> Lw(numVerts, numVerts);
	Eigen::SparseMatrix<double> A(numVerts, numVerts);
	Eigen::MatrixXd b(numVerts, 3);
	Eigen::MatrixXd x(numVerts, 3);

	M = Helper::FindDiagonalArea(cloud_verts, cloud_faces);
	Lw = Helper::FindCotanWeights(cloud_verts, cloud_faces);

	// Solving linear system and symmetrizing by multiplying M
	A = M - lambda * Lw;
	b = M * cloud_verts;

	// Direct sparse Cholesky factorizations (ConjugateGradient solver actually slower)
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> LDLT;
	LDLT.compute(A);
	x = LDLT.solve(b).eval();

	return x;
}