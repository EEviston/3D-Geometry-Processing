#include "Curvature.h"

Eigen::SparseMatrix<double> Curvature::ConstructUniformLaplace(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces)
{
	int numVerts = cloud_verts.rows();
	Eigen::MatrixXd vertices(numVerts, 3);
	Eigen::MatrixXi faces(numVerts, 3);
	Eigen::MatrixXd normals(numVerts, 3);
	vertices = cloud_verts;
	faces = cloud_faces;

	std::vector<Eigen::Triplet<double>> tripletList;
	// 6 is the average number of neighbours on a typical trianguated mesh
	tripletList.reserve(6 * numVerts);
	Eigen::SparseMatrix<double> laplaceMat(numVerts, numVerts);

	std::vector<int> neighbourIndices;
	int dimension = 3;
	int valence;

	// find neighbours
	for (int i = 0; i < vertices.rows(); i++) 
	{
		for (int j = 0; j < faces.rows(); j++) 
		{
			for (int k = 0; k < dimension; k++) 
			{
				if (faces(j, k) == i) {
					// if we find the vertex, add its neighbour
					// (order of vertices is always same in faces)
					neighbourIndices.push_back(faces(j, (k + 1) % dimension));
					break;
				}
			}
		}
		valence = neighbourIndices.size();

		for (int n = 0; n < valence; n++)
		{
			tripletList.push_back(Eigen::Triplet<double>(i, neighbourIndices[n], 1.0 / double(valence)));
		}
		// for itself push a weight of -1
		tripletList.push_back(Eigen::Triplet<double>(i, i, -1));
		neighbourIndices.clear();
	}
	laplaceMat.setFromTriplets(tripletList.begin(), tripletList.end());
	return laplaceMat;
}

Eigen::MatrixXd Curvature::ComputeMeanCurvature(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces, int discretizationType) 
{
	int numVerts = cloud_verts.rows();
	Eigen::MatrixXd normals(numVerts, 3);
	Eigen::MatrixXd vertices(numVerts, 3);
	vertices = cloud_verts;
	normals = Helper::ComputeNormals(cloud_verts);
	Eigen::SparseMatrix<double> laplaceOp(numVerts, numVerts);
	if (discretizationType == 0)
		laplaceOp = ConstructUniformLaplace(cloud_verts, cloud_faces);
	else if (discretizationType == 1)
		laplaceOp = ComputeCotanDiscretization(cloud_verts, cloud_faces);

	// need to multiply each vertex by the laplace operator
	Eigen::MatrixXd laplaceVerts(numVerts, 3);
	laplaceVerts = laplaceOp * vertices;

	// Compute the magnitude of the mean curvature using formula
	Eigen::MatrixXd H(numVerts, 1);
	H = laplaceVerts.rowwise().norm() / 2.0;
 
	// find the sign of the curvature
	Eigen::MatrixXd signCurv(numVerts, 1);
	signCurv = (laplaceVerts.cwiseProduct(normals)).rowwise().sum();
	signCurv = signCurv.array() / signCurv.array().abs().array();
	
	// flip the curvature 
	//H = H.cwiseProduct(-1 * signCurv);
	//std::cout << "signCurv: " << signCurv << std::endl;
	
	// use jet to transform scalar (H) to colour
	Eigen::MatrixXd meanShading(numVerts, 3);
	igl::jet(H, true, meanShading);

	return meanShading;
}

Eigen::MatrixXd Curvature::ComputeGaussianCurvature(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces)
{
	int numVerts = cloud_verts.rows();
	Eigen::MatrixXd vertices(numVerts, 3);
	Eigen::MatrixXi faces(numVerts, 3);
	Eigen::MatrixXd gaussCurvMatrix(numVerts, 1);
	vertices = cloud_verts;
	faces = cloud_faces;
	double currentArea, currentAngle, totalArea, totalAngle, currentGaussCurv;
	Eigen::Vector3d point1, point2, point3;

	for (int i = 0; i < vertices.rows(); i++) 
	{
		totalArea = 0.0;
		totalAngle = 0.0;
		currentGaussCurv = 0.0;
		for (int j = 0; j < faces.rows(); j++) 
		{
			for (int k = 0; k < 3; k++) 
			{
				if (faces(j, k) == i) 
				{
					// for this vertex in the face, find angles and area of triangle
					point1 = vertices.row(i);
					point2 = vertices.row(faces(j, (k + 1) % 3));
					point3 = vertices.row(faces(j, (k + 2) % 3));

					currentArea = Helper::FindArea(point1, point2, point3);
					currentAngle = Helper::FindAngle(point1, point2, point3);

					totalArea += currentArea;
					totalAngle += currentAngle;
					break;
				}
			}
		}
		// use angle deficit to get the gauss curvature 
		currentGaussCurv = (2.0*PI - totalAngle) / (totalArea/3.0);
		gaussCurvMatrix(i, 0) = currentGaussCurv;
	}

	Eigen::MatrixXd gaussShading(numVerts, 3);
	igl::jet(gaussCurvMatrix, false, gaussShading);

	return gaussShading;
}

Eigen::SparseMatrix<double> Curvature::ComputeCotanDiscretization(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces) {
	int numVerts = cloud_verts.rows();
	Eigen::SparseMatrix<double> C(numVerts, numVerts);
	Eigen::SparseMatrix<double> M(numVerts, numVerts);
	Eigen::SparseMatrix<double> M_inverse(numVerts, numVerts);
	
	C = Helper::FindCotanWeights(cloud_verts, cloud_faces);
	M = Helper::FindDiagonalArea(cloud_verts, cloud_faces);
	M_inverse = Helper::ComputeInverseSparse(M);

	Eigen::SparseMatrix<double> L(numVerts, numVerts);
	L = C * M_inverse;

	return L;
}

Eigen::MatrixXd Curvature::ComputeMeshReconstruction(Eigen::MatrixXd cloud_verts, Eigen::MatrixXi cloud_faces, int k)
{
	int numVerts = cloud_verts.rows();
	Eigen::SparseMatrix<double> laplaceOp(numVerts, numVerts);
	laplaceOp = ComputeCotanDiscretization(cloud_verts, cloud_faces);

	// Construct matrix operation object using the wrapper class SparseGenMatProd
	SparseGenMatProd<double> op(laplaceOp);

	// Construct eigen solver object, requesting the largest three eigenvalues
	GenEigsSolver<double, SMALLEST_MAGN, SparseGenMatProd<double> > eigs(&op, k, (2 * k + 1));

	// Initialize and compute
	eigs.init();
	int nconv = eigs.compute();

	// Retrieve results
	Eigen::VectorXd evalues;
	Eigen::MatrixXd evectors;
	//Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> evectors;
	if (eigs.info() == SUCCESSFUL)
	{
		evalues = eigs.eigenvalues().real();
		evectors = eigs.eigenvectors().real();
	}

	std::cout << "Eigenvectors found:\n " << evectors << std::endl;
	std::cout << "Eigenvalues found:\n " << evalues << std::endl;
	std::cout << "Eigenvectors size:\n " << evectors.size() << std::endl;

	Eigen::MatrixXd result(numVerts, 3);

	for (int i = 0; i < cloud_verts.cols(); i++)
	{
		Eigen::MatrixXd temp(numVerts, 1);
		temp.setZero();
		Eigen::MatrixXd cloudCol(numVerts, 1);
		cloudCol = cloud_verts.col(i);

		for (int j = 0; j < evectors.cols(); j++)
		{
			Eigen::MatrixXd vecCol(numVerts, 1);
			vecCol = evectors.col(j);
			Eigen::MatrixXd product = cloudCol.transpose() * vecCol;
			double doubleProd = product(0, 0);

			temp += doubleProd * vecCol;
		}

		for (int k = 0; k < numVerts; k++)
		{
			result(k, i) = temp(k, 0);
		}
	}

	return result;
}


