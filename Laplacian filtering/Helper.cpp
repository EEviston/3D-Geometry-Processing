#include "Helper.h"


Eigen::MatrixXd Helper::ComputeNormals(Eigen::MatrixXd p_cloud)
{
	Eigen::MatrixXd norm_matrix(p_cloud.rows(), 3);
	Eigen::MatrixXd meancenter(1, 3);
	Eigen::MatrixXd mat = p_cloud;
	meancenter = p_cloud.colwise().sum() / double(p_cloud.rows());

	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, Dynamic, Dynamic> >  my_kd_tree_t;

	my_kd_tree_t   mat_index(mat, 10);
	mat_index.index->buildIndex();

	for (int idx = 0; idx<p_cloud.rows(); idx++) {

		Eigen::RowVector3d query_pt;
		query_pt = p_cloud.row(idx);

		const size_t num_results = 3;
		vector<size_t>   ret_indexes(num_results);
		vector<double> out_dists_sqr(num_results);

		nanoflann::KNNResultSet<double> resultSet(num_results);

		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		Eigen::MatrixXd Selectpoints(num_results, 3);

		for (size_t i = 0; i<num_results; i++) {
			Selectpoints.row(i) = p_cloud.row(ret_indexes[i]);
		}

		Eigen::RowVector3d Nvt, Ct;
		igl::fit_plane(Selectpoints, Nvt, Ct);
		norm_matrix.row(idx) = Nvt;

		float x = (meancenter(0, 0) - p_cloud(idx, 0)) * norm_matrix(idx, 0);
		float y = (meancenter(0, 1) - p_cloud(idx, 1)) * norm_matrix(idx, 1);
		float z = (meancenter(0, 2) - p_cloud(idx, 2)) * norm_matrix(idx, 2);

		if (x + y + z > 0) {
			norm_matrix.row(idx) = -Nvt;
		}
	}

	return norm_matrix;
}

// Heron's formula to find area of triangle
double Helper::FindArea(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, const Eigen::Vector3d& point3) {
	Eigen::Vector3d vec1 = point2 - point1;
	Eigen::Vector3d vec2 = point2 - point3;
	Eigen::Vector3d vec3 = point3 - point1;

	double norm1 = vec1.norm();
	double norm2 = vec2.norm();
	double norm3 = vec3.norm();

	double semiPerimeter = (norm1 + norm2 + norm3) / 2.0;
	double area = sqrt(semiPerimeter*(semiPerimeter - norm1)*(semiPerimeter - norm2)*(semiPerimeter - norm3));
	
	return area;
}

double Helper::FindAngle(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, const Eigen::Vector3d& point3) {
	Eigen::Vector3d vec1 = point2 - point1;
	Eigen::Vector3d vec2 = point3 - point1;
	double norm1 = vec1.norm();
	double norm2 = vec2.norm();

	double angle = acos(vec1.dot(vec2) / (norm1*norm2));
	
	return angle;
}

// inverting the dot product formula: a.b = |a||b|cosTheta
double Helper::FindCotan(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, const Eigen::Vector3d& point3) {
	Eigen::Vector3d vec1 = point2 - point1;
	Eigen::Vector3d vec2 = point3 - point1;

	double norm1 = vec1.norm();
	double norm2 = vec2.norm();

	double angle = acos(vec1.dot(vec2) / (norm1*norm2));
	double cotan = 1.0 / tan(angle);

	return cotan;
}

// This gets the 'C' matrix, containing the weight of each neighbour using cotan
Eigen::SparseMatrix<double> Helper::FindCotanWeights(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces) {
	int numVerts = cloud_verts.rows();
	Eigen::MatrixXd vertices(numVerts, 3);
	Eigen::MatrixXi faces(numVerts, 3);
	vertices = cloud_verts;
	faces = cloud_faces;

	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(10 * numVerts); 
	Eigen::SparseMatrix<double> C(numVerts, numVerts);

	// need to get adjacencies for each triangle 
	Eigen::MatrixXi TT(numVerts, 3);
	Eigen::MatrixXi TTi(numVerts, 3);
	// element i,j in TT is the id of the triangle adjacent to the j edge of the triangle i
	// element i,j in TTi is the id off the edge of the triangle TT(i,j)  (libigl docs)
	igl::triangle_triangle_adjacency(faces, TT, TTi);

	// hold weight and position of neighbour
	std::vector<double> neighbourPosWeight;
	std::vector<std::vector<double>> neighboursList;

	int currentNeighbour, neighbourTri, neighbourEdge, valence;
	double totalWeight, weight, currentWeight;
	Eigen::Vector3d currentPoint, currentTriangle, neighbourPoint, neighbourTriangle;

	for (int i = 0; i < vertices.rows(); i++) 
	{
		totalWeight = 0.0;
		for (int j = 0; j < faces.rows(); j++)
		{
			for (int k = 0; k < 3; k++)
			{
				if (faces(j, k) == i)
				{
					// current triangle
					currentPoint = vertices.row(i);
					neighbourPoint = vertices.row(faces(j, (k + 2) % 3));
					currentTriangle = vertices.row(faces(j, (k + 1) % 3));

					// neighbour triangle
					neighbourTri = TT(j, (k + 2) % 3);
					neighbourEdge = TTi(j, (k + 2) % 3);
					neighbourTriangle = vertices.row(faces(neighbourTri, (neighbourEdge + 2) % 3));

					// sum of cotan 
					weight = FindCotan(currentTriangle, neighbourPoint, currentPoint) + FindCotan(neighbourTriangle, neighbourPoint, currentPoint);

					// store neighbour index/weight
					neighbourPosWeight.push_back(double(faces(j, (k + 2) % 3)));
					neighbourPosWeight.push_back(weight);
					neighboursList.push_back(neighbourPosWeight);

					weight = 0.0;
					neighbourPosWeight.clear();
					break;
				}
			}
		}
		valence = neighboursList.size();

		for (int index = 0; index < valence; index++) 
		{
			currentNeighbour = neighboursList[index][0];
			currentWeight = neighboursList[index][1];
			totalWeight += currentWeight;
			tripletList.push_back(Eigen::Triplet<double>(i, currentNeighbour, currentWeight));
		}
		tripletList.push_back(Eigen::Triplet<double>(i, i, -totalWeight));

		totalWeight = 0.0;
		valence = 0;
		neighboursList.clear();
	}
	C.setFromTriplets(tripletList.begin(), tripletList.end());

	return C;
}

Eigen::SparseMatrix<double> Helper::FindDiagonalArea(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces)
{
	int numVerts = cloud_verts.rows();
	Eigen::MatrixXd vertices(numVerts, 3);
	Eigen::MatrixXi faces(numVerts, 3);
	vertices = cloud_verts;
	faces = cloud_faces;

	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(numVerts);
	double currentArea, totalArea;
	Eigen::Vector3d point1, point2, point3;

	for (int i = 0; i < vertices.rows(); i++) 
	{
		totalArea = 0.0;
		for (int j = 0; j < faces.rows(); j++) 
		{
			for (int k = 0; k < 3; k++) 
			{
				// calculate the area of the triangle if this vert is in the face
				if (faces(j, k) == i) 
				{
					point1 = vertices.row(i);
					point2 = vertices.row(faces(j, (k + 1) % 3));
					point3 = vertices.row(faces(j, (k + 2) % 3));

					currentArea = FindArea(point1, point2, point3);
					totalArea += currentArea;
					break;
				}
			}
			currentArea = 0.0;
		}
		// area for current vertex is 1/3 of total area
		totalArea = totalArea / 3.0;  
		tripletList.push_back(Eigen::Triplet<double>(i, i, (2.0 * totalArea)));
	}
	Eigen::SparseMatrix<double> M(numVerts, numVerts);
	M.setFromTriplets(tripletList.begin(), tripletList.end());
	
	return M;
}

Eigen::SparseMatrix<double> Helper::ComputeInverseSparse(Eigen::SparseMatrix<double> M) {
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> LDLT;
	LDLT.compute(M);
	Eigen::SparseMatrix<double> I(M.rows(), M.cols());
	I.setIdentity();
	Eigen::SparseMatrix<double>  M_inv = LDLT.solve(I);

	return M_inv;
}

Eigen::MatrixXd Helper::AddGaussNoise(Eigen::MatrixXd cloud_verts, float sigma)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, sigma);

	for (int i = 0; i<cloud_verts.rows(); i++) {
		for (int j = 0; j < cloud_verts.cols(); j++)
		{
			double noise = distribution(generator);
			cloud_verts(i, j) += noise;
		}
	}
	return cloud_verts;
}

bool Helper::compare_head(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs)
{
	return lhs(0) < rhs(0);
}

double Helper::findLambda(Eigen::MatrixXd& cloud_verts, double percentage) 
{
	double boundingBoxSize = igl::bounding_box_diagonal(cloud_verts);
	double lambda = (percentage / 100.0) * boundingBoxSize;

	return lambda;
}