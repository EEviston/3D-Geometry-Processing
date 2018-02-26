#include "Icp.h"

MatrixXd meancenter(1, 3);
Eigen::MatrixXd ps;
Eigen::MatrixXd qs;


Icp::Icp()
{
}

Icp::~Icp()
{
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d> Icp::point_to_point(Eigen::MatrixXd V0, Eigen::MatrixXd V1)
{
	Eigen::Matrix<double, Dynamic, Dynamic>  mat(V0.rows(), 3);
	for (size_t i = 0; i < V0.rows(); i++)
		for (size_t d = 0; d < 3; d++)
			mat(i, d) = V0(i, d);

	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, Dynamic, Dynamic> >  my_kd_tree_t;
	my_kd_tree_t   mat_index(mat, 10);
	mat_index.index->buildIndex();

	Vector3d p_bar, q_bar;
	p_bar = (1.0 / V1.rows()) * V0.colwise().sum();
	q_bar = (1.0 / V1.rows()) * V1.colwise().sum();
	Eigen::MatrixXd A = MatrixXd::Zero(3, 3);
	Nv = Eigen::MatrixXd::Zero(V0.rows(), V0.cols());
	ps = Eigen::MatrixXd::Zero(Nv.rows(), 3);
	qs = Eigen::MatrixXd::Zero(Nv.rows(), 3);

	for (int idx = 0; idx < V1.rows(); idx++) {
		std::vector<double> query_pt(3);
		for (size_t d = 0; d < 3; d++)
			query_pt[d] = V1(idx, d);

		const size_t num_results = 3;
		vector<size_t>   ret_indexes(num_results);
		vector<double> out_dists_sqr(num_results);
		nanoflann::KNNResultSet<double> resultSet(num_results);
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		double p0 = V0(ret_indexes[0], 0);
		double p1 = V0(ret_indexes[0], 1);
		double p2 = V0(ret_indexes[0], 2);

		double q0 = V1(idx, 0);
		double q1 = V1(idx, 1);
		double q2 = V1(idx, 2);

		MatrixXd pi(1, 3);
		MatrixXd qi(1, 3);

		ps(idx, 0) = p0;
		ps(idx, 1) = p1;
		ps(idx, 2) = p2;
		qs(idx, 0) = q0;
		qs(idx, 1) = q1;
		qs(idx, 2) = q2;

		pi(0, 0) = p0 - p_bar(0);
		pi(0, 1) = p1 - p_bar(1);
		pi(0, 2) = p2 - p_bar(2);
		qi(0, 0) = q0 - q_bar(0);
		qi(0, 1) = q1 - q_bar(1);
		qi(0, 2) = q2 - q_bar(2);

		A += pi.transpose() * qi;
	}
	cout << A << endl;

	JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
	Matrix3d R = (svd.matrixV()*(svd.matrixU().transpose()));
	Vector3d t = p_bar - (R * q_bar);

	calculate_error(R, t, ps, qs);

	return std::make_tuple(R, t);
	
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d> Icp::point_to_point_subsample(Eigen::MatrixXd V0, Eigen::MatrixXd V1, int V0_factor, int V1_factor)
{
	Eigen::Matrix<double, Dynamic, Dynamic>  mat(V0.rows()/V0_factor, 3);
	for (size_t i = 0; i < V0.rows()/V0_factor; i++)
		for (size_t d = 0; d < 3; d++)
			mat(i, d) = V0(i, d);

	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, Dynamic, Dynamic> >  my_kd_tree_t;
	my_kd_tree_t   mat_index(mat, 10);
	mat_index.index->buildIndex();

	Vector3d p_bar, q_bar;
	p_bar = (1.0 / V1.rows()/V1_factor) * V0.colwise().sum();
	q_bar = (1.0 / V1.rows()/V1_factor) * V1.colwise().sum();
	Eigen::MatrixXd A = MatrixXd::Zero(3, 3);
	ps = Eigen::MatrixXd::Zero(V1.rows() / V1_factor, 3);
	qs = Eigen::MatrixXd::Zero(V1.rows() / V1_factor, 3);
	for (int idx = 0; idx < V1.rows()/V1_factor; idx++) {
		std::vector<double> query_pt(3);
		for (size_t d = 0; d < 3; d++)
			query_pt[d] = V1(idx, d);

		const size_t num_results = 1;
		vector<size_t>   ret_indexes(num_results);
		vector<double> out_dists_sqr(num_results);
		nanoflann::KNNResultSet<double> resultSet(num_results);
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		double p0 = V0(ret_indexes[0], 0);
		double p1 = V0(ret_indexes[0], 1);
		double p2 = V0(ret_indexes[0], 2);

		double q0 = V1(idx, 0);
		double q1 = V1(idx, 1);
		double q2 = V1(idx, 2);

		ps(idx, 0) = p0;
		ps(idx, 1) = p1;
		ps(idx, 2) = p2;
		qs(idx, 0) = q0;
		qs(idx, 1) = q1;
		qs(idx, 2) = q2;

		MatrixXd pi(1, 3);
		MatrixXd qi(1, 3);
		pi(0, 0) = p0 - p_bar(0);
		pi(0, 1) = p1 - p_bar(1);
		pi(0, 2) = p2 - p_bar(2);
		qi(0, 0) = q0 - q_bar(0);
		qi(0, 1) = q1 - q_bar(1);
		qi(0, 2) = q2 - q_bar(2);

		A += pi.transpose() * qi;
	}
	cout << A << endl;

	JacobiSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);
	Matrix3d R = (svd.matrixV()*(svd.matrixU().transpose()));
	Vector3d t = p_bar - (R * q_bar);

	calculate_error(R, t, ps, qs);

	return std::make_tuple(R, t);

}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d> Icp::point_to_plane(Eigen::MatrixXd V0, Eigen::MatrixXd V1)
{
	typedef KDTreeEigenMatrixAdaptor< Eigen::Matrix<double, Dynamic, Dynamic> >  my_kd_tree_t;
	my_kd_tree_t   mat_index(V0, 10);
	mat_index.index->buildIndex();

	Vector3d p_bar, q_bar;
	p_bar = (1.0 / V1.rows()) * V0.colwise().sum();
	q_bar = (1.0 / V1.rows()) * V1.colwise().sum();
	Nv = compute_normals(V0);
	ps = Eigen::MatrixXd::Zero(Nv.rows(), 3);
	qs = Eigen::MatrixXd::Zero(Nv.rows(), 3);

	for (int idx = 0; idx < V1.rows(); idx++) {
		std::vector<double> query_pt(3);
		for (size_t d = 0; d < 3; d++)
			query_pt[d] = V1(idx, d);

		const size_t num_results = 1;
		vector<size_t>   ret_indexes(num_results);
		vector<double> out_dists_sqr(num_results);
		nanoflann::KNNResultSet<double> resultSet(num_results);
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
		mat_index.index->findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParams(10));

		double p0 = V0(ret_indexes[0], 0);
		double p1 = V0(ret_indexes[0], 1);
		double p2 = V0(ret_indexes[0], 2);
		double q0 = V1(idx, 0);
		double q1 = V1(idx, 1);
		double q2 = V1(idx, 2);

		ps(idx, 0) = p0 - p_bar(0);
		ps(idx, 1) = p1 - p_bar(1);
		ps(idx, 2) = p2 - p_bar(2);
		qs(idx, 0) = q0 - q_bar(0);
		qs(idx, 1) = q1 - q_bar(1);
		qs(idx, 2) = q2 - q_bar(2);
	}

	Eigen::MatrixXd A(Nv.rows(), 6);
	Eigen::MatrixXd b(Nv.rows(), 1);

	std::cout << ps.rows() << " = ps" << std::endl;
	std::cout << qs.rows() << " = qs" << std::endl;
	std::cout << Nv.rows() << " = nvs" << std::endl;

	for (int i = 0; i < Nv.rows(); i++) {

		// Computing matrix A
		A(i, 0) = Nv(i, 2) * ps(i, 1) - Nv(i, 1) * ps(i, 2);
		A(i, 1) = Nv(i, 0) * ps(i, 2) - Nv(i, 2) * ps(i, 0);
		A(i, 2) = Nv(i, 1) * ps(i, 0) - Nv(i, 0) * ps(i, 1);
		A(i, 3) = Nv(i, 0);
		A(i, 4) = Nv(i, 1);
		A(i, 5) = Nv(i, 2);

		// Computing b
		b(i, 0) = Nv(i, 0)*qs(i, 0) + Nv(i, 1)*qs(i, 1) + Nv(i, 2)*qs(i, 2) - Nv(i, 0)*ps(i, 0) - Nv(i, 1)*ps(i, 1) - Nv(i, 2)*ps(i, 2);
	}

	Eigen::MatrixXd x(6, 1);

	JacobiSVD<Eigen::MatrixXd> svd(A, ComputeThinU | ComputeThinV);
	Eigen::MatrixXd U = svd.matrixU();
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::MatrixXd I = svd.singularValues().asDiagonal().inverse();

	Eigen::MatrixXd pseudoInv_A = V * I * U.transpose();

	x = pseudoInv_A * b;
	Eigen::Vector3d t = Eigen::Vector3d(x(3), x(4), x(5));
	Eigen::Matrix3d R = get_rotation_matrix(x);

	return std::make_tuple(R, t);
}

Eigen::MatrixXd Icp::compute_normals(Eigen::MatrixXd p_cloud)
{
	Eigen::MatrixXd norm_matrix(p_cloud.rows(), 3);
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

Eigen::Matrix3d Icp::get_rotation_matrix(Eigen::MatrixXd x) {

	Eigen::Matrix3d R(3, 3);

	R(0, 0) = cos(x(2)) * cos(x(1));
	R(0, 1) = -sin(x(2)) * cos(x(0)) + cos(x(2)) * sin(x(1)) * sin(x(0));
	R(0, 2) = sin(x(2)) * sin(x(0)) + cos(x(2)) * sin(x(1)) * cos(x(0));
	R(1, 0) = sin(x(2)) * cos(x(1));
	R(1, 1) = cos(x(2)) * cos(x(0)) + sin(x(2)) * sin(x(1)) * sin(x(0));
	R(1, 2) = -cos(x(2)) * sin(x(0)) + sin(x(2)) * sin(x(1)) * cos(x(0));
	R(2, 0) = -sin(x(1));
	R(2, 1) = cos(x(1)) * sin(x(0));
	R(2, 2) = cos(x(1)) * cos(x(0));

	return R;
}

Eigen::MatrixXd Icp::add_GaussNoise(Eigen::MatrixXd p_cloud, float sigma)
{
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, sigma);

	for (int i = 0; i<p_cloud.rows(); i++) {
		for (int j = 0; j < p_cloud.cols(); j++)
		{
			double noise = distribution(generator);
			p_cloud(i, j) += 0.01 * noise;
		}
	}
	return p_cloud;
}

void Icp::calculate_error(Matrix3d R, Vector3d t, MatrixXd ps, MatrixXd qs)
{
	Vector3d difference;
	Icp::error = 1.0;

	for (int i = 0; i < ps.rows(); i++)
	{
		difference = ps.row(i).transpose() - R * qs.row(i).transpose() - t;
		Icp::error += difference.norm();
	}
	Icp::error /= ps.rows();
	cout << error << " = error" << endl;
}