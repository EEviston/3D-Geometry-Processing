#pragma once
#include <igl/fit_plane.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/igl_inline.h>
#include <iostream>
#include <vector>
#include <nanoflann.hpp>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <random>
#include <igl\bounding_box_diagonal.h>

#define PI 3.14159265

using namespace Eigen;
using namespace nanoflann;
using namespace std;

class Helper
{
private:
	Helper() {}
	~Helper() {}
public:
	static Eigen::MatrixXd Helper::ComputeNormals(Eigen::MatrixXd p_cloud);
	static double Helper::FindArea(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, const Eigen::Vector3d& point3);
	static double Helper::FindAngle(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, const Eigen::Vector3d& point3);
	static double Helper::FindCotan(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2, const Eigen::Vector3d& point3);
	static Eigen::SparseMatrix<double> Helper::FindCotanWeights(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces);
	static Eigen::SparseMatrix<double> Helper::FindDiagonalArea(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces);
	static Eigen::SparseMatrix<double> Helper::ComputeInverseSparse(Eigen::SparseMatrix<double> M);
	static Eigen::MatrixXd Helper::AddGaussNoise(Eigen::MatrixXd cloud_verts, float sigma);
	static bool Helper::compare_head(const Eigen::VectorXd& lhs, const Eigen::VectorXd& rhs);
	static double Helper::findLambda(Eigen::MatrixXd& cloud_verts, double percentage);
};

