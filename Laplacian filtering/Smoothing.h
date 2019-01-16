#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include "igl/readOFF.h"
#include "igl/viewer/Viewer.h"
#include "igl/jet.h"
#include "Helper.h"

class Smoothing
{
private:
	Smoothing() {};
	~Smoothing() {};
public:
	static Eigen::MatrixXd ExplicitSmoothing(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces, int discretizationType, double lambda);
	static Eigen::MatrixXd ImplicitSmoothing(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces, double lambda);
	static Eigen::MatrixXd ImplicitSmoothingTest(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces, double lambda);
};

