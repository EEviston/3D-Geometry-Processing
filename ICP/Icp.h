#pragma once
#include <igl/fit_plane.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_face_normals.h>
#include "tutorial_shared_path.h"
#include <tuple>
#include <iostream>
#include <vector>
#include <nanoflann.hpp>
#include <chrono>
#include <random>

using namespace Eigen;
using namespace nanoflann;
using namespace std;

class Icp
{
public:
	double error = 1, norm_error = 1;
	MatrixXd Nv;
	
public:
	Icp();
	~Icp();
	std::tuple<Eigen::Matrix3d, Eigen::Vector3d> point_to_point(Eigen::MatrixXd V0, Eigen::MatrixXd V1);
	std::tuple<Eigen::Matrix3d, Eigen::Vector3d> point_to_point_subsample(Eigen::MatrixXd V0, Eigen::MatrixXd V1, int V0_factor, int V1_factor);
	std::tuple<Eigen::Matrix3d, Eigen::Vector3d> point_to_plane(Eigen::MatrixXd V0, Eigen::MatrixXd V1);
	Eigen::MatrixXd compute_normals(Eigen::MatrixXd p_cloud);
	Eigen::Matrix3d get_rotation_matrix(Eigen::MatrixXd x);
	Eigen::MatrixXd add_GaussNoise(Eigen::MatrixXd p_cloud, float sigma);
	void calculate_error(Matrix3d R, Vector3d t, MatrixXd ps, MatrixXd qs);
};

