#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include "igl/readOFF.h"
#include "igl/viewer/Viewer.h"
#include "igl/jet.h"
#include "Helper.h"
#include <GenEigsSolver.h>
#include <MatOp/SparseGenMatProd.h>
#include <iostream>
#include <SymEigsShiftSolver.h>
#include <vector>

using namespace Spectra;


class Curvature
{
private:
	Curvature() {};
	~Curvature() {};
public:
	static Eigen::SparseMatrix<double> Curvature::ConstructUniformLaplace(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces);
	static Eigen::MatrixXd Curvature::ComputeMeanCurvature(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces, int discretizationType);
	static Eigen::MatrixXd Curvature::ComputeGaussianCurvature(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces);
	static Eigen::SparseMatrix<double> Curvature::ComputeCotanDiscretization(const Eigen::MatrixXd& cloud_verts, const Eigen::MatrixXi& cloud_faces);
	static Eigen::MatrixXd Curvature::ComputeMeshReconstruction(Eigen::MatrixXd cloud_verts, Eigen::MatrixXi cloud_faces, int k);
};

