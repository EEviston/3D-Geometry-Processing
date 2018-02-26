#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
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
#include <c:\Users\XPS\Desktop\libigl\tutorial\build\coursework1\Icp.h>

using namespace Eigen;
using namespace std;
using namespace nanoflann;

Eigen::MatrixXd V0,V1,V2,V3,V4;
Eigen::MatrixXi F0,F1,F2,F3,F4;
Eigen::MatrixXd V;
Eigen::MatrixXi F;
double threshold = 0.01;
int iteration = 0;
Icp my_icp;

bool key_down(igl::viewer::Viewer& viewer, unsigned char key, int modifier)
{
  std::cout<<"Key: "<<key<<" "<<(unsigned int)key<<std::endl;
  if (key == '1')
  {
	  iteration = 1;
	  viewer.data.clear();
	  clock_t begin = clock();
	  cout << "iteration: " << iteration << endl;
	  std::tuple<Matrix3d, Vector3d> R_t = my_icp.point_to_point(V0, V1);
	  V1 = (V1 + get<1>(R_t).replicate(1, V1.rows()).transpose())* get<0>(R_t);
	  iteration++;
	  clock_t end = clock();
	  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	  std::cout << elapsed_secs << std::endl;

	  viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
	  viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
	  viewer.core.align_camera_center(V0, F0);
  }
  // COMPUTE AND DRAW NORMALS OF TARGET
  else if (key == '2')
  {
    viewer.data.clear();
	viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
	viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
	viewer.data.add_edges(V0, V0 + my_icp.compute_normals(V0)*.005, Eigen::RowVector3d(1, 0, 0));
	//viewer.data.add_points(my_icp.meancenter, Eigen::RowVector3d(1, 0, 0));
    viewer.core.align_camera_center(V0,F0);
	std::cout << "finished normal" << std::endl;
  }

  // ADD NOISE
  else if (key == '3')
  {
	  viewer.data.clear();
	  V1 = my_icp.add_GaussNoise(V1,0.7);
	  viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
	  viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
	  viewer.core.align_camera_center(V0, F0);
  }

  // POINT TO PLANE
  else if (key == '4') {

	  iteration = 1;
	  viewer.data.clear();

	  clock_t begin = clock();
	 
	  cout << "iteration: " << iteration << endl;
	  std::tuple<Matrix3d, Vector3d> R_t = my_icp.point_to_plane(V0, V1);
	  V1 = (V1 + get<1>(R_t).replicate(1, V1.rows()).transpose())* get<0>(R_t);
	  iteration++;
	  clock_t end = clock();
	  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	  std::cout << elapsed_secs << std::endl;

	  viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
	  viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
	  viewer.core.align_camera_center(V0, F0);
	  viewer.core.point_size = 2.0;
  }

  // 5 MODELS
  else if (key == '5')
  {
	  viewer.data.clear();
	  igl::readOFF(TUTORIAL_SHARED_PATH "/bun090.off", V2, F2);
	  igl::readOFF(TUTORIAL_SHARED_PATH "/bun180.off", V3, F3);
	  igl::readOFF(TUTORIAL_SHARED_PATH "/bun270.off", V4, F4);
	  
	  iteration = 1;
	  while (iteration < 8)
	  {
		  std::tuple<Matrix3d, Vector3d> R1_t1 = my_icp.point_to_point_multiple(V0, V1);
		  V1 = (V1 + get<1>(R1_t1).replicate(1, V1.rows()).transpose())* get<0>(R1_t1);
		  iteration++;
	  }
	  iteration = 1;
	  while (iteration < 8)
	  {
		  std::tuple<Matrix3d, Vector3d> R2_t2 = my_icp.point_to_point_multiple(V1, V2);
		  V2 = (V2 + get<1>(R2_t2).replicate(1, V2.rows()).transpose())* get<0>(R2_t2);
		  iteration++;
	  }
	  iteration = 1;
	  while (iteration < 8)
	  {
		  std::tuple<Matrix3d, Vector3d> R3_t3 = my_icp.point_to_point_multiple(V2, V3);
		  V3 = (V3 + get<1>(R3_t3).replicate(1, V3.rows()).transpose())* get<0>(R3_t3);
		  iteration++;
	  }
	  iteration = 1;
	  while (iteration < 8)
	  {
		  std::tuple<Matrix3d, Vector3d> R4_t4 = my_icp.point_to_point_multiple(V3, V4);
		  V4 = (V4 + get<1>(R4_t4).replicate(1, V4.rows()).transpose())* get<0>(R4_t4);
		  iteration++;
	  }  
	  viewer.core.align_camera_center(V0, F0);
	  viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
	  viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
	  viewer.data.add_points(V2, Eigen::RowVector3d(0, 0, 1));
	  viewer.data.add_points(V3, Eigen::RowVector3d(0, 1, 1));
	  viewer.data.add_points(V4, Eigen::RowVector3d(1, 0, 0));
  }

  // ROTATION
  else if (key == '6')
  {
	  igl::readOFF(TUTORIAL_SHARED_PATH "/bun000.off", V1, F1);
	  viewer.data.clear();
	  Matrix3d R;
	  R = AngleAxisd(M_PI / 6, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
	  V1 *= R;
	  viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
	  viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
	  viewer.core.align_camera_center(V0, F0);
  }
  //SUBSAMPLE
  if (key == '7')
  {
	  iteration = 1;
	  viewer.data.clear();
	  clock_t begin = clock();
	  cout << "iteration: " << iteration << endl;
	  std::tuple<Matrix3d, Vector3d> R_t = my_icp.point_to_point_subsample(V0, V1, 1, 2);
	  V1 = (V1 + get<1>(R_t).replicate(1, V1.rows()).transpose())* get<0>(R_t);
	  iteration++;
	  clock_t end = clock();
	  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	  std::cout << elapsed_secs << std::endl;

	  viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
	  viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
	  viewer.core.align_camera_center(V0, F0);
  }

  if (key == '9')
  {
	  iteration = 1;
	  viewer.data.clear();
	  clock_t begin = clock();
	  double error = 1.0;
	  while (error > threshold)
	  {
		  cout << "iteration: " << iteration << endl;
		  std::tuple<Matrix3d, Vector3d> R_t = my_icp.point_to_point(V0, V1);
		  V1 = (V1 + get<1>(R_t).replicate(1, V1.rows()).transpose())* get<0>(R_t);
		  error = my_icp.error;
		  iteration++;
	  }
	  clock_t end = clock();
	  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	  std::cout << elapsed_secs << std::endl;

	  viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
	  viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
	  viewer.core.align_camera_center(V0, F0);
  }
  return false;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXi> matrixConcatenate(Eigen::MatrixXd V0, Eigen::MatrixXd V1, Eigen::MatrixXi F0, Eigen::MatrixXi F1)
{
	Eigen::MatrixXd v_out(V0.rows() + V1.rows(), V0.cols());
	v_out << V0, V1;
	Eigen::MatrixXi f_out(F0.rows() + F1.rows(), F0.cols());
	f_out << F0, (F1.array() + V0.rows());
	return std::make_tuple(v_out, f_out);
}


int main(int argc, char *argv[])
{
  igl::readOFF(TUTORIAL_SHARED_PATH "/bun000.off", V0, F0);
  igl::readOFF(TUTORIAL_SHARED_PATH "/bun045.off", V1, F1);
	
  std::cout<<R"(
1 Perform point to point ICP manually
2 Compute and draw normals of target
3 Add noise to source (sigma = 0.7 by default)
4 Perform point to plane ICP manually
5 Load 3 more models and manually do ICP to align
6 Add rotation to source (30 degrees by default)
7 Subsample (1/2 of source by default) and manually do ICP
9 Perform point to point ICP automatically until convergence
    )";
  
  igl::viewer::Viewer viewer;
  viewer.callback_key_down = &key_down;

  viewer.data.set_points(V0, Eigen::RowVector3d(1, 1, 0));
  viewer.data.add_points(V1, Eigen::RowVector3d(1, 0, 1));
  viewer.core.align_camera_center(V0, F0);
  viewer.core.point_size = 2.0;

  viewer.launch();
}
