/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#include "UtilsGtsam.h"
#include "GtsamIO.h"
#include "LandmarkMotionTernaryFactor.h"
#include "Point3DFactor.h"

// #include "parser/dynaslam_like.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <thread>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/geometry/Pose2.h>
#include <type_traits>

#include <boost/type_traits.hpp>


using namespace dsc;
using namespace gtsam;

struct FakeClass
{
  int data;

  enum
  {
    dimension = 0
  };
  void print(const std::string& str = "") const
  {
  }
  bool equals(const FakeClass& other, double tol = 1e-9) const
  {
    return data == other.data;
  }
  size_t dim() const
  {
    return 0;
  }
  FakeClass retract(const Vector&, OptionalJacobian<dimension, dimension> H1 = {},
                    OptionalJacobian<dimension, dimension> H2 = {}) const
  {
    return FakeClass();
  }
  Vector localCoordinates(const FakeClass&, OptionalJacobian<dimension, dimension> H1 = {},
                          OptionalJacobian<dimension, dimension> H2 = {}) const
  {
    return Vector();
  }
};

namespace gtsam
{
template <>
struct traits<FakeClass> : public internal::Manifold<FakeClass>
{
};
}  // namespace gtsam

template <>
bool gtsam::io::value<FakeClass>::write(std::ostream& os, const FakeClass& t)
{
  os << t.data;
  return os.good();
}

template <>
bool gtsam::io::value<FakeClass>::read(std::istream& is, FakeClass& t)
{
  is >> t.data;
  return is.good();
}

// class Motion3 : public gtsam::Pose3 {

// public:
//   operator gtsam::Pose3() const { return static_cast<gtsam::Pose3>(*this); }

// };

// namespace gtsam {
// template <>
// struct traits<Motion3> : public internal::LieGroup<Motion3> {};

// template <>
// struct traits<const Motion3> : public internal::LieGroup<Motion3> {};

// // bearing and range traits, used in RangeFactor
// // template <>
// // struct Bearing<Motion3, Point3> : Bearing<Pose3, Point3> {};

// // template<>
// // struct Bearing<Pose3, Pose3> : HasBearing<Pose3, Pose3, Unit3> {};

// template <typename T>
// struct Range<Motion3, T> : Range<Pose3, T> {};

// }  // namespace gtsam

// GTSAM_REGISTER_TYPE(VERTEX_MOTION_SE3:QUAT, Motion3);

TEST(testGtsamUtils, ObjectOdometryFactor)
{
  dyna_like::ObjectOdometryFactor motion_factor(1, 2, 3, nullptr);
  gtsam::Pose3 random_motion = dsc::testing::makeRandomPose(42);
  gtsam::Pose3 previous_pose = dsc::testing::makeRandomPose(12);
  gtsam::Pose3 current_pose = random_motion * previous_pose;

  gtsam::Matrix J1, J2, J3;
  EXPECT_TRUE(gtsam::assert_equal(gtsam::Vector6::Constant(0),
                                  motion_factor.evaluateError(random_motion, previous_pose, current_pose, J1, J2, J3)));

  gtsam::print(J1, "J1= ");
  gtsam::print(J2, "J2= ");
  gtsam::print(J3, "J3= ");
}

// TEST(testGtsamUtils, ObjectCentricLandmarkMotionFactor)
// {
//   dyna_like::ObjectCentricLandmarkMotionFactor motion_factor(1, 2, 3, 4, nullptr);

//   gtsam::Pose3 motion(gtsam::Rot3::Identity(), gtsam::Point3(1, 0, 0));
//   gtsam::Point3 point_l(10, 10, 10);
//   // gtsam::Pose3 motion(gtsam::Rot3::RzRyRx(0, 0.01, 0.4), gtsam::Point3(1, 2, 3));
//   // gtsam::Point3 point_l(6, 8, 8);

//   gtsam::Pose3 previous_l(gtsam::Rot3::Identity(), gtsam::Point3(10, 0, 0));
//   gtsam::Pose3 current_l(gtsam::Rot3::Identity(), gtsam::Point3(11, 0, 0));

//   gtsam::Matrix J1, J2, J3, J4;
//   gtsam::Vector error = motion_factor.evaluateError(motion, previous_l, current_l, point_l, J1, J2, J3, J4);
//   EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(0, 0, 0), error));

//   gtsam::print(J1, "J1= ");
//   gtsam::print(J2, "J2= ");
//   gtsam::print(J3, "J3= ");
//   gtsam::print(J4, "J4= ");
// }

TEST(testGtsamUtils, ObjectCentricPoint3DFactor)
{
  gtsam::Point3 measured(5, 5, 5);
  dyna_like::ObjectCentricPoint3DFactor point_factor(1, 2, 3, measured, nullptr);

  gtsam::Pose3 cam_pose(gtsam::Rot3::RzRyRx(0, 0.01, 0.4), gtsam::Point3(1, 0, 0));
  gtsam::Point3 point(5, 5, 5);
  gtsam::Pose3 object_pose(gtsam::Rot3::Identity(), gtsam::Point3(11, 0, 0));

  gtsam::Matrix J1, J2, J3;
  gtsam::Vector error = point_factor.evaluateError(cam_pose, object_pose, point, J1, J2, J3);
  // EXPECT_TRUE(gtsam::assert_equal(gtsam::Point3(0, 0, 0), error));

  gtsam::print(J1, "J1= ");
  gtsam::print(J2, "J2= ");
  gtsam::print(J3, "J3= ");
}

TEST(testGtsamUtils, getKeysForBasicHessian)
{
  NonlinearFactorGraph graph;

  Pose2 priorMean(0.0, 0.0, 0.0);  // prior at origin
  noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
  // graph.emplace_shared<PriorFactor<Pose2> >(1, priorMean, priorNoise);

  // Add odometry factors
  Pose2 odometry(2.0, 1.0, 1.0);
  // For simplicity, we will use the same noise model for each odometry factor
  noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
  // Create odometry (Between) factors between consecutive poses
  graph.emplace_shared<BetweenFactor<Pose2>>(1, 2, odometry, odometryNoise);
  graph.emplace_shared<BetweenFactor<Pose2>>(2, 3, odometry, odometryNoise);
  graph.print("\nFactor Graph:\n");  // print

  // Create the data structure to hold the initialEstimate estimate to the solution
  // For illustrative purposes, these have been deliberately set to incorrect values
  Values initial;
  initial.insert(2, Pose2(2.3, 0.1, -0.2));
  initial.insert(3, Pose2(4.1, 0.1, 0.1));
  initial.insert(1, Pose2(0.5, 0.1, 0.2));

  GaussianFactorGraph::shared_ptr linear = graph.linearize(initial);
  Scatter scatter(*linear);
  HessianFactor combined(*linear, scatter);

  for (const auto s : scatter)
  {
    std::cout << s.toString() << " ";
  }
  std::cout << std::endl;
  // print(combined.information());
  combined.print();
}

TEST(testGtsamUtils, isGtsamFactor)
{
  EXPECT_TRUE(is_gtsam_factor_v<gtsam::BetweenFactor<gtsam::Pose3>>);
  EXPECT_TRUE(is_gtsam_factor_v<dsc::LandmarkMotionTernaryFactor>);
  EXPECT_TRUE(is_gtsam_factor_v<dsc::Point3DFactor>);

  EXPECT_FALSE(is_gtsam_factor_v<gtsam::Pose3>);
  EXPECT_FALSE(is_gtsam_factor_v<gtsam::Point3>);
}

TEST(testGtsamUtils, nonlinearFactorSize)
{
  EXPECT_EQ(gtsam::getFactorSize<gtsam::BetweenFactor<gtsam::Pose3>>(), 2u);
  EXPECT_EQ(gtsam::getFactorSize<dsc::LandmarkMotionTernaryFactor>(), 3u);
  EXPECT_EQ(gtsam::getFactorSize<dsc::Point3DFactor>(), 2u);
}

GTSAM_REGISTER_TYPE(EDGE_FAKE_CLASS, FakeClass);

TEST(testGtsamUtils, testRegisterFactorType)
{
  EXPECT_EQ(gtsam::io::tag_name<gtsam::BetweenFactor<gtsam::Pose3>>(), "EDGE_SE3:QUAT");
  auto creator = gtsam::io::Factory::instance()->getCreator<gtsam::BetweenFactor<gtsam::Pose3>>();
  EXPECT_EQ(creator->className(), "gtsam::BetweenFactor<gtsam::Pose3>");
  EXPECT_EQ(creator->tagName(), "EDGE_SE3:QUAT");
  EXPECT_EQ(creator->type(), io::Types::FACTOR);
}

TEST(testGtsamUtils, testNames)
{
  EXPECT_EQ("VERTEX_TRACKXYZ", gtsam::io::tag_name<gtsam::Point3>());
  EXPECT_EQ("VERTEX_SE3:QUAT", gtsam::io::tag_name<gtsam::Pose3>());
  // EXPECT_EQ("VERTEX_MOTION_SE3:QUAT", gtsam::io::tag_name<Motion3>());
  EXPECT_EQ("EDGE_SE3_MOTION", gtsam::io::tag_name<dsc::LandmarkMotionTernaryFactor>());
  EXPECT_EQ("EDGE_SE3_TRACKXYZ", gtsam::io::tag_name<dsc::Point3DFactor>());
}

TEST(testGtsamUtils, testRegisterValueType)
{
  EXPECT_EQ(gtsam::io::tag_name<FakeClass>(), "EDGE_FAKE_CLASS");
  // this is very gross
  auto creator = gtsam::io::Factory::instance()->getCreator<FakeClass>();
  EXPECT_EQ(creator->className(), "FakeClass");
  EXPECT_EQ(creator->tagName(), "EDGE_FAKE_CLASS");
  EXPECT_EQ(creator->type(), io::Types::VALUE);
}

TEST(testGtsamUtils, testBasicSaveValue)
{
  FakeClass fc;
  fc.data = 10;
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<FakeClass>();

  std::stringstream os;
  gtsam::Pose3 p;
  io::ValueType v(1, genericValue(fc));
  // gtsam::Values::KeyValuePair v(1, gtsam::genericValue(p));
  EXPECT_TRUE(creator->save(os, v));
  std::string output;
  std::getline(os, output);

  std::string expected_output = "EDGE_FAKE_CLASS 1 10";
  EXPECT_EQ(expected_output, output);
}

TEST(testGtsamUtils, testBasicConstructValue)
{
  std::stringstream ss;
  ss << "EDGE_FAKE_CLASS 1 16";
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<FakeClass>();

  io::ValueType value_type = creator->construct(ss);
  EXPECT_EQ(value_type.key(), 1);
  EXPECT_EQ(value_type.value<FakeClass>().data, 16);
}

TEST(testGtsamUtils, testfullMatrixFromUpperTriangular)
{
  std::stringstream ss;
  ss << "100 0 0 100 0 100";
  gtsam::Matrix Info = io::internal::fullMatrixFromUpperTriangular(ss, 3);
  gtsam::Matrix33 expected;
  expected << 100, 0, 0, 0, 100, 0, 0, 0, 100;
  EXPECT_TRUE(gtsam::assert_equal(expected, Info));
}

TEST(testGtsamUtils, testBasicSaveBetweenFactorPose3)
{
  gtsam::Pose3 measured = gtsam::Pose3::Identity();
  gtsam::noiseModel::Base::shared_ptr noise_model = gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

  gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr factor =
      boost::make_shared<gtsam::BetweenFactor<gtsam::Pose3>>(4, 5, measured, noise_model);
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<gtsam::BetweenFactor<gtsam::Pose3>>();

  std::stringstream os;
  EXPECT_TRUE(creator->save(os, factor));
  std::string output;
  std::getline(os, output);

  EXPECT_EQ(factor->dim(), 6u);
  std::string expected_output = "EDGE_SE3:QUAT 4 5 0 0 0 0 0 0 1 100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 100 0 0 100 0 100";
  EXPECT_EQ(expected_output, output);
}

TEST(testGtsamUtils, testBasicConstructBetweenFactor)
{
  std::stringstream ss;
  ss << "EDGE_SE3:QUAT 4 5 1 2 3 0 0 0 1 50 0 0 0 0 0 50 0 0 0 0 50 0 0 0 50 0 0 50 0 50";
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<gtsam::BetweenFactor<gtsam::Pose3>>();

  gtsam::NonlinearFactor::shared_ptr factor = creator->construct(ss);
  gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr factor_cast =
      boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor);
  EXPECT_TRUE(factor_cast != nullptr);
  const KeyVector& keys = factor->keys();
  EXPECT_EQ(keys.size(), 2u);
  EXPECT_EQ(keys[0], 4u);
  EXPECT_EQ(keys[1], 5u);

  gtsam::Matrix66 expected_info_matrix = gtsam::Matrix66::Identity();
  expected_info_matrix = 50 * expected_info_matrix;

  gtsam::Pose3 expected_pose(gtsam::Rot3::Identity(), gtsam::Point3(1, 2, 3));
  EXPECT_TRUE(gtsam::assert_equal(expected_pose, factor_cast->measured()));

  auto gaussianModel = boost::dynamic_pointer_cast<noiseModel::Gaussian>(factor_cast->noiseModel());
  EXPECT_TRUE(gtsam::assert_equal(expected_info_matrix, gaussianModel->information()));
}

TEST(testGtsamUtils, testLoadPoint3)
{
  std::stringstream ss;
  ss << "VERTEX_TRACKXYZ 262 1.64351 1.87958 12.997";
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<gtsam::Point3>();

  io::ValueType value = creator->construct(ss);
  gtsam::Point3 point = value.value<gtsam::Point3>();

  EXPECT_TRUE(gtsam::assert_equal(point, gtsam::Point3(1.64351, 1.87958, 12.997)));
  EXPECT_EQ(value.key(), 262);
}

TEST(testGtsamUtils, testLoadPose3)
{
  std::stringstream ss;
  ss << "VERTEX_SE3:QUAT 51137 -3.12349 0.211163 11.642 -0.0117509 -0.221891 0.000910893 0.975";
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<gtsam::Pose3>();

  io::ValueType value = creator->construct(ss);
  gtsam::Pose3 point = value.value<gtsam::Pose3>();

  // NOTE: Rot3 takes x,x,y,z but we save in the format x,y,z,w
  EXPECT_TRUE(gtsam::assert_equal(point, gtsam::Pose3(gtsam::Rot3(0.975, -0.0117509, -0.221891, 0.000910893),
                                                      gtsam::Point3(-3.12349, 0.211163, 11.642))));
  EXPECT_EQ(value.key(), 51137);
}

TEST(testGtsamUtils, testLoadLandmarkTernaryFactor)
{
  std::stringstream ss;
  ss << "EDGE_SE3_MOTION 16730 18241 16899 0 0 0 0.01 0 0 0.01 0 0.01";
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<dsc::LandmarkMotionTernaryFactor>();

  gtsam::NonlinearFactor::shared_ptr factor = creator->construct(ss);
  dsc::LandmarkMotionTernaryFactor::shared_ptr factor_cast =
      boost::dynamic_pointer_cast<dsc::LandmarkMotionTernaryFactor>(factor);

  EXPECT_TRUE(factor_cast != nullptr);
  const KeyVector& keys = factor->keys();
  EXPECT_EQ(keys.size(), 3u);
  EXPECT_EQ(factor_cast->PreviousPointKey(), 16730);
  EXPECT_EQ(factor_cast->CurrentPointKey(), 18241);
  EXPECT_EQ(factor_cast->MotionKey(), 16899);

  gtsam::Matrix33 expected_info_matrix = gtsam::Matrix33::Identity();
  expected_info_matrix = 0.01 * expected_info_matrix;
  auto gaussianModel = boost::dynamic_pointer_cast<noiseModel::Gaussian>(factor_cast->noiseModel());
  // faileding becuase its not gaussian it is robust (currently hardcoded in gtsam::io)
  //  ASSERT_TRUE(gaussianModel != nullptr);
  //  EXPECT_TRUE(gtsam::assert_equal(expected_info_matrix, gaussianModel->information()));
}

TEST(testGtsamUtils, testLoadLandmarkTernaryFactorWithFrameAndObjectID)
{
  std::stringstream ss;
  ss << "EDGE_SE3_MOTION 138304 138816 138815 0 0 0 0.01 0 0 0.01 0 0.01 6 122 138713";
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<dsc::LandmarkMotionTernaryFactor>();

  gtsam::NonlinearFactor::shared_ptr factor = creator->construct(ss);
  dsc::LandmarkMotionTernaryFactor::shared_ptr factor_cast =
      boost::dynamic_pointer_cast<dsc::LandmarkMotionTernaryFactor>(factor);

  EXPECT_TRUE(factor_cast != nullptr);
  const KeyVector& keys = factor->keys();
  EXPECT_EQ(keys.size(), 3u);
  EXPECT_EQ(factor_cast->PreviousPointKey(), 138304);
  EXPECT_EQ(factor_cast->CurrentPointKey(), 138816);
  EXPECT_EQ(factor_cast->MotionKey(), 138815);
  EXPECT_EQ(factor_cast->objectLabel(), 6);
  EXPECT_EQ(factor_cast->frameId(), 122);
  EXPECT_EQ(factor_cast->camera_pose_key_, 138713);

  gtsam::Matrix33 expected_info_matrix = gtsam::Matrix33::Identity();
  expected_info_matrix = 0.01 * expected_info_matrix;
  // auto gaussianModel = boost::dynamic_pointer_cast<noiseModel::Gaussian>(factor_cast->noiseModel());
  // EXPECT_TRUE(gtsam::assert_equal(expected_info_matrix, gaussianModel->information()));
}

TEST(testGtsamUtils, testLoadPoint3DFactor)
{
  std::stringstream ss;
  ss << "EDGE_SE3_TRACKXYZ 16827 18319 -5.47238 0.147258 14.8687 0.0125 0 0 0.0125 0 0.0125";
  auto creator = gtsam::io::Factory::instance()->getCreatorTyped<dsc::Point3DFactor>();

  gtsam::NonlinearFactor::shared_ptr factor = creator->construct(ss);
  Point3DFactor::shared_ptr factor_cast = boost::dynamic_pointer_cast<Point3DFactor>(factor);

  EXPECT_TRUE(factor_cast != nullptr);
  const KeyVector& keys = factor->keys();
  EXPECT_EQ(keys.size(), 2u);
  EXPECT_EQ(factor_cast->key1(), 16827);  // should be pose key
  EXPECT_EQ(factor_cast->key2(), 18319);  // should be pose key

  gtsam::Matrix33 expected_info_matrix = gtsam::Matrix33::Identity();
  expected_info_matrix = 0.0125 * expected_info_matrix;
  // auto gaussianModel = boost::dynamic_pointer_cast<noiseModel::Gaussian>(factor_cast->noiseModel());
  // EXPECT_TRUE(gtsam::assert_equal(expected_info_matrix, gaussianModel->information()));
}

TEST(testGtsamUtils, loadFromIOStream)
{
  io::GraphFileParser parser;

  std::stringstream input;
  input << "EDGE_SE3:QUAT 4 5 0 0 0 0 0 0 1 50 0 0 0 0 0 50 0 0 0 0 50 0 0 0 50 0 0 50 0 50"
        << "\n";
  input << "EDGE_FAKE_CLASS 1 16\n";
  input << "EDGE_FAKE_CLASS 3 12";

  // std::ifstream stream = input.rdbuf();

  io::ValuesFactorGraphPair pair = parser.load(input);
  gtsam::Values values = pair.first;
  gtsam::NonlinearFactorGraph graph = pair.second;

  EXPECT_EQ(values.size(), 2u);
  EXPECT_TRUE(values.exists(1));
  EXPECT_TRUE(values.exists(3));

  EXPECT_EQ(graph.size(), 1u);
}

TEST(testGtsamUtils, loadFromIOStreamBasicFactorGraph)
{
  io::GraphFileParser parser;

  std::stringstream input;
  // two poses and an edge connecting them 1->2
  input << "EDGE_SE3:QUAT 1 2 0 0 0 0 0 0 1 50 0 0 0 0 0 50 0 0 0 0 50 0 0 0 50 0 0 50 0 50"
        << "\n";
  input << "VERTEX_SE3:QUAT 1 -0.0158347 0.00434591 0.353949 -0.00222314 -0.00729577 -3.10337e-05 0.999971\n";
  input << "VERTEX_SE3:QUAT 2 0 0 0 0 0 0 1\n";
  // 1 vertices representing static points
  input << "VERTEX_TRACKXYZ 3 0.25851 1.90162 13.106\n";
  // 2 edges connecting the point to the two poses
  input << "EDGE_SE3_TRACKXYZ 1 3 7.82542 0.721465 19.1765 0.0125 0 0 0.0125 0 0.0125\n";
  input << "EDGE_SE3_TRACKXYZ 2 3 -4.17057 -0.0706339 13.2239 0.0125 0 0 0.0125 0 0.0125\n";
  // 2 vertices representing points on a dynamic object at time t=0 (ie pose key = 1)
  input << "VERTEX_TRACKXYZ 6 -5.27136 0.0628517 13.454\n";
  input << "VERTEX_TRACKXYZ 7 -5.12682 0.0456925 13.611\n";
  // 2 vertices representing points on a dynamic object at time t=1 (ie pose key = 2)
  input << "VERTEX_TRACKXYZ 8 -5.09327 0.0456927 13.6584\n";
  input << "VERTEX_TRACKXYZ 9 -5.04836 0.0456944 13.7228\n";
  // motion SE(3) for all dynamic points in this graph
  input << "VERTEX_SE3:QUAT 10 0 0 0 0 0 0 1\n";
  // landmark ternary edge connection points 6 to 8 with motion 10
  input << "EDGE_SE3_MOTION 6 8 10 0 0 0 0.01 0 0 0.01 0 0.01\n";
  input << "EDGE_SE3_MOTION 7 9 10 0 0 0 0.01 0 0 0.01 0 0.01\n";

  // total vales should be 2 poses at keys (1, 2); 1 static vertex and 4 dynamic vertices and 1 object motion
  //  == 8 vertices
  // total edges should be 1 between factor (1 -> 2) on the poses, 2 point3 edges between pose1,2 and the point3 (3)
  //  and then two landmark motion edges
  //  == 5 edges

  io::ValuesFactorGraphPair pair = parser.load(input);
  gtsam::Values values = pair.first;
  gtsam::NonlinearFactorGraph graph = pair.second;

  EXPECT_EQ(values.size(), 8u);
  EXPECT_EQ(graph.size(), 5u);

  EXPECT_TRUE(values.exists(1));
  EXPECT_TRUE(values.exists(2));
  EXPECT_TRUE(values.exists(3));
  EXPECT_TRUE(values.exists(6));
  EXPECT_TRUE(values.exists(7));
  EXPECT_TRUE(values.exists(8));
  EXPECT_TRUE(values.exists(9));
  EXPECT_TRUE(values.exists(10));

  {
    auto p = values.at<gtsam::Pose3>(1);
    EXPECT_TRUE(gtsam::assert_equal(p, gtsam::Pose3(gtsam::Rot3(0.999971, -0.00222314, -0.00729577, -3.10337e-05),
                                                    gtsam::Point3(-0.0158347, 0.00434591, 0.353949))));
  }
  {
    auto p = values.at<gtsam::Pose3>(2);
    EXPECT_TRUE(gtsam::assert_equal(p, gtsam::Pose3::Identity()));
  }

  {
    auto p = values.at<gtsam::Point3>(3);
    EXPECT_TRUE(gtsam::assert_equal(p, gtsam::Point3(0.25851, 1.90162, 13.106)));
  }
  {
    auto p = values.at<gtsam::Point3>(6);
    EXPECT_TRUE(gtsam::assert_equal(p, gtsam::Point3(-5.27136, 0.0628517, 13.454)));
  }
  {
    auto p = values.at<gtsam::Point3>(7);
    EXPECT_TRUE(gtsam::assert_equal(p, gtsam::Point3(-5.12682, 0.0456925, 13.611)));
  }
  {
    auto p = values.at<gtsam::Point3>(8);
    EXPECT_TRUE(gtsam::assert_equal(p, gtsam::Point3(-5.09327, 0.0456927, 13.6584)));
  }
  {
    auto p = values.at<gtsam::Point3>(9);
    EXPECT_TRUE(gtsam::assert_equal(p, gtsam::Point3(-5.04836, 0.0456944, 13.7228)));
  }
  {
    auto p = values.at<gtsam::Pose3>(10);
    EXPECT_TRUE(gtsam::assert_equal(p, gtsam::Pose3::Identity()));
  }

  {
    gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr factor_cast =
        boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(graph.at(0));
    EXPECT_EQ(factor_cast->key1(), 1);
    EXPECT_EQ(factor_cast->key2(), 2);
  }

  {
    dsc::Point3DFactor::shared_ptr factor_cast = boost::dynamic_pointer_cast<dsc::Point3DFactor>(graph.at(1));
    EXPECT_EQ(factor_cast->key1(), 1);
    EXPECT_EQ(factor_cast->key2(), 3);
  }

  {
    dsc::Point3DFactor::shared_ptr factor_cast = boost::dynamic_pointer_cast<dsc::Point3DFactor>(graph.at(2));
    EXPECT_EQ(factor_cast->key1(), 2);
    EXPECT_EQ(factor_cast->key2(), 3);
  }
}

TEST(testGtsamUtils, testValueTypeBasicConstruction)
{
  gtsam::Pose3 expected_pose = dsc::testing::makeRandomPose(23);
  io::ValueType value_type(10, gtsam::genericValue(expected_pose));

  EXPECT_TRUE(value_type.exists());
  EXPECT_EQ(value_type.key(), 10);

  gtsam::Pose3 pose = value_type.value<gtsam::Pose3>();
  EXPECT_TRUE(expected_pose.equals(pose));
}

TEST(DynaLike, testDynaLikeFormatter)
{
  gtsam::LabeledSymbol object_pose = ObjectPoseKey(10, 2);
  EXPECT_EQ(object_pose.chr(), kObjectPoseKey);
  EXPECT_EQ(object_pose.label(), (unsigned char)10 + '0');
  EXPECT_EQ(object_pose.index(), 2);

  int expected_object_id = object_pose.label() - '0';
  EXPECT_EQ(expected_object_id, 10);

  // std::string expected_string = std::to_string(kObjectPoseKey) + "10-2";
  // EXPECT_EQ(expected_string, ObjectCentricKeyFormatter(object_pose));
}
