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

/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    LevenbergMarquardtOptimizerWithLogging.h
 * @brief   A nonlinear optimizer that uses the Levenberg-Marquardt trust-region scheme
 * @author  Richard Roberts
 * @author  Frank Dellaert
 * @author  Luca Carlone
 * @date    Feb 26, 2012
 */

#pragma once

#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/linear/VectorValues.h>
#include <boost/date_time/posix_time/posix_time.hpp>

class NonlinearOptimizerMoreOptimizationTest;

namespace gtsam
{

struct LMLoggingInfo
{
  size_t iterations;
  size_t inner_iterations;
  double error;  // current state error
  double linear_costchange;
  double non_linear_costchange;
  double lambda;
};

class LevenbergMarquardtWithLoggingParams : public LevenbergMarquardtParams
{
public:
  using InnerIterationHook = std::function<void(const LMLoggingInfo&)>;

  InnerIterationHook inner_iteration_hook;

  static LevenbergMarquardtWithLoggingParams EnsureHasOrdering(LevenbergMarquardtWithLoggingParams params,
                                                               const NonlinearFactorGraph& graph)
  {
    if (!params.ordering)
      params.ordering = Ordering::Create(params.orderingType, graph);
    return params;
  }

  static LevenbergMarquardtWithLoggingParams ReplaceOrdering(LevenbergMarquardtWithLoggingParams params,
                                                             const Ordering& ordering)
  {
    params.ordering = ordering;
    return params;
  }
};

/**
 * This class performs Levenberg-Marquardt nonlinear optimization
 */
class GTSAM_EXPORT LevenbergMarquardtOptimizerWithLogging : public NonlinearOptimizer
{
protected:
  const LevenbergMarquardtWithLoggingParams params_;  ///< LM parameters
  boost::posix_time::ptime startTime_;

  void initTime();

public:
  typedef boost::shared_ptr<LevenbergMarquardtOptimizerWithLogging> shared_ptr;

  /// @name Constructors/Destructor
  /// @{

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   * @param params The optimization parameters
   */
  LevenbergMarquardtOptimizerWithLogging(
      const NonlinearFactorGraph& graph, const Values& initialValues,
      const LevenbergMarquardtWithLoggingParams& params = LevenbergMarquardtWithLoggingParams());

  /** Standard constructor, requires a nonlinear factor graph, initial
   * variable assignments, and optimization parameters.  For convenience this
   * version takes plain objects instead of shared pointers, but internally
   * copies the objects.
   * @param graph The nonlinear factor graph to optimize
   * @param initialValues The initial variable assignments
   */
  LevenbergMarquardtOptimizerWithLogging(
      const NonlinearFactorGraph& graph, const Values& initialValues, const Ordering& ordering,
      const LevenbergMarquardtWithLoggingParams& params = LevenbergMarquardtWithLoggingParams());

  /** Virtual destructor */
  ~LevenbergMarquardtOptimizerWithLogging() override
  {
  }

  /// @}

  /// @name Standard interface
  /// @{

  /// Access the current damping value
  double lambda() const;

  /// Access the current number of inner iterations
  int getInnerIterations() const;

  /// print
  void print(const std::string& str = "") const
  {
    std::cout << str << "LevenbergMarquardtOptimizerWithLogging" << std::endl;
    this->params_.print("  parameters:\n");
  }

  /// @}

  /// @name Advanced interface
  /// @{

  /**
   * Perform a single iteration, returning GaussianFactorGraph corresponding to
   * the linearized factor graph.
   */
  GaussianFactorGraph::shared_ptr iterate() override;

  /** Read-only access the parameters */
  const LevenbergMarquardtWithLoggingParams& params() const
  {
    return params_;
  }

  void writeLogFile(double currentError);

  /** linearize, can be overwritten */
  virtual GaussianFactorGraph::shared_ptr linearize() const;

  /** Build a damped system for a specific lambda -- for testing only */
  GaussianFactorGraph buildDampedSystem(const GaussianFactorGraph& linear,
                                        const VectorValues& sqrtHessianDiagonal) const;

  /** Inner loop, changes state, returns true if successful or giving up */
  bool tryLambda(const GaussianFactorGraph& linear, const VectorValues& sqrtHessianDiagonal);

  /// @}

protected:
  /** Access the parameters (base class version) */
  const NonlinearOptimizerParams& _params() const override
  {
    return params_;
  }
};

}  // namespace gtsam
