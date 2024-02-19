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
 * @file    LevenbergMarquardtOptimizerWithLogging.cpp
 * @brief   A nonlinear optimizer that uses the Levenberg-Marquardt trust-region scheme
 * @author  Richard Roberts
 * @author  Frank Dellaert
 * @author  Luca Carlone
 * @date    Feb 26, 2012
 */

#include "Optimizer.h"

#include <gtsam/nonlinear/internal/LevenbergMarquardtState.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/linearExceptions.h>
#include <gtsam/inference/Ordering.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/timing.h>

#include <boost/format.hpp>
#include <boost/optional.hpp>
#include <boost/range/adaptor/map.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

using namespace std;

namespace gtsam
{

using boost::adaptors::map_values;
typedef internal::LevenbergMarquardtState State;

/* ************************************************************************* */
LevenbergMarquardtOptimizerWithLogging::LevenbergMarquardtOptimizerWithLogging(
    const NonlinearFactorGraph& graph, const Values& initialValues, const LevenbergMarquardtWithLoggingParams& params)
  : NonlinearOptimizer(graph, std::unique_ptr<State>(new State(initialValues, graph.error(initialValues),
                                                               params.lambdaInitial, params.lambdaFactor)))
  , params_(LevenbergMarquardtWithLoggingParams::EnsureHasOrdering(params, graph))
{
}

LevenbergMarquardtOptimizerWithLogging::LevenbergMarquardtOptimizerWithLogging(
    const NonlinearFactorGraph& graph, const Values& initialValues, const Ordering& ordering,
    const LevenbergMarquardtWithLoggingParams& params)
  : NonlinearOptimizer(graph, std::unique_ptr<State>(new State(initialValues, graph.error(initialValues),
                                                               params.lambdaInitial, params.lambdaFactor)))
  , params_(LevenbergMarquardtWithLoggingParams::ReplaceOrdering(params, ordering))
{
}

/* ************************************************************************* */
void LevenbergMarquardtOptimizerWithLogging::initTime()
{
  startTime_ = boost::posix_time::microsec_clock::universal_time();
}

/* ************************************************************************* */
double LevenbergMarquardtOptimizerWithLogging::lambda() const
{
  auto currentState = static_cast<const State*>(state_.get());
  return currentState->lambda;
}

/* ************************************************************************* */
int LevenbergMarquardtOptimizerWithLogging::getInnerIterations() const
{
  auto currentState = static_cast<const State*>(state_.get());
  return currentState->totalNumberInnerIterations;
}

/* ************************************************************************* */
GaussianFactorGraph::shared_ptr LevenbergMarquardtOptimizerWithLogging::linearize() const
{
  return graph_.linearize(state_->values);
}

/* ************************************************************************* */
GaussianFactorGraph LevenbergMarquardtOptimizerWithLogging::buildDampedSystem(
    const GaussianFactorGraph& linear, const VectorValues& sqrtHessianDiagonal) const
{
  gttic(damp);
  auto currentState = static_cast<const State*>(state_.get());

  if (params_.verbosityLM >= LevenbergMarquardtParams::DAMPED)
    std::cout << "building damped system with lambda " << currentState->lambda << std::endl;

  if (params_.diagonalDamping)
    return currentState->buildDampedSystem(linear, sqrtHessianDiagonal);
  else
    return currentState->buildDampedSystem(linear);
}

/* ************************************************************************* */
// Log current error/lambda to file
inline void LevenbergMarquardtOptimizerWithLogging::writeLogFile(double currentError)
{
  auto currentState = static_cast<const State*>(state_.get());

  if (!params_.logFile.empty())
  {
    ofstream os(params_.logFile.c_str(), ios::app);
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    os << /*inner iterations*/ currentState->totalNumberInnerIterations << ","
       << 1e-6 * (currentTime - startTime_).total_microseconds() << "," << /*current error*/ currentError << ","
       << currentState->lambda << "," << /*outer iterations*/ currentState->iterations << endl;
  }
}

/* ************************************************************************* */
bool LevenbergMarquardtOptimizerWithLogging::tryLambda(const GaussianFactorGraph& linear,
                                                       const VectorValues& sqrtHessianDiagonal)
{
  auto currentState = static_cast<const State*>(state_.get());
  bool verbose = (params_.verbosityLM >= LevenbergMarquardtParams::TRYLAMBDA);

#ifdef GTSAM_USING_NEW_BOOST_TIMERS
  boost::timer::cpu_timer lamda_iteration_timer;
  lamda_iteration_timer.start();
#else
  boost::timer lamda_iteration_timer;
  lamda_iteration_timer.restart();
#endif

  if (verbose)
    cout << "trying lambda = " << currentState->lambda << endl;

  // Build damped system for this lambda (adds prior factors that make it like gradient descent)
  auto dampedSystem = buildDampedSystem(linear, sqrtHessianDiagonal);

  // Try solving
  double modelFidelity = 0.0;
  bool step_is_successful = false;
  bool stopSearchingLambda = false;
  double newError = numeric_limits<double>::infinity(), costChange;
  Values newValues;
  VectorValues delta;

  bool systemSolvedSuccessfully;
  try
  {
    // ============ Solve is where most computation happens !! =================
    delta = solve(dampedSystem, params_);
    systemSolvedSuccessfully = true;
  }
  catch (const IndeterminantLinearSystemException&)
  {
    systemSolvedSuccessfully = false;
  }

  if (systemSolvedSuccessfully)
  {
    if (verbose)
      cout << "linear delta norm = " << delta.norm() << endl;
    if (params_.verbosityLM >= LevenbergMarquardtParams::TRYDELTA)
      delta.print("delta");

    // Compute the old linearized error as it is not the same
    // as the nonlinear error when robust noise models are used.
    double oldLinearizedError = linear.error(VectorValues::Zero(delta));
    double newlinearizedError = linear.error(delta);

    // cost change in the linearized system (old - new)
    double linearizedCostChange = oldLinearizedError - newlinearizedError;
    if (verbose)
      cout << "newlinearizedError = " << newlinearizedError << "  linearizedCostChange = " << linearizedCostChange
           << endl;

    if (linearizedCostChange >= 0)
    {  // step is valid
      // update values
      gttic(retract);
      // ============ This is where the solution is updated ====================
      newValues = currentState->values.retract(delta);
      // =======================================================================
      gttoc(retract);

      // compute new error
      gttic(compute_error);
      if (verbose)
        cout << "calculating error:" << endl;
      newError = graph_.error(newValues);
      gttoc(compute_error);

      if (verbose)
        cout << "old error (" << currentState->error << ") new (tentative) error (" << newError << ")" << endl;

      // cost change in the original, nonlinear system (old - new)
      costChange = currentState->error - newError;

      if (linearizedCostChange > std::numeric_limits<double>::epsilon() * oldLinearizedError)
      {
        // the (linear) error has to decrease to satisfy this condition
        // fidelity of linearized model VS original system between
        modelFidelity = costChange / linearizedCostChange;
        // if we decrease the error in the nonlinear system and modelFidelity is above threshold
        step_is_successful = modelFidelity > params_.minModelFidelity;
        if (verbose)
          cout << "modelFidelity: " << modelFidelity << endl;
      }  // else we consider the step non successful and we either increase lambda or stop if error
         // change is small

      double minAbsoluteTolerance = params_.relativeErrorTol * currentState->error;
      // if the change is small we terminate
      if (std::abs(costChange) < minAbsoluteTolerance)
      {
        if (verbose)
          cout << "abs(costChange)=" << std::abs(costChange) << "  minAbsoluteTolerance=" << minAbsoluteTolerance
               << " (relativeErrorTol=" << params_.relativeErrorTol << ")" << endl;
        stopSearchingLambda = true;
      }
    }

    if (params_.inner_iteration_hook)
    {
      LMLoggingInfo logging_info;
      logging_info.iterations = currentState->iterations;
      logging_info.inner_iterations = currentState->totalNumberInnerIterations;
      logging_info.error = newError;
      logging_info.linear_costchange = linearizedCostChange;
      logging_info.non_linear_costchange = costChange;
      logging_info.lambda = currentState->lambda;
      params_.inner_iteration_hook(logging_info);
    }

  }  // if (systemSolvedSuccessfully)

  if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY)
  {
// do timing
#ifdef GTSAM_USING_NEW_BOOST_TIMERS
    double iterationTime = 1e-9 * lamda_iteration_timer.elapsed().wall;
#else
    double iterationTime = lamda_iteration_timer.elapsed();
#endif
    if (currentState->iterations == 0)
      cout << "iter      cost      cost_change    lambda  success iter_time" << endl;

    cout << boost::format("% 4d % 8e   % 3.2e   % 3.2e  % 4d   % 3.2e") % currentState->iterations % newError %
                costChange % currentState->lambda % systemSolvedSuccessfully % iterationTime
         << endl;
  }

  if (step_is_successful)
  {
    // we have successfully decreased the cost and we have good modelFidelity
    // NOTE(frank): As we return immediately after this, we move the newValues
    // TODO(frank): make Values actually support move. Does not seem to happen now.
    state_ = currentState->decreaseLambda(params_, modelFidelity, std::move(newValues), newError);
    return true;
  }
  else if (!stopSearchingLambda)
  {  // we failed to solved the system or had no decrease in cost
    if (verbose)
      cout << "increasing lambda" << endl;
    State* modifiedState = static_cast<State*>(state_.get());
    modifiedState->increaseLambda(params_);  // TODO(frank): make this functional with Values move

    // check if lambda is too big
    if (modifiedState->lambda >= params_.lambdaUpperBound)
    {
      if (params_.verbosity >= NonlinearOptimizerParams::TERMINATION ||
          params_.verbosityLM == LevenbergMarquardtParams::SUMMARY)
        cout << "Warning:  Levenberg-Marquardt giving up because "
                "cannot decrease error with maximum lambda"
             << endl;
      return true;
    }
    else
    {
      return false;  // only case where we will keep trying
    }
  }
  else
  {  // the change in the cost is very small and it is not worth trying bigger lambdas
    if (verbose)
      cout << "Levenberg-Marquardt: stopping as relative cost reduction is small" << endl;
    return true;
  }
}

/* ************************************************************************* */
GaussianFactorGraph::shared_ptr LevenbergMarquardtOptimizerWithLogging::iterate()
{
  auto currentState = static_cast<const State*>(state_.get());

  gttic(LM_iterate);

  // Linearize graph
  if (params_.verbosityLM >= LevenbergMarquardtParams::DAMPED)
    cout << "linearizing = " << endl;
  GaussianFactorGraph::shared_ptr linear = linearize();

  if (currentState->totalNumberInnerIterations == 0)
  {  // write initial error
    writeLogFile(currentState->error);

    if (params_.verbosityLM == LevenbergMarquardtParams::SUMMARY)
    {
      cout << "Initial error: " << currentState->error << ", values: " << currentState->values.size() << std::endl;
    }
  }

  // Only calculate diagonal of Hessian (expensive) once per outer iteration, if we need it
  VectorValues sqrtHessianDiagonal;
  if (params_.diagonalDamping)
  {
    sqrtHessianDiagonal = linear->hessianDiagonal();
    for (Vector& v : sqrtHessianDiagonal | map_values)
    {
      v = v.cwiseMax(params_.minDiagonal).cwiseMin(params_.maxDiagonal).cwiseSqrt();
    }
  }

  // Keep increasing lambda until we make make progress
  while (!tryLambda(*linear, sqrtHessianDiagonal))
  {
    auto newState = static_cast<const State*>(state_.get());
    writeLogFile(newState->error);
  }

  return linear;
}

} /* namespace gtsam */
