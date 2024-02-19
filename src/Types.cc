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

#include "Types.h"

#include <vector>
#include <cmath>
#include <cstdio>
#include <limits>

namespace dsc
{

bool fp_equal(double a, double b, double tol, bool check_relative_also)
{
  using std::abs;
  using std::isinf;
  using std::isnan;

  double DOUBLE_MIN_NORMAL = std::numeric_limits<double>::min() + 1.0;
  double larger = (abs(b) > abs(a)) ? abs(b) : abs(a);

  // handle NaNs
  if (isnan(a) || isnan(b))
  {
    return isnan(a) && isnan(b);
  }
  // handle inf
  else if (isinf(a) || isinf(b))
  {
    return isinf(a) && isinf(b);
  }
  // If the two values are zero or both are extremely close to it
  // relative error is less meaningful here
  else if (a == 0 || b == 0 || (abs(a) + abs(b)) < DOUBLE_MIN_NORMAL)
  {
    return abs(a - b) <= tol * DOUBLE_MIN_NORMAL;
  }
  // Check if the numbers are really close.
  // Needed when comparing numbers near zero or tol is in vicinity.
  else if (abs(a - b) <= tol)
  {
    return true;
  }
  // Check for relative error
  else if (abs(a - b) <= tol * std::min(larger, std::numeric_limits<double>::max()) && check_relative_also)
  {
    return true;
  }

  return false;
}

template <>
bool equals<KeypointCV>(const KeypointCV& a, const KeypointCV& b, double tol)
{
  return fp_equal(a.pt.x, b.pt.x, tol) && fp_equal(a.pt.y, b.pt.y, tol);
}

template <>
bool equals<cv::Point2d>(const cv::Point2d& a, const cv::Point2d& b, double tol)
{
  return fp_equal(a.x, b.x, tol) && fp_equal(a.y, b.y, tol);
}

}  // namespace dsc
