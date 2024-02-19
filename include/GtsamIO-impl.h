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

#pragma once

#include "GtsamIO.h"

#include <iostream>

namespace gtsam
{
namespace io
{

// template <typename T>
// std::string tag_name() { throw std::runtime_error("Tag name function has not been specalised for type " +
// std::string(typeid(T).name()) + ". Did you register it?"); }

template <typename T>
gtsam::NonlinearFactor::shared_ptr NonLinearFactorTypeCreator<T>::construct(std::istream& is) const
{
  std::string loaded_tag_name;
  is >> loaded_tag_name;

  std::string expected_tag = this->tagName();
  if (loaded_tag_name != expected_tag)
  {
    throw std::runtime_error("Error when constructing value type " + this->className() + " and tag " + this->tagName());
  }

  size_t num_keys = getFactorSize<T>();
  gtsam::KeyVector keys;
  for (size_t i = 0; i < num_keys; i++)
  {
    // KEY or index?
    gtsam::Key key;
    is >> key;
    keys.push_back(key);
  }

  gtsam::NonlinearFactor::shared_ptr noiseModelFactor = nullptr;
  // this should just fill in the measurement?
  factor<T>::read(is, keys, noiseModelFactor);
  if (!noiseModelFactor)
  {
    throw std::runtime_error("Noise model factor is nullptr - did you forget to allocate memory?");
  }
  return noiseModelFactor;
}

template <typename T>
bool NonLinearFactorTypeCreator<T>::save(std::ostream& os, const gtsam::NonlinearFactor::shared_ptr& t) const
{
  // for a factor we want to save the name, keys and then let the user defined function write the measurements
  os << this->tagName() << " ";
  const KeyVector& keys = t->keys();
  for (const gtsam::Key& key : keys)
  {
    os << this->indexing(key) << " ";
  }

  const boost::shared_ptr<T> cast_factor = boost::dynamic_pointer_cast<T>(t);
  CHECK(cast_factor);

  factor<T>::write(os, cast_factor);
  os << "\n";
  return os.good();
}

template <typename T>
Types NonLinearFactorTypeCreator<T>::type() const
{
  return Types::FACTOR;
}

// template<typename VALUE>
// ValueType::ValueType(gtsam::Key j, const VALUE& val) : gtsam::Values() {
//   gtsam::Values::insert(j, val);
// }

template <typename T>
const T ValueType::value() const
{
  Key k = this->key();
  return gtsam::Values::at<T>(k);
}

template <typename T>
ValueType ValueTypeCreator<T>::construct(std::istream& is) const
{
  std::string loaded_tag_name;
  is >> loaded_tag_name;

  std::string expected_tag = this->tagName();
  if (loaded_tag_name != expected_tag)
  {
    throw std::runtime_error("Error when constructing value type " + this->className() + " and tag " + this->tagName());
  }

  size_t id;  // this should be key
  is >> id;

  T v;
  value<T>::read(is, v);
  return ValueType(id, gtsam::genericValue(v));
}

template <typename T>
bool ValueTypeCreator<T>::save(std::ostream& os, const ValueType& t) const
{
  // save tag key and then user defined value
  os << this->tagName() << " ";
  os << this->indexing(t.key()) << " ";
  value<T>::write(os, t.value<T>());
  os << std::endl;
  return os.good();
}

template <typename T>
Types ValueTypeCreator<T>::type() const
{
  return Types::VALUE;
}

template <typename T>
std::shared_ptr<AbstractTypeCreator> Factory::getCreator()
{
  return this->getCreator(tag_name<T>());
}

}  // namespace io
}  // namespace gtsam
