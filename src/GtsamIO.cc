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

#include "GtsamIO.h"

#include <gtsam/base/types.h>
#include <gtsam/base/FastVector.h>
#include <gtsam/inference/Key.h>

namespace gtsam
{
namespace io
{

std::unique_ptr<Factory> Factory::factoryInstance_;

GraphFileParser::GraphFileParser()
{
}

ValuesFactorGraphPair GraphFileParser::load(std::istream& istream)
{
  Factory* factory = Factory::instance();

  gtsam::Values values;
  gtsam::NonlinearFactorGraph graph;
  while (!istream.eof())
  {
    // get one line at a time
    std::string line, complete_input;
    std::getline(istream, line);
    // copy line to complete input which will be fed back into the factory to simulate a full line from the graph file
    complete_input = line;

    // use string stream as in intermediate storage for each line
    std::stringstream ss;
    ss << line;
    // get the type from the Id
    std::string type;
    ss >> type;

    // LOG(INFO) << ss.str();

    auto creator = factory->getCreator(type);
    if (!creator)
    {
      // LOG(ERROR) << "Cannot get creator for tag " << type << " as creator does not exist";
      continue;
    }

    std::stringstream complete_input_ss;
    complete_input_ss << complete_input;

    if (creator->type() == Types::FACTOR)
    {
      auto non_linear_creator = std::dynamic_pointer_cast<TypeCreator<gtsam::NonlinearFactor::shared_ptr>>(creator);
      CHECK(non_linear_creator);

      gtsam::NonlinearFactor::shared_ptr factor = non_linear_creator->construct(complete_input_ss);
      CHECK(factor);
      // factor->print();
      graph.add(factor);
    }
    else if (creator->type() == Types::VALUE)
    {
      auto value_creator = std::dynamic_pointer_cast<TypeCreator<ValueType>>(creator);
      CHECK(value_creator);

      ValueType value_type = value_creator->construct(complete_input_ss);
      // value_type.value().print();
      gtsam::Values values_type = value_type.operator gtsam::Values();
      values.insert(values_type);
    }
    else
    {
      std::runtime_error("Unknown creator types in GraphFileParser::load");
    }
  }
  return std::make_pair(values, graph);
}

ValueType::ValueType(const gtsam::Values::ConstKeyValuePair& value) : gtsam::Values({ value })
{
}

ValueType::ValueType(gtsam::Key j, const gtsam::Value& val) : gtsam::Values()
{
  gtsam::Values::insert(j, val);
}

void ValueType::print(const std::string& str, const KeyFormatter& keyFormatter) const
{
  gtsam::Values::print(str, keyFormatter);
}

bool ValueType::equals(const ValueType& other, double tol) const
{
  return gtsam::Values::equals(other, tol);
}

Key ValueType::key() const
{
  CHECK_EQ(this->size(), 1u);
  return this->begin()->key;
}

const Value& ValueType::value() const
{
  CHECK_EQ(this->size(), 1u);
  return this->begin()->value;
}

Value& ValueType::value()
{
  return this->begin()->value;
}

bool ValueType::exists() const
{
  return this->size() == 1u;
}

Factory* Factory::instance()
{
  if (factoryInstance_.get() == nullptr)
  {
    factoryInstance_.reset(new Factory);
  }
  return factoryInstance_.get();
}

void Factory::registerType(const std::string& tag, std::shared_ptr<AbstractTypeCreator> c)
{
  CreatorMap::const_iterator foundIt = creators_.find(tag);
  if (foundIt != creators_.end())
  {
    LOG(FATAL) << "FACTORY WARNING: Overwriting Vertex tag " << tag;
    assert(0);
  }
  TagLookup::const_iterator tagIt = tag_lookup_.find(c->tagName());
  if (tagIt != tag_lookup_.end())
  {
    LOG(FATAL) << "FACTORY WARNING: Registering same class for two tags " << c->tagName();
    assert(0);
  }

  creators_[tag] = c;
  tag_lookup_[c->className()] = tag;
  tag_type_lookup_[tag] = c->type();
}

Types Factory::getTypeFromTag(const std::string& tag)
{
  if (!creatorExists(tag))
  {
    // LOG(ERROR) << "Cannot get type for tag " << tag << " as creator does not exist";
    assert(0);
  }
  else
  {
    return CHECK_NOTNULL(getCreator(tag))->type();
  }
}

bool Factory::creatorExists(const std::string& tag_name) const
{
  CreatorMap::const_iterator foundIt = creators_.find(tag_name);
  return foundIt != creators_.end();
}

AbstractTypeCreator::shared_ptr Factory::getCreator(const std::string tag)
{
  if (!creatorExists(tag))
  {
    // LOG(ERROR) << "Cannot get type for tag " << tag << " as creator does not exist";
    return nullptr;
  }

  auto b = creators_.at(tag);
  CHECK(b);
  return b;
}

namespace internal
{

gtsam::Matrix fullMatrixFromUpperTriangular(std::istream& is, size_t N)
{
  gtsam::Matrix m;
  m.resize(N, N);
  for (size_t i = 0; i < N; i++)
  {
    for (size_t j = i; j < N; j++)
    {
      double val;
      is >> val;
      m(i, j) = val;
      m(j, i) = m(i, j);
    }
  }
  return m;
}

bool saveMatrixAsUpperTriangular(std::ostream& os, const gtsam::Matrix& matrix)
{
  const size_t rows = matrix.rows();
  const size_t cols = matrix.cols();

  if (rows != cols)
  {
    throw std::runtime_error("Attempting to save matrix as upper triangular but input size was not square");
  }

  for (size_t i = 0; i < rows; i++)
  {
    for (size_t j = i; j < cols; j++)
    {
      os << " " << matrix(i, j);
    }
  }
  return os.good();
}

}  // namespace internal

}  // namespace io
}  // namespace gtsam
