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

#include "UtilsGtsam.h"  //for enable if's on gtsam types

#include <glog/logging.h>
#include <Eigen/Core>
#include <type_traits>

#include <gtsam/geometry/Unit3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

/// The following templating is for reading and writing of values and factors similar to g2o
namespace gtsam
{

template <typename NonLinearFactor>
constexpr size_t getFactorSize()
{
  return static_cast<size_t>(NonLinearFactor::N);
}

namespace io
{

template <typename T>
std::string tag_name();

// we need this in a struct becuase the read function takes no templated parameters
template <typename T>
struct factor
{
  // factor will be none
  static bool read(std::istream& is, const gtsam::KeyVector& keys, gtsam::NonlinearFactor::shared_ptr& factor);
  static bool write(std::ostream& os, const boost::shared_ptr<T> factor);
};

// dont need this in a struct but its nice for consistency
template <typename T>
struct value
{
  static bool read(std::istream& is, T& t);
  static bool write(std::ostream& os, const T& t);
};

using ValuesFactorGraphPair = std::pair<gtsam::Values, gtsam::NonlinearFactorGraph>;

class GraphFileParser
{
public:
  // todo: options
  GraphFileParser();

  ValuesFactorGraphPair load(std::istream& istream);
  bool write(std::ostream& os, const gtsam::Values& values, gtsam::NonlinearFactorGraph& graph);

protected:
};

enum class Types
{
  FACTOR,
  VALUE
};

using IndexingFuncion = std::function<std::uint64_t(gtsam::Key)>;
// using KeyConversionFunction = std::function<

class AbstractTypeCreator
{
public:
  using shared_ptr = std::shared_ptr<AbstractTypeCreator>;
  virtual ~AbstractTypeCreator() = default;

  virtual const std::string& className() const = 0;
  virtual const std::string& tagName() const = 0;
  virtual Types type() const = 0;
};

template <typename T>
class TypeCreator : public AbstractTypeCreator
{
public:
  using Type = T;
  using This = TypeCreator<T>;
  using shared_ptr = std::shared_ptr<This>;

  TypeCreator(const std::string& class_name, const std::string& tag) : class_name_(class_name), tag_(tag)
  {
    // indexing = [](gtsam::Key key) { return Symbol(key).index(); }; //I think we want the actual key and not the index
    indexing = [](gtsam::Key key) { return key; };  // I think we want the actual key and not the index
  }

  const std::string& className() const override
  {
    return class_name_;
  }
  const std::string& tagName() const override
  {
    return tag_;
  }

  void setIndexingFunction(const IndexingFuncion func)
  {
    // overwrites...?
    indexing = func;
  }

  virtual Type construct(std::istream& is) const = 0;
  virtual bool save(std::ostream& os, const Type& t) const = 0;

protected:
  IndexingFuncion indexing;
  const std::string class_name_;
  const std::string tag_;
};

// T should be the nonlinearfactorN
template <typename T>
class NonLinearFactorTypeCreator : public TypeCreator<gtsam::NonlinearFactor::shared_ptr>
{
public:
  static constexpr size_t NumValues = getFactorSize<T>();
  using Base = TypeCreator<gtsam::NonlinearFactor::shared_ptr>;
  NonLinearFactorTypeCreator(const std::string& class_name, const std::string& tag) : Base(class_name, tag)
  {
  }

  gtsam::NonlinearFactor::shared_ptr construct(std::istream& is) const override;
  bool save(std::ostream& os, const gtsam::NonlinearFactor::shared_ptr& t) const override;
  Types type() const;
};

// needs tests
class ValueType : private gtsam::Values
{
public:
  using KeyValuePair = gtsam::Values::KeyValuePair;
  using ConstKeyValuePair = gtsam::Values::ConstKeyValuePair;

  ValueType() = default;

  // /** Copy constructor duplicates all keys and values */
  // ValueType(const ValueType& other);

  // /** Move constructor */
  // ValueType(ValueType&& other);

  /** Constructor from initializer list. Example usage:
   * \code
   * Values v = {{k1, genericValue(pose1)}, {k2, genericValue(point2)}};
   * \endcode
   */
  ValueType(const gtsam::Values::ConstKeyValuePair&);

  // template<typename VALUE>
  // ValueType(gtsam::Key j, const VALUE& val);

  ValueType(gtsam::Key j, const gtsam::Value& val);

  // cast to values
  operator gtsam::Values() const
  {
    return static_cast<gtsam::Values>(*this);
  }

  /** print method for testing and debugging */
  void print(const std::string& str = "", const KeyFormatter& keyFormatter = DefaultKeyFormatter) const;

  /** Test whether the sets of keys and values are identical */
  bool equals(const ValueType& other, double tol = 1e-9) const;

  // template <typename ValueType>
  // const ValueType at(Key j) const;

  template <typename T>
  const T value() const;

  // template <typename T>
  // // T& value();

  const Value& value() const;
  Value& value();

  bool exists() const;

  Key key() const;

  // /** Retrieve a variable by key \c j.  This version returns a reference
  //  * to the base Value class, and needs to be casted before use.
  //  * @param j Retrieve the value associated with this key
  //  * @return A const reference to the stored value
  //  */
  // const Value& at(Key j) const;

  // /** Check if a value exists with key \c j.  See exists<>(Key j)
  //  * and exists(const TypedKey& j) for versions that return the value if it
  //  * exists. */
  // bool exists(Key j) const;
};

// T shoudl be the value
template <typename T>
class ValueTypeCreator : public TypeCreator<ValueType>
{
public:
  using Base = TypeCreator<ValueType>;
  ValueTypeCreator(const std::string& class_name, const std::string& tag) : Base(class_name, tag)
  {
  }

  ValueType construct(std::istream& is) const override;
  bool save(std::ostream& os, const ValueType& t) const override;
  Types type() const;
};

class Factory
{
public:
  static Factory* instance();

  Factory(Factory const&) = delete;
  Factory& operator=(Factory const&) = delete;

  void registerType(const std::string& tag, std::shared_ptr<AbstractTypeCreator> c);

  Types getTypeFromTag(const std::string& tag);

  template <typename T>
  AbstractTypeCreator::shared_ptr getCreator();

  AbstractTypeCreator::shared_ptr getCreator(const std::string tag);

  /**
   * @brief Gets the either the TypeCreator<gtsam::NonlinearFactor::shared_ptr> or TypeCreator<ValueType>
   * depending on if T is a factor or a value type. The TypeCreator knows about the type T
   *
   * @tparam T
   * @return auto
   */
  template <typename T>
  auto getCreatorTyped()
  {
    // not the most extentable approach but whatever
    if constexpr (is_gtsam_factor_v<T>)
    {
      return std::dynamic_pointer_cast<TypeCreator<gtsam::NonlinearFactor::shared_ptr>>(this->getCreator<T>());
      // return this->getCreatorAuto<gtsam::NonlinearFactor::shared_ptr>(tag_name<T>());
    }
    else
    {
      return std::dynamic_pointer_cast<TypeCreator<ValueType>>(this->getCreator<T>());
      // return this->getCreatorAuto<KeyTypeValuePair<T>>(tag_name<T>());
    }
  }

protected:
  using CreatorMap = std::map<std::string, std::shared_ptr<AbstractTypeCreator>>;  // tag to creator
  using TagLookup = std::map<std::string, std::string>;                            // class name to tag
  using TagTypeLookup = std::map<std::string, Types>;
  Factory() = default;

  CreatorMap creators_;
  TagLookup tag_lookup_;
  TagTypeLookup tag_type_lookup_;

private:
  bool creatorExists(const std::string& tag_name) const;

  static std::unique_ptr<Factory> factoryInstance_;
};

template <typename T>
class RegisterTypeProxy
{
public:
  // otherwise assume its a value since no base class
  RegisterTypeProxy(const std::string& class_name, const std::string& tag)
  {
    std::shared_ptr<AbstractTypeCreator> creator = nullptr;
    if constexpr (is_gtsam_factor_v<T>)
    {
      LOG(ERROR) << __FUNCTION__ << ": Registering " << tag << " of factor type " << class_name;
      creator = std::make_shared<NonLinearFactorTypeCreator<T>>(class_name, tag);
    }
    else
    {
      LOG(ERROR) << __FUNCTION__ << ": Registering " << tag << " of value type " << class_name;
      creator = std::make_shared<ValueTypeCreator<T>>(class_name, tag);
    }
    CHECK(creator);
    Factory::instance()->registerType(tag, creator);
  }
};

namespace internal
{

template <int n>
struct factorial
{
  enum
  {
    val = n * factorial<n - 1>::val
  };
};

template <>
struct factorial<1>
{
  enum
  {
    val = 1
  };
};

gtsam::Matrix fullMatrixFromUpperTriangular(std::istream& is, size_t N);

bool saveMatrixAsUpperTriangular(std::ostream& os, const gtsam::Matrix& matrix);

}  // namespace internal

}  // namespace io
}  // namespace gtsam

#include "GtsamIO-impl.h"
#include <boost/algorithm/string/replace.hpp>

#define CONCATENATE_DIRECT(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_DIRECT(s1, s2)

#define GTSAM_REGISTER_TYPE(tagname, classname)                                                                        \
  GTSAM_REGISTER_TYPE_IMPL(tagname, classname, CONCATENATE(g_type_proxy, __LINE__))

// These macros are used to automate registering types and forcing linkage
// actually want to use the class or tag name so that user cannot register type or name twice but compileation issues
// atm...
#define GTSAM_REGISTER_TYPE_IMPL(tagname, classname, var)                                                              \
  static gtsam::io::RegisterTypeProxy<classname> var(#classname, #tagname);                                            \
  template <>                                                                                                          \
  std::string gtsam::io::tag_name<classname>()                                                                         \
  {                                                                                                                    \
    return #tagname;                                                                                                   \
  }
