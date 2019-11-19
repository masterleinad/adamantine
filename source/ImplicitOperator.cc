/* Copyright (c) 2016 - 2019, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ImplicitOperator.hh>
#include <instantiation.hh>
#include <utils.hh>

namespace adamantine
{
template <typename NumberType>
ImplicitOperator<NumberType>::ImplicitOperator(
    std::shared_ptr<Operator<NumberType>> explicit_operator, bool jfnk)
    : _jfnk(jfnk), _explicit_operator(explicit_operator)
{
}

template <typename NumberType>
void ImplicitOperator<NumberType>::vmult(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> &dst,
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const &src) const
{
  if (_jfnk == true)
  {
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> tmp_dst(dst.get_partitioner());
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> tmp_src(src.get_partitioner());
    tmp_src.import(src, dealii::VectorOperation::insert);
    const double delta = 1.e-8;
    tmp_src *= (1. + delta);
    _explicit_operator->vmult(dst, tmp_src);
    _explicit_operator->vmult(tmp_dst, src);
    dst -= tmp_dst;
    dst /= delta;
  }
  else
    _explicit_operator->jacobian_vmult(dst, src);

  dst.scale(*_inverse_mass_matrix);
  dst *= -_tau;
  dst += src;
}

template <typename NumberType>
void ImplicitOperator<NumberType>::Tvmult(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> & /*dst*/,
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const & /*src*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <typename NumberType>
void ImplicitOperator<NumberType>::vmult_add(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> & /*dst*/,
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const & /*src*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}

template <typename NumberType>
void ImplicitOperator<NumberType>::Tvmult_add(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> & /*dst*/,
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const & /*src*/) const
{
  ASSERT_THROW_NOT_IMPLEMENTED();
}
} // namespace adamantine

INSTANTIATE_NUM(ImplicitOperator)
