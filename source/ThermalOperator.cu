/* Copyright (c) 2016 - 2019, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <ThermalOperator.hh>
#include <instantiation.hh>

#include <deal.II/base/index_set.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/distributed/tria_base.h>

namespace adamantine
{

  template <int dim, int fe_degree, typename NumberType>
  __device__ void ThermalOperatorQuad<dim, fe_degree, NumberType>::
                  operator()(dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree+1, 1, NumberType> *fe_eval) const
  {
  dealii::Tensor<1, dim> unit_tensor;
  for (unsigned int i = 0; i < dim; ++i)
    unit_tensor[i] = 1.;

      fe_eval->submit_gradient(-_thermal_conductivity *
                                  (fe_eval->get_gradient()* _alpha +
                                   unit_tensor * _beta));
  }

template <int dim, int fe_degree, typename NumberType>
__device__ void LocalThermalOperator<dim, fe_degree, NumberType>::operator()(
    const unsigned int cell,
    const typename dealii::CUDAWrappers::MatrixFree<dim, NumberType>::Data *gpu_data,
    dealii::CUDAWrappers::SharedData<dim, NumberType> * shared_data,
    const NumberType *src,
    NumberType* dst) const
{
  const unsigned int pos = dealii::CUDAWrappers::local_q_point_id<dim,NumberType>(cell, gpu_data, n_dofs_1d, n_q_points);
  dealii::CUDAWrappers::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> fe_eval(
      cell, gpu_data, shared_data);

    // Store in a local vector the local values of src
    fe_eval.read_dof_values(src);
    // Evaluate the only the function gradients on the reference cell
    fe_eval.evaluate(false, true);
    // Apply the Jacobian of the transformation, multiply by the variable
    // coefficients and the quadrature points
    auto const local_thermal_conductivity = _thermal_conductivity[pos];
    auto const local_alpha = _alpha[pos];
    auto const local_beta = _beta[pos];

    ThermalOperatorQuad<dim, fe_degree, NumberType> thermal_operator_quad(local_thermal_conductivity, local_alpha, local_beta);
    fe_eval.apply_for_each_quad_point(thermal_operator_quad);
    // Sum over the quadrature points.
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
}


template <int dim, int fe_degree, typename NumberType>
ThermalOperator<dim, fe_degree, NumberType>::ThermalOperator(
    MPI_Comm const &communicator,
    std::shared_ptr<MaterialProperty<dim>> material_properties)
    : _communicator(communicator), _material_properties(material_properties),
      _inverse_mass_matrix(new dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA>())
{
_matrix_free_data.mapping_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points;
}

template <int dim, int fe_degree, typename NumberType>
template <typename QuadratureType>
void ThermalOperator<dim, fe_degree, NumberType>::setup_dofs(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<NumberType> const &affine_constraints,
    QuadratureType const &quad)
{
  _matrix_free.reinit(dof_handler, affine_constraints, quad, _matrix_free_data);
  _m = dof_handler.n_dofs();
  _dof_handler = &dof_handler;
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::reinit(
    dealii::DoFHandler<dim> const &dof_handler,
    dealii::AffineConstraints<NumberType> const &affine_constraints)
{
  // Compute the inverse of the mass matrix
  dealii::QGaussLobatto<dim> mass_matrix_quad(fe_degree + 1);
  /*dealii::CUDAWrappers::MatrixFree<dim, NumberType> mass_matrix_free;
  mass_matrix_free.reinit(dof_handler, affine_constraints, mass_matrix_quad,
                          _matrix_free_data);*/
  //TODO Do we need the diagonal for preconditioning? Try without preconditioner first.
//  mass_matrix_free.initialize_dof_vector(*_inverse_mass_matrix);
  dealii::IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  _inverse_mass_matrix->reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  dealii::LinearAlgebra::distributed::Vector<NumberType> inverse_mass_matrix_host(_inverse_mass_matrix->get_partitioner());

  const unsigned int n_dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = mass_matrix_quad.size();
  dealii::Vector<NumberType> local_diagonal(n_dofs_per_cell);
std::vector<dealii::types::global_dof_index> local_dof_indices(n_dofs_per_cell);

dealii::FEValues<dim> fe_values(dof_handler.get_fe(), mass_matrix_quad,
                          dealii::update_values | dealii::update_JxW_values);
  for (const auto& cell:dof_handler.active_cell_iterators())
if (cell->is_locally_owned())
{
local_diagonal =0.;
  fe_values.reinit(cell);

  for (unsigned int q=0; q<n_q_points; ++q)
{
for (unsigned int i=0; i<n_dofs_per_cell; ++ i)
{
   const auto shape_value = fe_values.shape_value(i,q);
   local_diagonal(i) += shape_value*shape_value*fe_values.JxW(q);
}
}

  cell->get_dof_indices(local_dof_indices);
affine_constraints.distribute_local_to_global(local_diagonal, local_dof_indices, inverse_mass_matrix_host);
}

  /*dealii::VectorizedArray<NumberType> one =
      dealii::make_vectorized_array(static_cast<NumberType>(1.));
  dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, NumberType> fe_eval(
      mass_matrix_free);
  unsigned int const n_q_points = fe_eval.n_q_points;
  for (unsigned int cell = 0; cell < mass_matrix_free.n_macro_cells(); ++cell)
  {
    fe_eval.reinit(cell);
    for (unsigned int q = 0; q < n_q_points; ++q)
      fe_eval.submit_value(one, q);
    fe_eval.integrate(true, false);
    fe_eval.distribute_local_to_global(*_inverse_mass_matrix);
  }
  _inverse_mass_imatrix->compress(dealii::VectorOperation::add);*/
  unsigned int const local_size = _inverse_mass_matrix->local_size();
  for (unsigned int k = 0; k < local_size; ++k)
  {
    if (inverse_mass_matrix_host.local_element(k) > 1e-15)
      inverse_mass_matrix_host.local_element(k) =
          1. / inverse_mass_matrix_host.local_element(k);
    else
      inverse_mass_matrix_host.local_element(k) = 0.;
  }
  _inverse_mass_matrix->import(inverse_mass_matrix_host, dealii::VectorOperation::insert);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::clear()
{
  //_matrix_free.clear();
  _inverse_mass_matrix->reinit(0);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::vmult(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> &dst,
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const &src) const
{
  dst = 0.;
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::Tvmult(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> &dst,
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const &src) const
{
  dst = 0.;
  Tvmult_add(dst, src);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::vmult_add(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> &dst,
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const &src) const
{
  LocalThermalOperator<dim, fe_degree, NumberType> thermal_operator(_thermal_conductivity.get_values(), _alpha.get_values(), _beta.get_values());
  // Execute the matrix-free matrix-vector multiplication
  _matrix_free.cell_loop(thermal_operator, src, dst);

  // Because cell_loop resolves the constraints, the constrained dofs are not
  // called they stay at zero. Thus, we need to force the value on the
  // constrained dofs by hand. The variable scaling is used so that we get the
  // right order of magnitude.
  // TODO: for now the value of scaling is set to 1
  _matrix_free.copy_constrained_values(src, dst);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::Tvmult_add(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> &dst,
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const &src) const
{
  // The system of equation is symmetric so we can use vmult_add
  vmult_add(dst, src);
}

template <int dim, int fe_degree, typename NumberType>
void ThermalOperator<dim, fe_degree, NumberType>::evaluate_material_properties(
    dealii::LA::distributed::Vector<NumberType, dealii::MemorySpace::CUDA> const &state)
{
  // Update the state of the materials
  _material_properties->update_state(*_dof_handler, state);
  dealii::LA::distributed::Vector<NumberType> state_host(state.get_partitioner());
  state_host.import(state, dealii::VectorOperation::insert);

  const unsigned int n_owned_cells =
    dynamic_cast<const dealii::parallel::TriangulationBase<dim> *>(
      &_dof_handler->get_triangulation())
      ->n_locally_owned_active_cells();

  const unsigned int n_q_points = dealii::Utilities::pow(fe_degree+1, dim);

  std::vector<NumberType> alpha_host(n_owned_cells*n_q_points);
  _alpha.reinit(n_owned_cells*n_q_points);
  std::vector<NumberType> beta_host(n_owned_cells*n_q_points);
  _beta.reinit(n_owned_cells*n_q_points);
  std::vector<NumberType> thermal_conductivity_host(n_owned_cells*n_q_points);
  _thermal_conductivity.reinit(n_owned_cells*n_q_points);

  unsigned int cell_no=0;
  for (const auto& cell: _dof_handler->active_cell_iterators())
    if (cell->is_locally_owned())
    {
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Cast to Triangulation<dim>::cell_iterator to access the material_id
        typename dealii::Triangulation<dim>::active_cell_iterator cell_tria(
            cell);

        thermal_conductivity_host[cell_no*n_q_points+q] = _material_properties->get(
            cell_tria, Property::thermal_conductivity, state_host);

        double liquid_ratio = _material_properties->get_state_ratio(
            cell_tria, MaterialState::liquid);

        // If there is less than 1e-13% of liquid, assume that there is no
        // liquid.
        if (liquid_ratio < 1e-15)
        {
          alpha_host[cell_no*n_q_points+q] =
              1. /
              (_material_properties->get(cell_tria, Property::density, state_host) *
               _material_properties->get(cell_tria, Property::specific_heat,
                                         state_host));
        }
        else
        {
          // If there is less than 1e-13% of solid, assume that there is no
          // solid. Otherwise, we have a mix of liquid and solid (mushy zone).
          if (liquid_ratio > (1 - 1e-15))
          {
            alpha_host[cell_no*n_q_points+q] =
                1. / (_material_properties->get(cell_tria, Property::density,
                                                state_host) *
                      _material_properties->get(
                          cell_tria, Property::specific_heat, state_host));
            beta_host[cell_no*n_q_points+q] =
                _material_properties->get_liquid_beta(cell_tria);
          }
          else
          {
            alpha_host[cell_no*n_q_points+q] =
                _material_properties->get_mushy_alpha(cell_tria);
            beta_host[cell_no*n_q_points+q] = _material_properties->get_mushy_beta(cell_tria);
          }
        }
      }
++cell_no;
}
dealii::Utilities::CUDA::copy_to_dev(thermal_conductivity_host, _thermal_conductivity.get_values());
dealii::Utilities::CUDA::copy_to_dev(alpha_host, _alpha.get_values());
dealii::Utilities::CUDA::copy_to_dev(beta_host, _beta.get_values());
}
} // namespace adamantine

INSTANTIATE_DIM_FEDEGREE_NUM(TUPLE(ThermalOperator))

// Instantiate the function template.
namespace adamantine
{
template void ThermalOperator<2, 1, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 2, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 3, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 4, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 5, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 6, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 7, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 8, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 9, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 10, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<2, 1, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 2, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 3, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 4, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 5, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 6, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 7, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 8, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 9, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 10, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<2, 1, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 2, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 3, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 4, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 5, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 6, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 7, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 8, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 9, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<2, 10, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<2> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<2, 1, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 2, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 3, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 4, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 5, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 6, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 7, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 8, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 9, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<2, 10, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<2> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<3, 1, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 2, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 3, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 4, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 5, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 6, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 7, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 8, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 9, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 10, float>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<float> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<3, 1, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 2, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 3, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 4, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 5, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 6, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 7, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 8, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 9, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 10, float>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<float> const &,
        dealii::QGaussLobatto<1> const &);

template void ThermalOperator<3, 1, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 2, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 3, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 4, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 5, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 6, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 7, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 8, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 9, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);
template void ThermalOperator<3, 10, double>::setup_dofs<dealii::QGauss<1>>(
    dealii::DoFHandler<3> const &, dealii::AffineConstraints<double> const &,
    dealii::QGauss<1> const &);

template void
    ThermalOperator<3, 1, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 2, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 3, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 4, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 5, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 6, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 7, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 8, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 9, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
template void
    ThermalOperator<3, 10, double>::setup_dofs<dealii::QGaussLobatto<1>>(
        dealii::DoFHandler<3> const &,
        dealii::AffineConstraints<double> const &,
        dealii::QGaussLobatto<1> const &);
} // namespace adamantine
