/* Copyright (c) 2021, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#define BOOST_TEST_MODULE DataAssimilator

#include <DataAssimilator.hh>
#include <Geometry.hh>

#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "main.cc"

namespace adamantine
{
class DataAssimilatorTester
{
public:
  void test_constructor()
  {
    boost::property_tree::ptree database;

    // First checking the dealii default values
    DataAssimilator da0(database);

    double tol = 1.0e-12;
    BOOST_CHECK_SMALL(da0._solver_control.tolerance() - 1.0e-10, tol);
    BOOST_CHECK(da0._solver_control.max_steps() == 100);
    BOOST_CHECK(da0._additional_data.max_n_tmp_vectors == 30);

    // Now explicitly setting them
    database.put("solver.convergence_tolerance", 1.0e-6);
    database.put("solver.max_iterations", 25);
    database.put("solver.max_number_of_temp_vectors", 4);
    DataAssimilator da1(database);
    BOOST_CHECK_SMALL(da1._solver_control.tolerance() - 1.0e-6, tol);
    BOOST_CHECK(da1._solver_control.max_steps() == 25);
    BOOST_CHECK(da1._additional_data.max_n_tmp_vectors == 4);
  };

  void test_calc_kalman_gain()
  {
    // Create the DoF mapping
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    unsigned int sim_size = 5;
    unsigned int expt_size = 2;

    dealii::Vector<double> expt_vec(2);
    expt_vec(0) = 2.5;
    expt_vec(1) = 9.5;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(2);
    indices_and_offsets.second.resize(3); // Offset vector is one longer
    indices_and_offsets.first[0] = 1;
    indices_and_offsets.first[1] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da._num_ensemble_members = 3;
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);

    // Create the simulation data
    std::vector<dealii::LA::distributed::Vector<double>> data(3);
    data[0].reinit(5);
    data[0](0) = 1.0;
    data[0](1) = 3.0;
    data[0](2) = 6.0;
    data[0](3) = 9.0;
    data[0](4) = 11.0;
    data[1].reinit(5);
    data[1](0) = 1.5;
    data[1](1) = 3.2;
    data[1](2) = 6.3;
    data[1](3) = 9.7;
    data[1](4) = 11.9;
    data[2].reinit(5);
    data[2](0) = 1.1;
    data[2](1) = 3.1;
    data[2](2) = 6.1;
    data[2](3) = 9.1;
    data[2](4) = 11.1;

    // Build the sparse experimental covariance matrix
    dealii::SparsityPattern pattern(expt_size, expt_size, 1);
    pattern.add(0, 0);
    pattern.add(1, 1);
    pattern.compress();

    dealii::SparseMatrix<double> R(pattern);
    R.add(0, 0, 0.002);
    R.add(1, 1, 0.001);

    // Create the (perturbed) innovation
    std::vector<dealii::Vector<double>> perturbed_innovation(3);
    for (unsigned int sample = 0; sample < perturbed_innovation.size();
         ++sample)
    {
      perturbed_innovation[sample].reinit(expt_size);
      dealii::Vector<double> temp = da.calc_Hx(data[sample]);
      for (unsigned int i = 0; i < expt_size; ++i)
      {
        perturbed_innovation[sample][i] = expt_vec[i] - temp[i];
      }
    }

    perturbed_innovation[0][0] = perturbed_innovation[0][0] + 0.0008;
    perturbed_innovation[0][1] = perturbed_innovation[0][1] - 0.0005;
    perturbed_innovation[1][0] = perturbed_innovation[1][0] - 0.001;
    perturbed_innovation[1][1] = perturbed_innovation[1][1] + 0.0002;
    perturbed_innovation[2][0] = perturbed_innovation[2][0] + 0.0002;
    perturbed_innovation[2][1] = perturbed_innovation[2][1] - 0.0009;

    // Apply the Kalman gain
    std::vector<dealii::LA::distributed::Vector<double>> forecast_shift =
        da.apply_kalman_gain(data, R, perturbed_innovation);

    double tol = 1.0e-4;

    // Reference solution calculated using Python
    BOOST_CHECK_CLOSE(forecast_shift[0][0], 0.21352564, tol);
    BOOST_CHECK_CLOSE(forecast_shift[0][1], -0.14600986, tol);
    BOOST_CHECK_CLOSE(forecast_shift[0][2], -0.02616469, tol);
    BOOST_CHECK_CLOSE(forecast_shift[0][3], 0.45321598, tol);
    BOOST_CHECK_CLOSE(forecast_shift[0][4], 0.69290631, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][0], -0.27786325, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][1], -0.32946285, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][2], -0.31226298, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][3], -0.24346351, tol);
    BOOST_CHECK_CLOSE(forecast_shift[1][4], -0.20906377, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][0], 0.12767094, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][1], -0.20319395, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][2], -0.09290565, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][3], 0.34824753, tol);
    BOOST_CHECK_CLOSE(forecast_shift[2][4], 0.56882413, tol);
  };

  void test_update_dof_mapping()
  {
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    unsigned int sim_size = 4;
    unsigned int expt_size = 3;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(3);
    indices_and_offsets.second.resize(4); // offset vector is one longer
    indices_and_offsets.first[0] = 0;
    indices_and_offsets.first[1] = 1;
    indices_and_offsets.first[2] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;
    indices_and_offsets.second[3] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);

    BOOST_CHECK(da._expt_to_dof_mapping.first[0] == 0);
    BOOST_CHECK(da._expt_to_dof_mapping.first[1] == 1);
    BOOST_CHECK(da._expt_to_dof_mapping.first[2] == 2);
    BOOST_CHECK(da._expt_to_dof_mapping.second[0] == 0);
    BOOST_CHECK(da._expt_to_dof_mapping.second[1] == 1);
    BOOST_CHECK(da._expt_to_dof_mapping.second[2] == 3);
  };

  void test_calc_H()
  {
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    unsigned int sim_size = 4;
    unsigned int expt_size = 3;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(3);
    indices_and_offsets.second.resize(4); // offset vector is one longer
    indices_and_offsets.first[0] = 0;
    indices_and_offsets.first[1] = 1;
    indices_and_offsets.first[2] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;
    indices_and_offsets.second[3] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);

    dealii::SparsityPattern pattern(expt_size, sim_size, expt_size);

    dealii::SparseMatrix<double> H = da.calc_H(pattern);

    double tol = 1e-12;
    for (unsigned int i = 0; i < expt_size; ++i)
    {
      for (unsigned int j = 0; j < sim_size; ++j)
      {
        if (i == 0 && j == 0)
          BOOST_CHECK_CLOSE(H(i, j), 1.0, tol);
        else if (i == 1 && j == 1)
          BOOST_CHECK_CLOSE(H(i, j), 1.0, tol);
        else if (i == 2 && j == 3)
          BOOST_CHECK_CLOSE(H(i, j), 1.0, tol);
        else
          BOOST_CHECK_CLOSE(H.el(i, j), 0.0, tol);
      }
    }
  };
  void test_calc_Hx()
  {
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    int sim_size = 4;
    int expt_size = 3;

    dealii::LA::distributed::Vector<double> sim_vec(dof_handler.n_dofs());
    sim_vec(0) = 2.0;
    sim_vec(1) = 4.0;
    sim_vec(2) = 5.0;
    sim_vec(3) = 7.0;

    dealii::Vector<double> expt_vec(3);
    expt_vec(0) = 2.5;
    expt_vec(1) = 4.5;
    expt_vec(2) = 8.5;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(3);
    indices_and_offsets.second.resize(4); // Offset vector is one longer
    indices_and_offsets.first[0] = 0;
    indices_and_offsets.first[1] = 1;
    indices_and_offsets.first[2] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;
    indices_and_offsets.second[3] = 3;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);
    dealii::Vector<double> Hx = da.calc_Hx(sim_vec);

    double tol = 1e-10;
    BOOST_CHECK_CLOSE(Hx(0), 2.0, tol);
    BOOST_CHECK_CLOSE(Hx(1), 4.0, tol);
    BOOST_CHECK_CLOSE(Hx(2), 7.0, tol);
  };

  void test_calc_sample_covariance_dense()
  {
    double tol = 1e-10;

    // Trivial case of identical vectors, covariance should be the zero matrix
    std::vector<dealii::LA::distributed::Vector<double>> data1(3);
    data1[0].reinit(4);
    data1[0](0) = 1.0;
    data1[0](1) = 3.0;
    data1[0](2) = 6.0;
    data1[0](3) = 9.0;
    data1[1].reinit(4);
    data1[1](0) = 1.0;
    data1[1](1) = 3.0;
    data1[1](2) = 6.0;
    data1[1](3) = 9.0;
    data1[2].reinit(4);
    data1[2](0) = 1.0;
    data1[2](1) = 3.0;
    data1[2](2) = 6.0;
    data1[2](3) = 9.0;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    dealii::FullMatrix<double> cov = da.calc_sample_covariance_dense(data1);

    // Check results
    for (unsigned int i = 0; i < 4; ++i)
    {
      for (unsigned int j = 0; j < 4; ++j)
      {
        BOOST_CHECK_SMALL(std::abs(cov(i, j)), tol);
      }
    }

    // Non-trivial case, using NumPy solution as the reference
    std::vector<dealii::LA::distributed::Vector<double>> data2(3);
    data2[0].reinit(5);
    data2[0](0) = 1.0;
    data2[0](1) = 3.0;
    data2[0](2) = 6.0;
    data2[0](3) = 9.0;
    data2[0](4) = 11.0;
    data2[1].reinit(5);
    data2[1](0) = 1.5;
    data2[1](1) = 3.2;
    data2[1](2) = 6.3;
    data2[1](3) = 9.7;
    data2[1](4) = 11.9;
    data2[2].reinit(5);
    data2[2](0) = 1.1;
    data2[2](1) = 3.1;
    data2[2](2) = 6.1;
    data2[2](3) = 9.1;
    data2[2](4) = 11.1;

    da._sim_size = 5;
    dealii::FullMatrix<double> cov2 = da.calc_sample_covariance_dense(data2);

    BOOST_CHECK_CLOSE(cov2(0, 0), 0.07, tol);
    BOOST_CHECK_CLOSE(cov2(1, 0), 0.025, tol);
    BOOST_CHECK_CLOSE(cov2(2, 0), 0.04, tol);
    BOOST_CHECK_CLOSE(cov2(3, 0), 0.1, tol);
    BOOST_CHECK_CLOSE(cov2(4, 0), 0.13, tol);
    BOOST_CHECK_CLOSE(cov2(0, 1), 0.025, tol);
    BOOST_CHECK_CLOSE(cov2(1, 1), 0.01, tol);
    BOOST_CHECK_CLOSE(cov2(2, 1), 0.015, tol);
    BOOST_CHECK_CLOSE(cov2(3, 1), 0.035, tol);
    BOOST_CHECK_CLOSE(cov2(4, 1), 0.045, tol);
    BOOST_CHECK_CLOSE(cov2(0, 2), 0.04, tol);
    BOOST_CHECK_CLOSE(cov2(1, 2), 0.015, tol);
    BOOST_CHECK_CLOSE(cov2(2, 2), 0.02333333333333, tol);
    BOOST_CHECK_CLOSE(cov2(3, 2), 0.05666666666667, tol);
    BOOST_CHECK_CLOSE(cov2(4, 2), 0.07333333333333, tol);
    BOOST_CHECK_CLOSE(cov2(0, 3), 0.1, tol);
    BOOST_CHECK_CLOSE(cov2(1, 3), 0.035, tol);
    BOOST_CHECK_CLOSE(cov2(2, 3), 0.05666666666667, tol);
    BOOST_CHECK_CLOSE(cov2(3, 3), 0.14333333333333, tol);
    BOOST_CHECK_CLOSE(cov2(4, 3), 0.18666666666667, tol);
    BOOST_CHECK_CLOSE(cov2(0, 4), 0.13, tol);
    BOOST_CHECK_CLOSE(cov2(1, 4), 0.045, tol);
    BOOST_CHECK_CLOSE(cov2(2, 4), 0.07333333333333, tol);
    BOOST_CHECK_CLOSE(cov2(3, 4), 0.18666666666667, tol);
    BOOST_CHECK_CLOSE(cov2(4, 4), 0.24333333333333, tol);
  };

  void test_fill_noise_vector()
  {
    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);

    dealii::SparsityPattern pattern(3, 3, 3);
    pattern.add(0, 0);
    pattern.add(1, 0);
    pattern.add(1, 1);
    pattern.add(0, 1);
    pattern.add(2, 2);
    pattern.compress();

    dealii::SparseMatrix<double> R(pattern);

    R.add(0, 0, 0.1);
    R.add(1, 0, 0.3);
    R.add(1, 1, 1.0);
    R.add(0, 1, 0.3);
    R.add(2, 2, 0.2);

    std::vector<dealii::Vector<double>> data;
    dealii::Vector<double> ensemble_member(3);
    for (unsigned int i = 0; i < 1000; ++i)
    {
      da.fill_noise_vector(ensemble_member, R);
      data.push_back(ensemble_member);
    }

    dealii::FullMatrix<double> Rtest = da.calc_sample_covariance_dense(data);

    double tol = 20.; // Loose 20% tolerance because this is a statistical check
    BOOST_CHECK_CLOSE(R(0, 0), Rtest(0, 0), tol);
    BOOST_CHECK_CLOSE(R(1, 0), Rtest(1, 0), tol);
    BOOST_CHECK_CLOSE(R(1, 1), Rtest(1, 1), tol);
    BOOST_CHECK_CLOSE(R(0, 1), Rtest(0, 1), tol);
    BOOST_CHECK_CLOSE(R(2, 2), Rtest(2, 2), tol);
  };

  void test_update_ensemble()
  {
    // Create the DoF mapping
    MPI_Comm communicator = MPI_COMM_WORLD;

    boost::property_tree::ptree database;
    database.put("import_mesh", false);
    database.put("length", 1);
    database.put("length_divisions", 2);
    database.put("height", 1);
    database.put("height_divisions", 2);
    adamantine::Geometry<2> geometry(communicator, database);
    dealii::parallel::distributed::Triangulation<2> const &tria =
        geometry.get_triangulation();

    dealii::FE_Q<2> fe(1);
    dealii::DoFHandler<2> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    int sim_size = 5;
    int expt_size = 2;

    std::vector<double> expt_vec(2);
    expt_vec[0] = 2.5;
    expt_vec[1] = 9.5;

    std::pair<std::vector<int>, std::vector<int>> indices_and_offsets;
    indices_and_offsets.first.resize(2);
    indices_and_offsets.second.resize(3); // Offset vector is one longer
    indices_and_offsets.first[0] = 1;
    indices_and_offsets.first[1] = 3;
    indices_and_offsets.second[0] = 0;
    indices_and_offsets.second[1] = 1;
    indices_and_offsets.second[2] = 2;

    boost::property_tree::ptree solver_settings_database;
    DataAssimilator da(solver_settings_database);
    da._sim_size = sim_size;
    da._expt_size = expt_size;
    da._num_ensemble_members = 3;

    da.update_dof_mapping<2>(dof_handler, indices_and_offsets);

    // Create the simulation data
    std::vector<dealii::LA::distributed::Vector<double>> data(3);
    data[0].reinit(5);
    data[0](0) = 1.0;
    data[0](1) = 3.0;
    data[0](2) = 6.0;
    data[0](3) = 9.0;
    data[0](4) = 11.0;
    data[1].reinit(5);
    data[1](0) = 1.5;
    data[1](1) = 3.2;
    data[1](2) = 6.3;
    data[1](3) = 9.7;
    data[1](4) = 11.9;
    data[2].reinit(5);
    data[2](0) = 1.1;
    data[2](1) = 3.1;
    data[2](2) = 6.1;
    data[2](3) = 9.1;
    data[2](4) = 11.1;

    // Build the sparse experimental covariance matrix
    dealii::SparsityPattern pattern(expt_size, expt_size, 1);
    pattern.add(0, 0);
    pattern.add(1, 1);
    pattern.compress();

    dealii::SparseMatrix<double> R(pattern);
    R.add(0, 0, 0.002);
    R.add(1, 1, 0.001);

    // Save the data at the observation points before assimilation
    std::vector<double> sim_at_expt_pt_1_before(3);
    sim_at_expt_pt_1_before.push_back(data[0][1]);
    sim_at_expt_pt_1_before.push_back(data[1][1]);
    sim_at_expt_pt_1_before.push_back(data[2][1]);

    std::vector<double> sim_at_expt_pt_2_before(3);
    sim_at_expt_pt_2_before.push_back(data[0][3]);
    sim_at_expt_pt_2_before.push_back(data[1][3]);
    sim_at_expt_pt_2_before.push_back(data[2][3]);

    // Update the simulation data
    da.update_ensemble(data, expt_vec, R);

    // Save the data at the observation points after assimilation
    std::vector<double> sim_at_expt_pt_1_after(3);
    sim_at_expt_pt_1_after.push_back(data[0][1]);
    sim_at_expt_pt_1_after.push_back(data[1][1]);
    sim_at_expt_pt_1_after.push_back(data[2][1]);

    std::vector<double> sim_at_expt_pt_2_after(3);
    sim_at_expt_pt_2_after.push_back(data[0][3]);
    sim_at_expt_pt_2_after.push_back(data[1][3]);
    sim_at_expt_pt_2_after.push_back(data[2][3]);

    // Check the solution
    // The observed points should get closer to the experimental values
    // Large entries in R could make these fail spuriously
    for (int member = 0; member < 3; ++member)
    {
      BOOST_CHECK(std::abs(expt_vec[0] - sim_at_expt_pt_1_after[member]) <=
                  std::abs(expt_vec[0] - sim_at_expt_pt_1_before[member]));
      BOOST_CHECK(std::abs(expt_vec[1] - sim_at_expt_pt_2_after[member]) <=
                  std::abs(expt_vec[1] - sim_at_expt_pt_2_before[member]));
    }
  };
};

BOOST_AUTO_TEST_CASE(data_assimilator)
{
  DataAssimilatorTester dat;

  dat.test_constructor();
  dat.test_update_dof_mapping();
  dat.test_calc_sample_covariance_dense();
  dat.test_fill_noise_vector();
  dat.test_calc_H();
  dat.test_calc_Hx();
  dat.test_calc_kalman_gain();
  dat.test_update_ensemble();
}
} // namespace adamantine
