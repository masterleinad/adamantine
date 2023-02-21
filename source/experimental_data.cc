/* Copyright (c) 2021-2023, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <experimental_data.hh>
#include <utils.hh>

#include <deal.II/arborx/distributed_tree.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <Kokkos_HostSpace.hpp>

#include <cstdlib>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <unordered_set>

#include <ArborX_Ray.hpp>

namespace adamantine
{
/**
 * This class implements a predicate for the intersection of Ray and the first
 * BoundingBox. In ArborX, this is a nearest search between a Ray and the
 * BoundingBoxes.
 */
class RayNearestPredicate
{
public:
  /**
   * Constructor. @p points is a list of points which we are interested in
   * knowing if they intersect ArborXWrappers::BVH bounding boxes.
   */
  RayNearestPredicate(std::vector<Ray<3>> const &rays) : _rays(rays) {}

  /**
   * Number of rays stored in the structure.
   */
  std::size_t size() const { return _rays.size(); }

  /**
   * Return the `i`th Ray stored in the object.
   */
  Ray<3> const &get(unsigned int i) const { return _rays[i]; }

private:
  std::vector<Ray<3>> _rays;
};
} // namespace adamantine

namespace ArborX
{
template <>
struct AccessTraits<adamantine::RayNearestPredicate, PredicatesTag>
{
  using memory_space = Kokkos::HostSpace;

  static std::size_t size(adamantine::RayNearestPredicate const &ray_nearest)
  {
    return ray_nearest.size();
  }

  static auto get(adamantine::RayNearestPredicate const &ray_nearest,
                  std::size_t i)
  {
    auto const &ray = ray_nearest.get(i);
    auto const &origin = ray.origin;
    auto const &direction = ray.direction;
    ArborX::Experimental::Ray arborx_ray = {
        ArborX::Point{(float)origin[0], (float)origin[1], (float)origin[2]},
        ArborX::Experimental::Vector{(float)direction[0], (float)direction[1],
                                     (float)direction[2]}};
    return nearest(arborx_ray, 1);
  }
};
} // namespace ArborX

namespace adamantine
{
template <int dim>
std::vector<PointsValues<dim>> read_experimental_data_point_cloud(
    MPI_Comm const &communicator,
    boost::property_tree::ptree const &experiment_database)
{
  // Format of the file names: the format is pretty arbitrary, #frame and
  // #camera are replaced by the frame and the camera number.
  // PropertyTreeInput experiment.file
  std::string data_filename = experiment_database.get<std::string>("file");
  // PropertyTreeInput experiment.first_frame
  unsigned int first_frame = experiment_database.get("first_frame", 0);
  // PropertyTreeInput experiment.last_frame
  unsigned int last_frame = experiment_database.get<unsigned int>("last_frame");
  // PropertyTreeInput experiment.first_camera_id
  unsigned int first_camera_id =
      experiment_database.get<unsigned int>("first_camera_id");
  // PropertyTreeInput experiment.last_camera_id
  unsigned int last_camera_id = experiment_database.get<int>("last_camera_id");
  // PropertyTreeInput experiment.data_columns
  std::string data_columns =
      experiment_database.get<std::string>("data_columns");

  std::vector<PointsValues<dim>> points_values_all_frames(last_frame + 1 -
                                                          first_frame);
  for (unsigned int frame = first_frame; frame < last_frame + 1; ++frame)
  {
    PointsValues<dim> points_values;
    for (unsigned int camera_id = first_camera_id;
         camera_id < last_camera_id + 1; ++camera_id)
    {
      // Use regex to get the next file to read
      std::regex camera_regex("#camera");
      std::regex frame_regex("#frame");
      auto regex_filename =
          std::regex_replace((std::regex_replace(data_filename, camera_regex,
                                                 std::to_string(camera_id))),
                             frame_regex, std::to_string(frame));
      std::string error_message =
          "The file " + regex_filename + " does not exist.";
      ASSERT(boost::filesystem::exists(regex_filename), error_message.c_str());
      std::string filename("data_" + std::to_string(frame) + "_" +
                           std::to_string(camera_id) + ".csv");

      // Use bash to create a new file that only contains the columns that we
      // care about. For large files this divides by four the time to parse the
      // files. It also simplifies reading the files. Only rank zero renames
      // the file.
      if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
      {
        std::string cut_command("cut -d, -f" + data_columns + " " +
                                regex_filename + " > " + filename);
        [[maybe_unused]] int error_code = std::system(cut_command.c_str());
        ASSERT(error_code == 0, "Problem with the cut command.");
      }

      // Wait for rank zero before reading the file.
      MPI_Barrier(communicator);

      // Read and parse the file
      std::ifstream file;
      file.open(filename);
      std::string line;
      std::getline(file, line);
      while (std::getline(file, line))
      {
        std::size_t pos = 0;
        std::size_t last_pos = 0;
        std::size_t line_length = line.length();
        unsigned int i = 0;
        dealii::Point<dim> point;
        double value = 0.;
        while (last_pos < line_length + 1)
        {
          pos = line.find_first_of(",", last_pos);
          // If no comma was found that we read until the end of the file
          if (pos == std::string::npos)
          {
            pos = line_length;
          }

          if (pos != last_pos)
          {
            char *end = line.data() + pos;
            if (i < dim)
            {
              point[i] = std::strtod(line.data() + last_pos, &end);
            }
            else
            {
              value = std::strtod(line.data() + last_pos, &end);
            }

            ++i;
          }

          last_pos = pos + 1;
        }

        points_values.points.push_back(point);
        points_values.values.push_back(value);
      }

      // Wait for every rank to be done reading the temporary stripped file and
      // then remove it.
      MPI_Barrier(communicator);
      if (dealii::Utilities::MPI::this_mpi_process(communicator) == 0)
      {
        std::string rm_command("rm " + filename);
        [[maybe_unused]] int error_code = std::system(rm_command.c_str());
        ASSERT(error_code == 0, "Error with the rm command.");
      }
    }
    points_values_all_frames[frame - first_frame] = points_values;
  }

  return points_values_all_frames;
}

template <int dim>
std::pair<std::vector<dealii::types::global_dof_index>,
          std::vector<dealii::Point<dim>>>
get_dof_to_support_mapping(dealii::DoFHandler<dim> const &dof_handler)
{
  std::vector<dealii::types::global_dof_index> dof_indices;
  std::vector<dealii::Point<dim>> support_points;
  std::unordered_set<dealii::types::global_dof_index> visited_dof_indices;

  // Manually do what dealii::DoFTools::map_dofs_to_support_points does, since
  // that doesn't currently work with FE_Nothing
  const dealii::FiniteElement<dim> &fe = dof_handler.get_fe(0);

  dealii::FEValues<dim, dim> fe_values(fe, fe.get_unit_support_points(),
                                       dealii::update_quadrature_points);

  std::vector<dealii::types::global_dof_index> local_dof_indices(
      fe.n_dofs_per_cell());

  for (auto const &cell : dealii::filter_iterators(
           dof_handler.active_cell_iterators(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0, true)))
  {
    fe_values.reinit(cell);
    cell->get_dof_indices(local_dof_indices);
    const std::vector<dealii::Point<dim>> &points =
        fe_values.get_quadrature_points();
    for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    {
      // Skip duplicate points like vertices
      if (visited_dof_indices.count(local_dof_indices[i]) == 0)
      {
        dof_indices.push_back(local_dof_indices[i]);
        support_points.push_back(points[i]);
        visited_dof_indices.insert(local_dof_indices[i]);
      }
    }
  }

  return {dof_indices, support_points};
}

template <int dim>
std::pair<std::vector<int>, std::vector<int>>
get_expt_to_dof_mapping(PointsValues<dim> const &points_values,
                        dealii::DoFHandler<dim> const &dof_handler)
{
  auto [dof_indices, support_points] = get_dof_to_support_mapping(dof_handler);

  // Perform the search. This needs to be a distributed tree so that we only
  // take care of locally owned nearest points_values. Otherwise all the
  // points_values will be found on every rank.
  auto communicator = dof_handler.get_communicator();
  dealii::ArborXWrappers::DistributedTree distributed_tree(communicator,
                                                           support_points);
  dealii::ArborXWrappers::PointNearestPredicate pt_nearest(points_values.points,
                                                           1);
  auto [indices_ranks, offset] = distributed_tree.query(pt_nearest);

  // Convert the indices and offsets to a pair that maps experimental indices to
  // dof indices
  int const my_rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  std::pair<std::vector<int>, std::vector<int>> expt_to_dof_mapping;
  unsigned int const n_queries = points_values.points.size();
  for (unsigned int i = 0; i < n_queries; ++i)
  {
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      if (indices_ranks[j].second == my_rank)
      {
        expt_to_dof_mapping.first.push_back(i);
        expt_to_dof_mapping.second.push_back(
            dof_indices[indices_ranks[j].first]);
      }
    }
  }

  return expt_to_dof_mapping;
}

template <int dim>
void set_with_experimental_data(
    PointsValues<dim> const &points_values,
    std::pair<std::vector<int>, std::vector<int>> &expt_to_dof_mapping,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature)
{
  for (unsigned int i = 0; i < points_values.values.size(); ++i)
  {
    temperature[expt_to_dof_mapping.second[i]] = points_values.values[i];
  }

  temperature.compress(dealii::VectorOperation::insert);
}

std::vector<std::vector<double>>
read_frame_timestamps(boost::property_tree::ptree const &experiment_database)
{
  // PropertyTreeInput experiment.log_filename
  std::string log_filename =
      experiment_database.get<std::string>("log_filename");

  [[maybe_unused]] std::string error_message =
      "The file " + log_filename + " does not exist.";
  ASSERT(boost::filesystem::exists(log_filename), error_message.c_str());

  // PropertyTreeInput experiment.first_frame_temporal_offset
  double first_frame_offset =
      experiment_database.get("first_frame_temporal_offset", 0.0);

  // PropertyTreeInput experiment.first_frame
  unsigned int first_frame =
      experiment_database.get<unsigned int>("first_frame", 0);
  // PropertyTreeInput experiment.last_frame
  unsigned int last_frame = experiment_database.get<unsigned int>("last_frame");

  // PropertyTreeInput experiment.first_camera_id
  unsigned int first_camera_id =
      experiment_database.get<unsigned int>("first_camera_id");
  // PropertyTreeInput experiment.last_camera_id
  unsigned int last_camera_id =
      experiment_database.get<unsigned int>("last_camera_id");

  unsigned int num_cameras = last_camera_id - first_camera_id + 1;
  std::vector<std::vector<double>> time_stamps(num_cameras);

  std::vector<double> first_frame_value(num_cameras);

  // Read and parse the file
  std::ifstream file;
  file.open(log_filename);
  std::string line;
  while (std::getline(file, line))
  {
    unsigned int entry_index = 0;
    std::stringstream s_stream(line);
    bool frame_of_interest = false;
    unsigned int frame = std::numeric_limits<unsigned int>::max();
    while (s_stream.good())
    {
      std::string substring;
      std::getline(s_stream, substring, ',');
      boost::trim(substring);

      if (entry_index == 0)
      {
        error_message = "The file " + log_filename +
                        " does not have consecutive frame indices.";
        ASSERT_THROW(std::stoi(substring) - frame == 1 ||
                         frame == std::numeric_limits<unsigned int>::max(),
                     error_message.c_str());
        frame = std::stoi(substring);
        if (frame >= first_frame && frame <= last_frame)
          frame_of_interest = true;
      }
      else
      {
        if (frame == first_frame && substring.size() > 0)
          first_frame_value[entry_index - 1] = std::stod(substring);

        if (frame_of_interest && substring.size() > 0)
          time_stamps[entry_index - 1].push_back(
              std::stod(substring) - first_frame_value[entry_index - 1] +
              first_frame_offset);
      }
      entry_index++;
    }
  }
  return time_stamps;
}

RayTracing::RayTracing(boost::property_tree::ptree const &experiment_database)
{

  // Format of the file names: the format is pretty arbitrary, #frame and
  // #camera are replaced by the frame and the camera number.
  // PropertyTreeInput experiment.file
  _data_filename = experiment_database.get<std::string>("file");
  // PropertyTreeInput experiment.first_frame
  _next_frame = experiment_database.get("first_frame", 0);
  // PropertyTreeInput experiment.first_camera_id
  _first_camera_id = experiment_database.get<unsigned int>("first_camera_id");
  // PropertyTreeInput experiment.last_camera_id
  _last_camera_id = experiment_database.get<int>("last_camera_id");
}

void RayTracing::read_next_frame()
{
  _rays_current_frame.clear();
  _values_current_frame.clear();
  for (unsigned int camera_id = _first_camera_id;
       camera_id < _last_camera_id + 1; ++camera_id)
  {
    // Use regex to get the next file to read
    std::regex camera_regex("#camera");
    std::regex frame_regex("#frame");
    auto filename =
        std::regex_replace((std::regex_replace(_data_filename, camera_regex,
                                               std::to_string(camera_id))),
                           frame_regex, std::to_string(_next_frame));
    unsigned int counter = 1;
    while (boost::filesystem::exists(filename))
    {
      // Spin loop waiting for the file to appear
      if (counter == 0)
        std::cout << "Waiting for the next frame" << std::endl;
      ++counter;
    }

    // Read and parse the file
    std::ifstream file;
    file.open(filename);
    std::string line;
    std::getline(file, line); // skip the header
    while (std::getline(file, line))
    {
      std::size_t pos = 0;
      std::size_t last_pos = 0;
      std::size_t line_length = line.length();
      unsigned int i = 0;
      dealii::Point<dim> point;
      dealii::Tensor<1, dim> direction;
      double value = 0.;
      while (last_pos < line_length + 1)
      {
        pos = line.find_first_of(",", last_pos);
        // If no comma was found that we read until the end of the file
        if (pos == std::string::npos)
        {
          pos = line_length;
        }

        if (pos != last_pos)
        {
          char *end = line.data() + pos;
          if (i < dim)
          {
            point[i] = std::strtod(line.data() + last_pos, &end);
          }
          else if (i < 2 * dim)
          {
            direction[i - dim] = std::strtod(line.data() + last_pos, &end);
          }
          else
          {
            value = std::strtod(line.data() + last_pos, &end);
          }

          ++i;
        }

        last_pos = pos + 1;
      }

      Ray<dim> ray{point, direction};
      _rays_current_frame.push_back(ray);
      _values_current_frame.push_back(value);
    }
  }
  ++_next_frame;
}

PointsValues<3>
RayTracing::get_intersection(dealii::DoFHandler<3> const &dof_handler)
{
  PointsValues<dim> points_values;

  // Perform the ray tracing to get the cells that are intersected by rays

  // Create the bounding boxes associated to the locally owned cells with FE
  // index = 0
  std::vector<dealii::BoundingBox<dim>> bounding_boxes;
  std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>
      cell_iterators;
  for (auto const &cell : dealii::filter_iterators(
           dof_handler.active_cell_iterators(),
           dealii::IteratorFilters::LocallyOwnedCell(),
           dealii::IteratorFilters::ActiveFEIndexEqualTo(0)))
  {
    bounding_boxes.push_back(cell->bounding_box());
    cell_iterators.push_back(cell);
  }

  // Use ArborX to find where the rays intersect the activated cells. All the
  // processors have access to all the rays but we still need to use
  // DistributedTree because some rays can be stopped by activated cells on a
  // different processors. Since the rays are on all the processors, we don't
  // need to communicate the results to other processors.
  auto communicator = dof_handler.get_communicator();
  dealii::ArborXWrappers::DistributedTree distributed_tree(communicator,
                                                           bounding_boxes);
  RayNearestPredicate ray_nearest(_rays_current_frame);
  auto [indices_ranks, offset] = distributed_tree.query(ray_nearest);

  // Find the exact intersections points
  // See https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
  // NOTE that we assume that the faces are flat. If the faces are curved,
  // this is wrong.
  int const my_rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  unsigned int const n_rays = _rays_current_frame.size();
  unsigned int n_intersections = 0;
  for (unsigned int i = 0; i < n_rays; ++i)
  {
    if (offset[i] != offset[i + 1])
    {
      for (int j = offset[i]; j < offset[i + 1]; ++j)
      {
        if (indices_ranks[j].second == my_rank)
        {
          ++n_intersections;
        }
      }
    }
  }

  // If there are no intersections, leave early
  if (n_intersections == 0)
    return points_values;

  std::vector<double> distances(n_intersections,
                                std::numeric_limits<double>::max());
  points_values.points.resize(n_intersections);
  points_values.values.resize(n_intersections);
  auto constexpr reference_cell = dealii::ReferenceCells::get_hypercube<dim>();
  double constexpr tolerance = 1e-12;
  unsigned int ii = 0;
  for (unsigned int i = 0; i < n_rays; ++i)
  {
    points_values.values[ii] = _values_current_frame[i];
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      if (indices_ranks[j].second == my_rank)
      {
        auto const &cell = cell_iterators[indices_ranks[j].first];
        // We know that the ray intersects the bounding box but we don't know
        // where it intersects the cells. We need to check the intersection of
        // the ray with each face of the cell.
        for (unsigned int f = 0; f < reference_cell.n_faces(); ++f)
        {
          // First we check if the ray is parallel to the face. If this is the
          // case, either the ray misses the face or the ray hits the edge of
          // the face. In that last case, the ray is also orthogonal to
          // another face and it is safe to discard all rays parallel to a
          // face.
          auto const point_0 = cell->face(f)->vertex(0);
          auto const point_1 = cell->face(f)->vertex(1);
          auto const point_2 = cell->face(f)->vertex(2);
          dealii::Tensor<1, dim> edge_01({point_1[0] - point_0[0],
                                          point_1[1] - point_0[1],
                                          point_1[2] - point_0[2]});
          dealii::Tensor<1, dim> edge_02({point_2[0] - point_0[0],
                                          point_2[1] - point_0[1],
                                          point_2[2] - point_0[2]});
          auto const &ray_direction = _rays_current_frame[i].direction;
          dealii::Tensor<2, dim> matrix(
              {{-ray_direction[0], -ray_direction[1], -ray_direction[2]},
               {edge_01[0], edge_01[1], edge_01[2]},
               {edge_02[0], edge_02[1], edge_02[2]}});
          double det = dealii::determinant(matrix);
          // If determinant is close to zero, the ray is parallel to the face
          // and we go to the next face.
          if (std::abs(det) < tolerance)
            continue;
          // Compute the distance along the ray direction between the origin
          // of the ray and the intersection point.
          auto const cross_product = dealii::cross_product_3d(edge_01, edge_02);
          auto const &ray_origin = _rays_current_frame[i].origin;
          dealii::Tensor<1, dim> p0_ray({ray_origin[0] - point_0[0],
                                         ray_origin[1] - point_0[1],
                                         ray_origin[2] - point_0[2]});
          double d = cross_product * p0_ray / det;
          // We can finally compute the intersection point. It is possible
          // that a ray intersects multiple faces. For instance if the mesh is
          // a cube the ray will get into the cube from one face and it will
          // get out of the cube by the opposite face. The correct
          // intersection point is the one with the smallest distance.
          if (d < distances[ii])
          {
            // The point intersects the plane of the face but maybe not the
            // face itself. Check that the point is on the face. NOTE: We
            // assume that the face is an axis-aligned rectangle.
            dealii::Point<dim> intersection = ray_origin + d * ray_direction;
            std::vector<double> min(dim, std::numeric_limits<double>::max());
            std::vector<double> max(dim, std::numeric_limits<double>::lowest());
            for (unsigned int coord = 0; coord < dim; ++coord)
            {
              if (point_0[coord] < min[coord])
                min[coord] = point_0[coord];
              if (point_0[coord] > max[coord])
                max[coord] = point_0[coord];

              if (point_1[coord] < min[coord])
                min[coord] = point_1[coord];
              if (point_1[coord] > max[coord])
                max[coord] = point_1[coord];

              if (point_2[coord] < min[coord])
                min[coord] = point_2[coord];
              if (point_2[coord] > max[coord])
                max[coord] = point_2[coord];
            }

            bool on_the_face = true;
            for (unsigned int coord = 0; coord < 3; ++coord)
            {
              // NOTE: We could add a tolerance if the intersection point is
              // on the edge. Currently, we may lose some rays but I don't
              // think it matters. The mesh does not match exactly the real
              // object anyway.
              if ((intersection[coord] < min[coord]) ||
                  (intersection[coord] > max[coord]))
              {
                on_the_face = false;
                break;
              }
            }

            if (on_the_face)
            {
              points_values.points[ii] = intersection;
              distances[ii] = d;
            }
          }
        }
      }
      if (offset[i] != offset[i + 1])
        ++ii;
    }
  }

  return points_values;
}

} // namespace adamantine

//-------------------- Explicit Instantiations --------------------//
namespace adamantine
{
template std::vector<PointsValues<2>> read_experimental_data_point_cloud(
    MPI_Comm const &communicator,
    boost::property_tree::ptree const &experiment_database);
template std::vector<PointsValues<3>> read_experimental_data_point_cloud(
    MPI_Comm const &communicator,
    boost::property_tree::ptree const &experiment_database);

template void set_with_experimental_data(
    PointsValues<2> const &points_values,
    std::pair<std::vector<int>, std::vector<int>> &expt_to_dof_mapping,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature);
template void set_with_experimental_data(
    PointsValues<3> const &points_values,
    std::pair<std::vector<int>, std::vector<int>> &expt_to_dof_mapping,
    dealii::LinearAlgebra::distributed::Vector<double> &temperature);
template std::pair<std::vector<int>, std::vector<int>>
get_expt_to_dof_mapping(PointsValues<2> const &points_values,
                        dealii::DoFHandler<2> const &dof_handler);
template std::pair<std::vector<int>, std::vector<int>>
get_expt_to_dof_mapping(PointsValues<3> const &points_values,
                        dealii::DoFHandler<3> const &dof_handler);
} // namespace adamantine
