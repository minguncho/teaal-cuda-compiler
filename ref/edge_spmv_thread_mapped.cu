/**
 * @file output.cu
 * Generated Loops code (What should look like)
*/

#include "helpers.hxx"
#include <loops/schedule.hxx>
#include <loops/container/formats.hxx>
#include <loops/container/vector.hxx>
#include <loops/util/launch.hxx>
#include <loops/util/device.hxx>
#include <loops/memory.hxx>
#include <iostream>

using namespace loops;

template <typename setup_t,
          typename index_t,
          typename offset_t,
          typename type_t>
__global__ void __thread_mapped(setup_t config,
                                const std::size_t rows,
                                const std::size_t cols,
                                const std::size_t nnz,
                                const offset_t* offsets,
                                const index_t* indices,
                                const type_t* values,
                                const type_t* x,
                                type_t* y) {
  for (auto row : config.tiles()) {
    type_t sum = 0;

    for (auto nz : config.atoms(row)) {
      sum += values[nz] * x[indices[nz]];
    }

    y[row] = sum;
  }
}

int main(int argc, char** argv) {
  using index_t = int;
  using offset_t = int;
  using type_t = float;

  // ... I/O parameters, mtx, etc.
  parameters_t parameters(argc, argv);

  matrix_market_t<index_t, offset_t, type_t> mtx;
  csr_t<index_t, offset_t, type_t> csr(mtx.load(parameters.filename));

  // Input and output vectors.
  vector_t<type_t> x(csr.cols);
  vector_t<type_t> y(csr.rows);

  // Generate random numbers between [0, 1].
  generate::random::uniform_distribution(x.begin(), x.end(), 1, 10);

  // Create a schedule.
  using setup_t = schedule::setup<schedule::algorithms_t::thread_mapped, 1, 1,
                                  index_t, offset_t>;
  setup_t config(csr.offsets.data().get(), csr.rows, csr.nnzs);

  // Set-up kernel launch parameters and run the kernel.
  constexpr std::size_t block_size = 128;
  std::size_t grid_size = (csr.rows + block_size - 1) / block_size;
  cudaStream_t stream = 0;

  // Run the benchmark.
  util::timer_t timer;
  timer.start();

  launch::non_cooperative(
      stream, __thread_mapped<setup_t, index_t, offset_t, type_t>, grid_size,
      block_size, config, csr.rows, csr.cols, csr.nnzs,
      csr.offsets.data().get(), csr.indices.data().get(),
      csr.values.data().get(), x.data().get(), y.data().get());

  cudaStreamSynchronize(stream);
  timer.stop();

  std::cout << "thread_mapped," << mtx.dataset << "," << csr.rows << ","
            << csr.cols << "," << csr.nnzs << "," << timer.milliseconds()
            << std::endl;

  // Validation.
  if (parameters.validate)
    cpu::validate(parameters, csr, x, y);
}