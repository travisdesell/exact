#include <assert.h>
#include <mpi.h>

#include <iostream>
using std::cout;

#include <chrono>
#include <thread>
#include <limits>
using std::numeric_limits;

#include <algorithm>
using std::max;
using std::min;
using std::swap;

#include "../common/log.hxx"
#include "../common/archipelago_config.hxx"
// #include "../rnn/archipelago_node.hxx"

#include <chrono>
#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "common/weight_initialize.hxx"
#include "examm_mpi_core.cxx"
#include "mpi.h"
#include "rnn/examm.hxx"
#include "time_series/time_series.hxx"
using namespace std::literals;

int main(int argc, char **argv) {
  std::cout << "Initializing mpi....";
  MPI_Init(&argc, &argv);
  std::cout << " done." << std::endl;

  int rank, max_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

  std::cout << "got rank " << rank << " and max rank " << max_rank << std::endl;

  arguments = vector<string>(argv, argv + argc);

  std::cout << "got arguments!" << std::endl;

  Log::initialize(arguments);
  Log::set_rank(rank);
  Log::set_id("main_" + to_string(rank));
  Log::restrict_to_rank(0);
  
#define EXAMM_ARCHIPELAGO
#include "common/examm_argparse.cxx"
  
  ifstream f(archipelago_config_path);
  if (!f.good()) {
    Log::fatal("Failed to read archipelago configuration file %s\n", archipelago_config_path.c_str());
    exit(1);
  }
  stringstream buf;
  buf << f.rdbuf();
  string config = buf.str();
  ArchipelagoConfig archipelago_config = ArchipelagoConfig::from_string(config, max_rank + 1);
}
