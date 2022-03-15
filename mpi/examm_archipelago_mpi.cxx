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

#include "common/log.hxx"
#include "common/archipelago_config.hxx"
#include "rnn/archipelago_node.hxx"

#include "rnn/generate_nn.hxx"

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

struct MPIArchipelagoIO : public ArchipelagoIO {
  static constexpr int32_t DEFAULT_TAG = 0;

  virtual void send_msg_to(Msg *msg, node_index_type dst) {
    ostringstream oss;
    msg->write_to_stream(oss);

    auto view = oss.view();
    const char *data = view.data();
    const int32_t length = view.size();

    MPI_Send(data, length, MPI_CHAR, dst, DEFAULT_TAG, MPI_COMM_WORLD);
  }

  virtual void send_msg_all(Msg *msg, vector<node_index_type> &dst) {
    ostringstream oss;
    msg->write_to_stream(oss);

    auto view = oss.view();
    const char *data = view.data();
    const int32_t length = view.size();
 
    for (auto d : dst)
      MPI_Send(data, length, MPI_CHAR, d, DEFAULT_TAG, MPI_COMM_WORLD);
  }

  virtual pair<unique_ptr<Msg>, node_index_type> receive_msg() {
    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    int message_length;
    MPI_Get_count(&status, MPI_CHAR, &message_length);

    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    assert(tag == DEFAULT_TAG);

    unique_ptr<char[]> buf(new char[message_length]);
    MPI_Recv(buf.get(), message_length, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);

    return pair(unique_ptr<Msg>(Msg::read_from_array(buf.get(), message_length)), source);
  }

  MPIArchipelagoIO() {}
  virtual ~MPIArchipelagoIO() {}
};

int main(int argc, char **argv) {
  std::cout << "Initializing mpi....";
  MPI_Init(&argc, &argv);
  std::cout << " done." << std::endl;

  int rank, max_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &max_rank);
  max_rank -= 1;

  std::cout << "got rank " << rank << " and max rank " << max_rank << std::endl;

  arguments = vector<string>(argv, argv + argc);

  std::cout << "got arguments!" << std::endl;

  Log::initialize(arguments);
  Log::set_rank(rank);
  Log::set_id("main_" + to_string(rank));
  Log::restrict_to_rank(rank);
  
#define EXAMM_ARCHIPELAGO
#define EXAMM_MPI
#include "common/examm_argparse.cxx"
  
  ifstream f(archipelago_config_path);
  if (!f.good()) {
    Log::fatal("Failed to read archipelago configuration file %s\n", archipelago_config_path.c_str());
    exit(1);
  }
  stringstream buf;
  buf << f.rdbuf();
  string config = buf.str();
  map<string, node_index_type> define_map = {
    {"n_islands", number_islands}
  };
  ArchipelagoConfig archipelago_config = ArchipelagoConfig::from_string(config, max_rank + 1, define_map);
  MPIArchipelagoIO io;

  auto role = archipelago_config.node_roles[rank];

  unique_ptr<ArchipelagoNode<MPIArchipelagoIO>> node;
  auto nn = archipelago_config.connections.size();
  if (rank == 0) {
    Log::info("");
    for (int i = 0; i < nn; i++) {
      string s;
      switch (archipelago_config.node_roles[i]) {
        case node_role::MASTER:
          s = "M"; break;
        case node_role::MANAGERS:
          s = "m"; break;
        case node_role::ISLANDS:
          s = "L"; break;
        case node_role::WORKERS:
          s = "W"; break;
      }
      Log::info_no_header("%s ", s.c_str());
    }
    Log::info_no_header("\n");
    for (int i = 0; i < nn; i++) {
      Log::info("");
      for (int j = 0; j < nn; j++) { Log::info_no_header("%s ", archipelago_config.connections[i][j] ? "X" : "~"); }
      Log::info_no_header("\n");
    }
  }

  if (seed_genome == nullptr) {
    auto seed = unique_ptr<RNN_Genome>(create_ff(dataset_meta.input_parameter_names, 0, 0,
                                            dataset_meta.output_parameter_names, 0, training_parameters,
                                            weight_initialize, weight_inheritance, mutated_component_weight));
    seed->initialize_randomly();
    seed->set_generated_by("initial");
    seed_genome = move(seed);
  }

  Dataset d = {training_inputs, training_outputs, validation_inputs, validation_outputs};


  switch (role) {
    case node_role::MASTER:
      node = unique_ptr<ArchipelagoNode<MPIArchipelagoIO>>(new ArchipelagoMaster<MPIArchipelagoIO>(rank, archipelago_config, io, output_directory + "/fitness_log.csv", max_genomes));
      break;
    case node_role::ISLANDS:
      node = unique_ptr<ArchipelagoNode<MPIArchipelagoIO>>(new ArchipelagoIslandCluster<MPIArchipelagoIO>(rank, archipelago_config, io, 1, 10, move(seed_genome), pair(2, 2), genome_operators));
      break;
    case node_role::MANAGERS:
      node = unique_ptr<ArchipelagoNode<MPIArchipelagoIO>>(new ArchipelagoManager<MPIArchipelagoIO>(rank, archipelago_config, io, 10));
      break;
    case node_role::WORKERS:
      node = unique_ptr<ArchipelagoNode<MPIArchipelagoIO>>(new ArchipelagoWorker<MPIArchipelagoIO>(rank, archipelago_config, io, genome_operators, d));
  }

  node->run();

  delete time_series_sets;

  Log::fatal("REACHED END OF PROGRAM\n");
}
