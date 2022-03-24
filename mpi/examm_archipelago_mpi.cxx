#include <assert.h>
#include <mpi.h>

#include <iostream>
using std::cout;

#include <chrono>
#include <limits>
#include <thread>
using std::numeric_limits;

#include <mutex>
using std::mutex;

#include <algorithm>
using std::max;
using std::min;
using std::swap;

#include <chrono>
#include <iomanip>

#include "common/archipelago_config.hxx"
#include "common/log.hxx"
#include "rnn/archipelago_node.hxx"
#include "rnn/generate_nn.hxx"
using std::fixed;
using std::setprecision;
using std::setw;

#include <mutex>
using std::mutex;

#include <condition_variable>
using std::condition_variable;

#include <atomic>
using std::atomic_bool;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include <deque>
using std::deque;

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

  atomic_bool done = atomic_bool(false);
  mutex mpi_lock;
  condition_variable outgoing_message_pending;

  // Pairs of sent message requests and the associated data.
  deque<pair<vector<MPI_Request>, unique_ptr<ostringstream>>> pending;

  virtual void send_msg_to(Msg *msg, node_index_type dst) {
    unique_ptr<ostringstream> oss = make_unique<ostringstream>();
    msg->write_to_stream(*oss);

    auto content = oss->view();
    const char *data = content.data();
    const int32_t length = content.size();

    vector<MPI_Request> handles;

    MPI_Request request;
    mpi_lock.lock();
    int r = MPI_Isend(data, length, MPI_CHAR, dst, DEFAULT_TAG, MPI_COMM_WORLD, &request);
    handles.push_back(request);
    pending.push_back(pair(move(handles), move(oss)));
    mpi_lock.unlock();
    outgoing_message_pending.notify_one();
  }

  virtual void send_msg_all(Msg *msg, vector<node_index_type> &dst) {
    if (dst.size() == 0) return;
    unique_ptr<ostringstream> oss = make_unique<ostringstream>();
    msg->write_to_stream(*oss);

    auto content = oss->view();
    const char *data = content.data();
    const int32_t length = content.size();

    vector<MPI_Request> handles;

    mpi_lock.lock();
    for (auto d : dst) {
      MPI_Request request;
      int r = MPI_Isend(data, length, MPI_CHAR, d, DEFAULT_TAG, MPI_COMM_WORLD, &request);
      handles.push_back(request);
    }
    pending.push_back(pair(move(handles), move(oss)));
    mpi_lock.unlock();

    outgoing_message_pending.notify_one();
  }

  virtual pair<unique_ptr<Msg>, node_index_type> receive_msg() {
    mpi_lock.lock();

    MPI_Status status;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    int message_length;
    MPI_Get_count(&status, MPI_CHAR, &message_length);

    int source = status.MPI_SOURCE;
    int tag = status.MPI_TAG;
    assert(tag == DEFAULT_TAG);


    unique_ptr<char[]> buf(new char[message_length]);
    MPI_Recv(buf.get(), message_length, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
    Log::info("Received message from %d\n", source);

    mpi_lock.unlock();

    return pair(unique_ptr<Msg>(Msg::read_from_array(buf.get(), message_length)), source);
  }

  MPIArchipelagoIO() {}
  virtual ~MPIArchipelagoIO() {}

  void start_monitor_thread(int rank) {
    std::thread([this, rank]() {
      Log::set_id("monitor_" + to_string(rank));
      while (!done) {
        std::unique_lock<std::mutex> lk(mpi_lock);
        outgoing_message_pending.wait(lk);

        for (;;) {
          if (pending.size() == 0)
            break;
          
          auto &p = this->pending[0];
          assert(p.first.size() > 0);

          int complete = 0;
          MPI_Status status[p.first.size()];
          int s = MPI_Testall(p.first.size(), &p.first[0], &complete, status);

          if (!complete) {
            break;
          }

          pending.pop_front();
        }
      }
    }).detach();
  }

  void clean_up() {
    pending.clear();
  }
};

int main(int argc, char **argv) {
  std::cout << "Initializing mpi...";
  MPI_Init(&argc, &argv);
  std::cout << " done." << std::endl;

  int rank, max_rank, n_mpi_nodes;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_mpi_nodes);
  max_rank = n_mpi_nodes - 1;

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
  
  int n_cluster_nodes = -1;
  get_argument(arguments, "--n-nodes", false, n_cluster_nodes);
  int node_size = -1;
  get_argument(arguments, "--node-size", false, node_size);

  ifstream f(archipelago_config_path);
  if (!f.good()) {
    Log::fatal("Failed to read archipelago configuration file %s\n", archipelago_config_path.c_str());
    exit(1);
  }
  stringstream buf;
  buf << f.rdbuf();
  string config = buf.str();
  map<string, node_index_type> define_map = {
      {"n_islands", number_islands},
      {"n_nodes", n_cluster_nodes},
      {"node_size", node_size}
  };
  ArchipelagoConfig archipelago_config = ArchipelagoConfig::from_string(config, max_rank + 1, define_map);
  MPIArchipelagoIO io;

  auto role = archipelago_config.node_roles[rank];

  unique_ptr<ArchipelagoNode<MPIArchipelagoIO>> node;
  auto nn = archipelago_config.connections.size();
  Log::info("MASTER IS %d\n", archipelago_config.master_id);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == archipelago_config.master_id) {
    Log::info("");
    for (int i = 0; i < nn; i++) {
      string s;
      switch (archipelago_config.node_roles[i]) {
        case node_role::MASTER:
          s = "M";
          break;
        case node_role::MANAGERS:
          s = "m";
          break;
        case node_role::ISLANDS:
          s = "L";
          break;
        case node_role::WORKERS:
          s = "W";
          break;
        case node_role::NOROLE:
          s = " ";
          break;
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
  
  MPI_Barrier(MPI_COMM_WORLD);

  if (seed_genome == nullptr) {
    auto seed = unique_ptr<RNN_Genome>(create_ff(dataset_meta.input_parameter_names, 0, 0,
                                                 dataset_meta.output_parameter_names, 0, training_parameters,
                                                 weight_initialize, weight_inheritance, mutated_component_weight));
    seed->initialize_randomly();
    seed->set_generated_by("initial");

    seed_genome = move(seed);
  }

  edge_inon eic = edge_inon(seed_genome->get_max_edge_inon().inon + 1);
  node_inon nic = node_inon(seed_genome->get_max_node_inon().inon + 1);

  edge_inon::init(eic.inon + rank, n_mpi_nodes);
  node_inon::init(nic.inon + rank, n_mpi_nodes);

  Dataset d = {training_inputs, training_outputs, validation_inputs, validation_outputs};

  switch (role) {
    case node_role::MASTER:
      node = unique_ptr<ArchipelagoNode<MPIArchipelagoIO>>(new ArchipelagoMaster<MPIArchipelagoIO>(
          rank, archipelago_config, io, output_directory + "/fitness_log.csv", max_genomes));
      break;
    case node_role::ISLANDS:
      node = unique_ptr<ArchipelagoNode<MPIArchipelagoIO>>(new ArchipelagoIslandCluster<MPIArchipelagoIO>(
          rank, archipelago_config, io, 1, 10, move(seed_genome), pair(2, 2), genome_operators));
      break;
    case node_role::MANAGERS:
      node = unique_ptr<ArchipelagoNode<MPIArchipelagoIO>>(
          new ArchipelagoManager<MPIArchipelagoIO>(rank, archipelago_config, io, 10));
      break;
    case node_role::WORKERS:
      node = unique_ptr<ArchipelagoNode<MPIArchipelagoIO>>(
          new ArchipelagoWorker<MPIArchipelagoIO>(rank, archipelago_config, io, genome_operators, d));
    case node_role::NOROLE:
      {}
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (node != nullptr) {
    io.start_monitor_thread(rank);
    node->run();
  } else {
    delete time_series_sets;
    goto done;
  }

  delete time_series_sets;
  io.done.store(true);
  io.outgoing_message_pending.notify_one();
  Log::fatal("Cleaning up rank %d\n", rank);

  io.clean_up();
done:
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  Log::fatal("REACHED END OF PROGRAM\n");
}
