#include <chrono>
#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <algorithm>
using std::min;
using std::max;

#include <utility>
using std::swap;

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
#include "mpi.h"
#include "rnn/examm.hxx"
#include "time_series/time_series.hxx"

#define WORK_REQUEST_TAG 1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG 3
#define TERMINATE_TAG 4

mutex examm_mutex;

vector<string> arguments;

EXAMM *examm;

bool finished = false;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > validation_inputs;
vector<vector<vector<double> > > validation_outputs;

int main(int argc, char **argv) {
  std::cout << "starting up!" << std::endl;
  MPI_Init(&argc, &argv);
  std::cout << "did mpi init!" << std::endl;

  int rank, max_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

#include "common/examm_argparse.cxx"

  examm = make_examm();

  unique_ptr<Msg> m = examm->generate_work();
  WorkMsg *wm = dynamic_cast<WorkMsg *>(m.get());

  unique_ptr<RNN_Genome> genome = wm->get_genome(genome_operators);

  char *byte_array;
  uint32_t length;

  genome->write_to_array(&byte_array, length);

  Log::debug("write to array successful!\n");

  Log::set_id("main_" + to_string(rank));

  finished = true;

  Log::debug("rank %d completed!\n");
  Log::release_id("main_" + to_string(rank));

  MPI_Finalize();
  delete time_series_sets;
  return 0;
}
