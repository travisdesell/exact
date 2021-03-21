#include <chrono>

#include <iomanip>
using std::setw;
using std::fixed;
using std::setprecision;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "mpi.h"

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "common/weight_initialize.hxx"

#include "rnn/examm.hxx"
#include "rnn/work/work.hxx"

#include "time_series/time_series.hxx"

#include "examm_mpi_core.cxx"

int main(int argc, char** argv) {
    std::cout << "starting up!" << std::endl;
    MPI_Init(&argc, &argv);
    std::cout << "did mpi init!" << std::endl;

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


    std::cout << "initailized log!" << std::endl;
   
#define EXAMM_MPI_STRESS_TEST 1
#include "common/examm_argparse.cxx"
        
    //only have the master process print TSS info
    if (rank == 0 && argument_exists(arguments, "--write_time_series")) {
        string base_filename;
        get_argument(arguments, "--write_time_series", true, base_filename);
        time_series_sets->write_time_series_sets(base_filename);
    }

    Log::clear_rank_restriction();

    if (rank == 0) {
        examm = make_examm();
        master(max_rank, genome_operators);
    } else {
        Log::info("starting worker %d\n", rank);
        worker(rank, make_genome_operators(rank));
    }

    Log::set_id("main_" + to_string(rank));

    finished = true;

    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));

    MPI_Finalize();
    delete time_series_sets;

    return 0;
}
