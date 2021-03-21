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

#include "rnn/examm.hxx"

#include "word_series/word_series.hxx"

#include "examm_mpi_core.cxx"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);

#define EXAMM_NLP
#include "common/examm_argparse.cxx"

    if (rank == 0) {
        //only have the master process print TSS info
        if (argument_exists(arguments, "--write_word_series")) {
            string base_filename;
            get_argument(arguments, "--write_word_series", true, base_filename);
            corpus_sets->write_sentence_series_sets(base_filename);
        }
    }

    Log::clear_rank_restriction();

    if (rank == 0) {
        examm = make_examm();
        master(max_rank, make_genome_operators(rank));
    } else {
        worker(rank, make_genome_operators(rank));
    }
    Log::set_id("main_" + to_string(rank));

    finished = true;

    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));

    MPI_Finalize();
    delete corpus_sets;
    return 0;
}
