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

#include "common/process_arguments.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"
#include "rnn/generate_nn.hxx"
#include "examm/examm.hxx"

#include "examm/examm.hxx"

#include "time_series/time_series.hxx"

#define WORK_REQUEST_TAG 1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG 3
#define TERMINATE_TAG 4

mutex examm_mutex;

vector<string> arguments;

EXAMM *examm;

bool finished = false;

vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

int main(int argc, char** argv) {
    std::cout << "starting up!" << std::endl;
    MPI_Init(&argc, &argv);
    std::cout << "did mpi init!" << std::endl;

    int32_t rank, max_rank;
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

    TimeSeriesSets *time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    get_train_validation_data(arguments, time_series_sets, training_inputs, training_outputs, validation_inputs, validation_outputs);

    WeightUpdate *weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    WeightRules *weight_rules = new WeightRules();
    weight_rules->generate_weight_initialize_from_arguments(arguments);

    RNN_Genome *seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);

    if (rank == 0) {
        //only have the master process print TSS info
        write_time_series_to_file(arguments, time_series_sets);
    } 
    Log::clear_rank_restriction();

    examm = generate_examm_from_arguments(arguments, time_series_sets, weight_rules, seed_genome);
    RNN_Genome *genome = examm->generate_genome();

    char *byte_array;
    int32_t length;

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
