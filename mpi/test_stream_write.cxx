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


    TimeSeriesSets *time_series_sets = NULL;

    if (rank == 0) {
        //only have the master process print TSS info
        time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
        if (argument_exists(arguments, "--write_time_series")) {
            string base_filename;
            get_argument(arguments, "--write_time_series", true, base_filename);
            time_series_sets->write_time_series_sets(base_filename);
        }
    } else {
        time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    }



    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    int number_inputs = time_series_sets->get_number_inputs();
    int number_outputs = time_series_sets->get_number_outputs();

    Log::debug("number_inputs: %d, number_outputs: %d\n", number_inputs, number_outputs);

    int32_t population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int32_t number_islands;
    get_argument(arguments, "--number_islands", true, number_islands);

    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    string speciation_method = "";
    get_argument(arguments, "--speciation_method", false, speciation_method);

    int32_t extinction_event_generation_number = 0;
    get_argument(arguments, "--extinction_event_generation_number", false, extinction_event_generation_number);
  
    int32_t islands_to_exterminate;
    get_argument(arguments, "--islands_to_exterminate", false, islands_to_exterminate);

    string island_ranking_method = "";
    get_argument(arguments, "--island_ranking_method", false, island_ranking_method);

    string repopulation_method = "";
    get_argument(arguments, "--repopulation_method", false, repopulation_method);

    int32_t repopulation_mutations = 0;
    get_argument(arguments, "--repopulation_mutations", false, repopulation_mutations);

    double species_threshold = 0.0;
    get_argument(arguments, "--species_threshold", false, species_threshold);
    
    double fitness_threshold = 100;
    get_argument(arguments, "--fitness_threshold", false, fitness_threshold);

    double neat_c1 = 1;
    get_argument(arguments, "--neat_c1", false, neat_c1);

    double neat_c2 = 1;
    get_argument(arguments, "--neat_c2", false, neat_c2);

    double neat_c3 = 1;
    get_argument(arguments, "--neat_c3", false, neat_c3);
    bool repeat_extinction = argument_exists(arguments, "--repeat_extinction");

    int32_t epochs_acc_freq = 0;
    get_argument(arguments, "--epochs_acc_freq", false, epochs_acc_freq);
    
    // get_argument(arguments, "--repeat_extinction", false, repeat_extinction);

    //bool use_regression = argument_exists(arguments, "--use_regression");
    bool use_regression = true; //time series will always use regression


    string weight_initialize_string = "random";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);
    WeightType weight_initialize;
    weight_initialize = get_enum_from_string(weight_initialize_string);
    
    string weight_inheritance_string = "lamarckian";
    get_argument(arguments, "--weight_inheritance", false, weight_inheritance_string);
    WeightType weight_inheritance;
    weight_inheritance = get_enum_from_string(weight_inheritance_string);

    string mutated_component_weight_string = "lamarckian";
    get_argument(arguments, "--mutated_component_weight", false, mutated_component_weight_string);
    WeightType mutated_component_weight;
    mutated_component_weight = get_enum_from_string(mutated_component_weight_string);

    int32_t bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    double learning_rate = 0.001;
    get_argument(arguments, "--learning_rate", false, learning_rate);

    double high_threshold = 1.0;
    bool use_high_threshold = get_argument(arguments, "--high_threshold", false, high_threshold);

    double low_threshold = 0.05;
    bool use_low_threshold = get_argument(arguments, "--low_threshold", false, low_threshold);

    double dropout_probability = 0.0;
    bool use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);

    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    vector<string> possible_node_types;
    get_argument_vector(arguments, "--possible_node_types", false, possible_node_types);

    int32_t min_recurrent_depth = 1;
    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);

    int32_t max_recurrent_depth = 10;
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);


    RNN_Genome *seed_genome = NULL;
    string genome_file_name = "";
    if (get_argument(arguments, "--genome_bin", false, genome_file_name)) {
        seed_genome = new RNN_Genome(genome_file_name);
        seed_genome->set_normalize_bounds(time_series_sets->get_normalize_type(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), time_series_sets->get_normalize_avgs(), time_series_sets->get_normalize_std_devs());

        string transfer_learning_version;
        get_argument(arguments, "--transfer_learning_version", true, transfer_learning_version);

        bool epigenetic_weights = argument_exists(arguments, "--epigenetic_weights");

        seed_genome->transfer_to(time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), transfer_learning_version, epigenetic_weights, min_recurrent_depth, max_recurrent_depth);
    }

    bool start_filled = false;
    get_argument(arguments, "--start_filled", false, start_filled);

    Log::clear_rank_restriction();

    examm = new EXAMM(population_size, number_islands, max_genomes, extinction_event_generation_number, islands_to_exterminate, island_ranking_method,
            repopulation_method, repopulation_mutations, repeat_extinction, epochs_acc_freq,
            speciation_method,
            species_threshold, fitness_threshold,
            neat_c1, neat_c2, neat_c3,
            time_series_sets->get_input_parameter_names(),
            time_series_sets->get_output_parameter_names(),
            time_series_sets->get_normalize_type(),
            time_series_sets->get_normalize_mins(),
            time_series_sets->get_normalize_maxs(),
            time_series_sets->get_normalize_avgs(),
            time_series_sets->get_normalize_std_devs(),
            weight_initialize, weight_inheritance, mutated_component_weight,
            bp_iterations, learning_rate,
            use_high_threshold, high_threshold,
            use_low_threshold, low_threshold,
            use_dropout, dropout_probability,
            min_recurrent_depth, max_recurrent_depth,
            use_regression,
            output_directory,
            seed_genome,
            start_filled);

    if (possible_node_types.size() > 0) examm->set_possible_node_types(possible_node_types);

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
