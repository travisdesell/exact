#include <chrono>
#include <algorithm>  
#include <iomanip>
using std::shuffle;
using std::setw;
using std::fixed;
using std::setprecision;
using std::min;

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
// #include "rnn/onenet_speciation_strategy.hxx"

#include "time_series/time_series.hxx"
#include "time_series/online_series.hxx"

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

bool random_sequence_length;
int sequence_length_lower_bound = 30;
int sequence_length_upper_bound = 100;

vector<int32_t> time_series_index;
int32_t generation_genomes = 10;
double noise_std = 0.1;
int32_t number_islands;


void send_work_request(int target) {
    int work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}

void receive_work_request(int source) {
    MPI_Status status;
    int work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

RNN_Genome* receive_genome_from(int source) {
    MPI_Status status;
    int length_message[1];
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);

    int length = length_message[0];

    Log::debug("receiving genome of length: %d from: %d\n", length, source);

    char* genome_str = new char[length + 1];

    Log::debug("receiving genome from: %d\n", source);
    MPI_Recv(genome_str, length, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);

    genome_str[length] = '\0';

    Log::trace("genome_str:\n%s\n", genome_str);

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete [] genome_str;
    return genome;
}

void send_genome_to(int target, RNN_Genome* genome) {
    char *byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("sending genome of length: %d to: %d\n", length, target);

    int length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    Log::debug("sending genome to: %d\n", target);
    MPI_Send(byte_array, length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);

    free(byte_array);
}

void send_terminate_message(int target) {
    int terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void receive_terminate_message(int source) {
    MPI_Status status;
    int terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
}


void master(int max_rank, string transfer_learning_version, int32_t seed_stirs) {
    //the "main" id will have already been set by the main function so we do not need to re-set it here
    Log::debug("MAX INT: %d\n", numeric_limits<int>::max());
    int terminates_sent = 0;
    int generated_genome = 0;
    int evaluated_genome = 0;
    Log::debug("Master: total number of genomes need to be generated is %d\n", generation_genomes*number_islands);
    while (true) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);
        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);
            if (generated_genome < generation_genomes * number_islands) {
                examm_mutex.lock();
                RNN_Genome *genome = examm->generate_genome(seed_stirs);
                examm_mutex.unlock();

                if (genome != NULL) {
                    Log::debug("sending genome to: %d\n", source);
                    send_genome_to(source, genome);

                    //delete this genome as it will not be used again
                    delete genome;
                    generated_genome ++;
                } else {
                    Log::fatal("Returned NULL genome from generate genome function, this should never happen!\n");
                    exit(1);
                }
            } else {
                Log::info("terminating worker: %d\n", source);
                send_terminate_message(source);
                terminates_sent++;

                Log::info("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) {
                    Log::debug("Ending genome, generated genome is %d, evaluated genome is %d\n", generated_genome, evaluated_genome);
                    return;
                }
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome from: %d\n", source);
            RNN_Genome *genome = receive_genome_from(source);

            examm_mutex.lock();
            examm->insert_genome(genome);
            examm_mutex.unlock();

            //delete the genome as it won't be used again, a copy was inserted
            delete genome;
            evaluated_genome++;

        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(int rank, bool smooth_data, OnlineSeries* online_series) {
    Log::set_id("worker_" + to_string(rank));

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome!\n");
            RNN_Genome* genome = receive_genome_from(0);
            vector< vector< vector<double> > > current_training_inputs;
            vector< vector< vector<double> > > current_training_outputs;
            vector< vector< vector<double> > > current_validation_inputs;
            vector< vector< vector<double> > > current_validation_outputs;

            vector<int32_t> train_index = online_series->get_training_index();
            vector<int32_t> validation_index = online_series->get_validation_index();

            for ( int i = 0; i < train_index.size(); i++) {
                current_training_inputs.push_back(training_inputs[train_index[i]]);
                current_training_outputs.push_back(training_outputs[train_index[i]]);
            }
            for (int i = 0; i < validation_index.size(); i++) {
                current_validation_inputs.push_back(training_inputs[validation_index[i]]);
                current_validation_outputs.push_back(training_outputs[validation_index[i]]);
            }

            //have each worker write the backproagation to a separate log file
            string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);
            genome->backpropagate_stochastic(current_training_inputs, current_training_outputs, current_validation_inputs, current_validation_outputs, random_sequence_length, sequence_length_lower_bound, sequence_length_upper_bound, noise_std);
            // genome->set_genome_type(GENERATED);
            genome->evaluate_online(current_validation_inputs, current_validation_outputs);
            Log::release_id(log_id);

            //go back to the worker's log for MPI communication
            Log::set_id("worker_" + to_string(rank));

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    //release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank));
}

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
    
    // training data will be smoothed
    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs); 
    int32_t num_sets = training_inputs.size();
    // validation data is original without smooth
    OnlineSeries* online_series = new OnlineSeries(num_sets, arguments);
     
    // time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);
    bool smooth_data = argument_exists(arguments, "--data_smooth_method");

    int number_inputs = time_series_sets->get_number_inputs();
    int number_outputs = time_series_sets->get_number_outputs();

    Log::debug("number_inputs: %d, number_outputs: %d\n", number_inputs, number_outputs);

    get_argument(arguments, "--number_islands", true, number_islands);

    get_argument(arguments, "--generation_genomes", true, generation_genomes);

    int32_t num_generations;
    get_argument(arguments, "--num_generations", true, num_generations);
    // validate_generation_number(num_generations, training_inputs.size());

    int32_t elite_population_size;
    get_argument(arguments, "--elite_population_size", true, elite_population_size);

    string speciation_method = "";
    get_argument(arguments, "--speciation_method", false, speciation_method);

    int32_t extinction_event_generation_number = 0;
    get_argument(arguments, "--extinction_event_generation_number", false, extinction_event_generation_number);

    int32_t islands_to_exterminate = 0;
    get_argument(arguments, "--islands_to_exterminate", false, islands_to_exterminate);

    string island_ranking_method = "";
    get_argument(arguments, "--island_ranking_method", false, island_ranking_method);

    string repopulation_method = "";
    get_argument(arguments, "--repopulation_method", false, repopulation_method);

    int32_t repopulation_mutations = 0;
    get_argument(arguments, "--repopulation_mutations", false, repopulation_mutations);

    int32_t epochs_acc_freq = 0;
    get_argument(arguments, "--epochs_acc_freq", false, epochs_acc_freq);

    int32_t bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    int32_t time_series_length;
    get_argument(arguments, "--time_series_length", true, time_series_length);

    double learning_rate = 0.001;
    get_argument(arguments, "--learning_rate", false, learning_rate);

    double high_threshold = 1.0;
    bool use_high_threshold = get_argument(arguments, "--high_threshold", false, high_threshold);

    double low_threshold = 0.05;
    bool use_low_threshold = get_argument(arguments, "--low_threshold", false, low_threshold);

    double dropout_probability = 0.0;
    bool use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);

    get_argument(arguments, "--noise_std", false, noise_std);

    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    vector<string> possible_node_types;
    get_argument_vector(arguments, "--possible_node_types", false, possible_node_types);

    int32_t min_recurrent_depth = 1;
    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);

    int32_t max_recurrent_depth = 10;
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);

    //bool use_regression = argument_exists(arguments, "--use_regression");
    bool use_regression = true; //time series will always use regression

    random_sequence_length = argument_exists(arguments, "--random_sequence_length");
    get_argument(arguments, "--sequence_length_lower_bound", false, sequence_length_lower_bound);
    get_argument(arguments, "--sequence_length_upper_bound", false, sequence_length_upper_bound);

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

    RNN_Genome *seed_genome = NULL;
    string genome_file_name = "";
    string transfer_learning_version = "";
    if (get_argument(arguments, "--genome_bin", false, genome_file_name)) {
        seed_genome = new RNN_Genome(genome_file_name);
        seed_genome->set_normalize_bounds(time_series_sets->get_normalize_type(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), time_series_sets->get_normalize_avgs(), time_series_sets->get_normalize_std_devs());

        get_argument(arguments, "--transfer_learning_version", true, transfer_learning_version);

        bool epigenetic_weights = argument_exists(arguments, "--epigenetic_weights");

        seed_genome->transfer_to(time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), transfer_learning_version, epigenetic_weights, min_recurrent_depth, max_recurrent_depth);
        seed_genome->tl_with_epigenetic = epigenetic_weights ;
    }

    int32_t seed_stirs = 0;
    get_argument(arguments, "--seed_stirs", false, seed_stirs);

    bool start_filled = false;
    get_argument(arguments, "--start_filled", false, start_filled);

    Log::clear_rank_restriction();

    // Log::error("validation size is %d, num training example is %d\n", validation_size, num_training_sets);
    if (rank == 0) {
        examm = new EXAMM(generation_genomes, number_islands, generation_genomes, elite_population_size, extinction_event_generation_number, islands_to_exterminate, island_ranking_method,
            repopulation_method, repopulation_mutations, false, epochs_acc_freq,
            speciation_method,
            0, 0, 0, 0, 0,
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

        if (possible_node_types.size() > 0)  {
            examm->set_possible_node_types(possible_node_types);
        }
    }

    // time_series_index.push_back(current_time_index);
    for (int current_generation = 0; current_generation < num_generations; current_generation ++) {
        online_series->set_current_index(current_generation);
        if (rank ==0) {
            master(max_rank, transfer_learning_version, seed_stirs);           
        } else {
            worker(rank, smooth_data, online_series);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            vector <int32_t> validation_index = online_series->get_validation_index();
            int32_t test_index = online_series->get_test_index();

            vector< vector< vector<double> > > current_test_inputs;
            vector< vector< vector<double> > > current_test_outputs;
            vector< vector< vector<double> > > current_validation_inputs;
            vector< vector< vector<double> > > current_validation_outputs;

            current_test_inputs.push_back(training_inputs[test_index]);
            current_test_outputs.push_back(training_outputs[test_index]);

            for (int i = 0; i < validation_index.size(); i++) {
                current_validation_inputs.push_back(training_inputs[validation_index[i]]);
                current_validation_outputs.push_back(training_outputs[validation_index[i]]);
            }

            string filename = output_directory + "/generation_" + std::to_string(current_generation);
            examm->finalize_generation(filename, current_validation_inputs, current_validation_outputs, current_test_inputs, current_test_outputs);
            // best_genome->write_predictions(output_directory, "generation_"  std::to_string(current_generation), test_input, test_output, time_series_sets );
            examm->update_log();

        }
        if (rank == 0) Log::error("generation %d finished\n", current_generation);
        
    }

    Log::set_id("main_" + to_string(rank));

    finished = true;

    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));

    MPI_Finalize();
    delete time_series_sets;

    return 0;
}
