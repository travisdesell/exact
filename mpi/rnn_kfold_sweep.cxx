#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <iomanip>
using std::setw;

#include <iostream>
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <cstring>
using std::memcpy;
using std::strchr;

#include <vector>
using std::vector;

//for mkdir
#include <sys/stat.h>
#include <errno.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif /* HAVE_UNISTD_H */

typedef struct stat Stat;

#include "mpi.h"


#include "common/arguments.hxx"
#include "common/log.hxx"
#include "common/weight_initialize.hxx"

#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "rnn/generate_nn.hxx"

#include "time_series/time_series.hxx"

#define WORK_REQUEST_TAG 1
#define JOB_TAG 2
#define TERMINATE_TAG 3
#define RESULT_TAG 4

int32_t time_offset = 1;
int bp_iterations;
string output_directory;
int32_t repeats = 5;
int fold_size = 2;

string weight_initialize_string = "random";
WeightType weight_initialize = get_enum_from_string(weight_initialize_string);

string process_name;

vector<string> rnn_types({
        "one_layer_ff", "two_layer_ff",
        "jordan",
        "elman",
        "one_layer_mgu", "two_layer_mgu",
        "one_layer_gru", "two_layer_gru",
        "one_layer_ugrnn", "two_layer_ugrnn",
        "one_layer_delta", "two_layer_delta",
        "one_layer_lstm", "two_layer_lstm"
    });

TimeSeriesSets* time_series_sets = NULL;


struct ResultSet {
    int job;
    double training_mae;
    double training_mse;
    double test_mae;
    double test_mse;
    long milliseconds;
};

vector<ResultSet> results;



void send_work_request_to(int target) {
    int work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}

void receive_work_request_from(int source) {
    MPI_Status status;
    int work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

void send_job_to(int target, int current_job) {
    int job_message[1];
    job_message[0] = current_job;

    Log::debug("sending job %d of %d to %d\n", current_job, results.size(), target);
    MPI_Send(job_message, 1, MPI_INT, target, JOB_TAG, MPI_COMM_WORLD);
}


int receive_job_from(int source) {
    MPI_Status status;
    int job_message[1];
    MPI_Recv(job_message, 1, MPI_INT, source, JOB_TAG, MPI_COMM_WORLD, &status);

    int current_job = job_message[0];

    Log::debug("receiving current_job: %d from %d\n", current_job, source);

    return current_job;
}

string result_to_string(ResultSet result) {
    return "[result, job: " + to_string(result.job) + ", training mae: " + to_string(result.training_mae) + ", training mse: " + to_string(result.training_mse) + ", test mae: " + to_string(result.test_mae) + ", test mse: " + to_string(result.test_mae) + ", millis: " + to_string(result.milliseconds) + "]";
}

void send_result_to(int target, ResultSet result) {
    size_t result_size = sizeof(result);
    char bytes[result_size];
    memcpy(bytes, &result, result_size);

    MPI_Send(bytes, result_size, MPI_CHAR, target, RESULT_TAG, MPI_COMM_WORLD);
}

ResultSet receive_result_from(int source) {
    ResultSet result;

    MPI_Status status;
    size_t result_size = sizeof(result);
    char bytes[result_size];

    MPI_Recv(bytes, result_size, MPI_CHAR, source, RESULT_TAG, MPI_COMM_WORLD, &status);

    memcpy(&result, bytes, result_size);

    return result;
}

void send_terminate_to(int target) {
    int terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void receive_terminate_from(int source) {
    MPI_Status status;
    int terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
}


//tweaked from: https://stackoverflow.com/questions/675039/how-can-i-create-directory-tree-in-c-linux/29828907
static int do_mkdir(const char *path, mode_t mode) {
    Stat            st;
    int             status = 0;

    if (stat(path, &st) != 0) {
        /* Directory does not exist. EEXIST for race condition */
        if (mkdir(path, mode) != 0 && errno != EEXIST) {
            status = -1;
        }

    } else if (!S_ISDIR(st.st_mode)) {
        errno = ENOTDIR;
        status = -1;
    }

    return(status);
}

/**
 * ** mkpath - ensure all directories in path exist
 * ** Algorithm takes the pessimistic view and works top-down to ensure
 * ** each directory in path exists, rather than optimistically creating
 * ** the last element and working backwards.
 * */
int mkpath(const char *path, mode_t mode) {
    char           *pp;
    char           *sp;
    int             status;
    char           *copypath = strdup(path);

    status = 0;
    pp = copypath;
    while (status == 0 && (sp = strchr(pp, '/')) != 0) {
        Log::debug("trying to create directory: '%s'\n", copypath);
        if (sp != pp) {
            /* Neither root nor double slash in path */
            *sp = '\0';
            status = do_mkdir(copypath, mode);
            *sp = '/';
        }
        pp = sp + 1;
    }

    if (status == 0) {
        status = do_mkdir(path, mode);
    }

    free(copypath);
    return (status);
}




void master(int max_rank) {
    if (output_directory != "") {
        Log::debug("creating directory: '%s'\n", output_directory.c_str());
        mkpath(output_directory.c_str(), 0777);

        mkdir(output_directory.c_str(), 0777);
    }

    //initialize the results with -1 as the job so we can determine if a particular rnn type has completed
    results = vector<ResultSet>(rnn_types.size() * time_series_sets->get_number_series() * repeats, {-1, 0.0, 0.0, 0.0, 0.0, 0});

    int terminates_sent = 0;
    int current_job = 0;
    int last_job = rnn_types.size() * (time_series_sets->get_number_series() / fold_size) * repeats;

    while (true) {
        //wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int message_source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", message_source, tag);

        //if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request_from(message_source);

            if (current_job >= last_job) {
                //no more jobs to process if the current job is >= the result vector
                //send terminate message
                Log::debug("terminating worker: %d\n", message_source);
                send_terminate_to(message_source);
                terminates_sent++;

                Log::debug("sent: %d terminates of: %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) return;

            } else {
                //send job
                Log::debug("sending job to: %d\n", message_source);
                send_job_to(message_source, current_job);

                //increment the current job for the next worker
                current_job++;
            }
        } else if (tag == RESULT_TAG) {
            Log::debug("receiving job from: %d\n", message_source);
            ResultSet result = receive_result_from(message_source);
            results[result.job] = result;

            //TODO:
            //check and see if this particular set of jobs for rnn_type has completed,
            //then write the file for that type if it has
            int32_t jobs_per_rnn = (time_series_sets->get_number_series() / fold_size) * repeats;

            //get the particular rnn type this job was for, and which results should be there
            int32_t rnn = result.job / jobs_per_rnn;
            int32_t rnn_job_start = rnn * jobs_per_rnn;
            int32_t rnn_job_end = (rnn + 1) * jobs_per_rnn;

            bool rnn_finished = true;
            Log::debug("testing finished for rnn: '%s'\n", rnn_types[rnn].c_str());
            for (int i = rnn_job_start; i < rnn_job_end; i++) {
                if (i == rnn_job_start) {
                    Log::debug(" %d", results[i].job);
                } else {
                    Log::debug_no_header(" %d", results[i].job);
                }

                if (results[i].job < 0) {
                    rnn_finished = false;
                    break;
                }
            }
            Log::debug_no_header("\n");

            Log::debug("rnn '%s' finished? %d\n", rnn_types[rnn].c_str(), rnn_finished);

            if (rnn_finished) {
                ofstream outfile(output_directory + "/combined_" + rnn_types[rnn] + ".csv");

                int32_t current = rnn_job_start;
                for (int32_t j = 0; j < (time_series_sets->get_number_series() / fold_size); j++) {
                    for (int32_t k = 0; k < repeats; k++) {

                        outfile << j << "," << k << "," << results[current].milliseconds << "," << results[current].training_mse << "," << results[current].training_mae << "," << results[current].test_mse << "," << results[current].test_mae << endl;

                        Log::debug("%s, tested on series[%d], repeat: %d, result: %s\n", rnn_types[rnn].c_str(), j, k, result_to_string(results[current]).c_str());
                        current++;
                    }
                }
                outfile.close();
            }


        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d\n", message_source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

ResultSet handle_job(int rank, int current_job) {
    int32_t jobs_per_rnn = (time_series_sets->get_number_series() / fold_size) * repeats;

    //get rnn_type
    string rnn_type = rnn_types[ current_job / jobs_per_rnn ] ;
    //get j, k
    int32_t jobs_per_j = repeats;
    int32_t j = (current_job % jobs_per_rnn) / jobs_per_j;

    //get repeat
    int32_t repeat = current_job % jobs_per_j;

    Log::debug("evaluating rnn type '%s' with j: %d, repeat: %d\n", rnn_type.c_str(), j, repeat);

    vector<int> training_indexes;
    vector<int> test_indexes;

    for (uint32_t k = 0; k < time_series_sets->get_number_series(); k += fold_size) {
        if (j == (k / fold_size)) {
            for (int l = 0; l < fold_size; l++) {
                test_indexes.push_back(k + l);
            }
        } else {
            for (int l = 0; l < fold_size; l++) {
                training_indexes.push_back(k + l);
            }
        }
    }

    Log::debug("test_indexes.size(): %d, training_indexes.size(): %d\n", test_indexes.size(), training_indexes.size());

    time_series_sets->set_training_indexes(training_indexes);
    time_series_sets->set_test_indexes(test_indexes);

    vector< vector< vector<double> > > training_inputs;
    vector< vector< vector<double> > > training_outputs;
    vector< vector< vector<double> > > validation_inputs;
    vector< vector< vector<double> > > validation_outputs;

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    vector<string> input_parameter_names = time_series_sets->get_input_parameter_names();
    vector<string> output_parameter_names = time_series_sets->get_output_parameter_names();

    int number_inputs = time_series_sets->get_number_inputs();
    //int number_outputs = time_series_sets->get_number_outputs();

    RNN_Genome *genome = NULL;
    if (rnn_type == "one_layer_lstm") {
        genome = create_lstm(input_parameter_names, 1, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "two_layer_lstm") {
        genome = create_lstm(input_parameter_names, 2, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "one_layer_delta") {
        genome = create_delta(input_parameter_names, 1, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "two_layer_delta") {
        genome = create_delta(input_parameter_names, 2, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "one_layer_gru") {
        genome = create_gru(input_parameter_names, 1, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "two_layer_gru") {
        genome = create_gru(input_parameter_names, 2, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "one_layer_mgu") {
        genome = create_mgu(input_parameter_names, 1, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "two_layer_mgu") {
        genome = create_mgu(input_parameter_names, 2, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "one_layer_delta") {
        genome = create_delta(input_parameter_names, 1, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "two_layer_delta") {
        genome = create_delta(input_parameter_names, 2, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "one_layer_ugrnn") {
        genome = create_ugrnn(input_parameter_names, 1, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "two_layer_ugrnn") {
        genome = create_ugrnn(input_parameter_names, 2, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "one_layer_ff") {
        genome = create_ff(input_parameter_names, 1, number_inputs, output_parameter_names, 0, weight_initialize, WeightType::NONE, WeightType::NONE);

    } else if (rnn_type == "two_layer_ff") {
        genome = create_ff(input_parameter_names, 2, number_inputs, output_parameter_names, 0, weight_initialize, WeightType::NONE, WeightType::NONE);

    } else if (rnn_type == "jordan") {
        genome = create_jordan(input_parameter_names, 1, number_inputs, output_parameter_names, 1, weight_initialize);

    } else if (rnn_type == "elman") {
        genome = create_elman(input_parameter_names, 1, number_inputs, output_parameter_names, 1, weight_initialize);
    }

    RNN* rnn = genome->get_rnn();

    uint32_t number_of_weights = genome->get_number_weights();
    Log::debug("RNN INFO FOR '%s', nodes: %d, edges: %d, rec: %d, weights: %d\n", rnn_type.c_str(), genome->get_enabled_node_count(), genome->get_enabled_edge_count(), genome->get_enabled_recurrent_edge_count(), number_of_weights);

    vector<double> min_bound(number_of_weights, -1.0); 
    vector<double> max_bound(number_of_weights, 1.0); 

    vector<double> best_parameters;

    genome->initialize_randomly();
    genome->set_bp_iterations(bp_iterations, 0);

    string first_directory = output_directory + "/" + rnn_type;
    mkdir(first_directory.c_str(), 0777);

    string second_directory = first_directory + "/slice_" + to_string(j);
    mkdir(second_directory.c_str(), 0777);

    string log_filename = second_directory + "/repeat_" + to_string(repeat) + ".txt";
    genome->set_log_filename(log_filename);

    string backprop_log_id = rnn_type + "_slice_" + to_string(j) + "_repeat_" + to_string(repeat);
    Log::set_id(backprop_log_id);

    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs, false, 30, 100);
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

    long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    genome->get_weights(best_parameters);
    rnn->set_weights(best_parameters);

    double training_mse = genome->get_mse(best_parameters, training_inputs, training_outputs);
    double training_mae = genome->get_mae(best_parameters, training_inputs, training_outputs);

    double test_mse = genome->get_mse(best_parameters, validation_inputs, validation_outputs);
    double test_mae = genome->get_mae(best_parameters, validation_inputs, validation_outputs);

    Log::release_id(backprop_log_id);
    Log::set_id("worker_" + to_string(rank));

    Log::debug("deleting genome and rnn.\n");

    delete genome;
    delete rnn;

    ResultSet result;
    result.job = current_job;
    result.training_mse = training_mse;
    result.training_mae = training_mae;
    result.test_mse = test_mse;
    result.test_mae = test_mae;
    result.milliseconds = milliseconds;

    Log::debug("finished job, result: %s\n", result_to_string(result).c_str());

    return result;
}

void worker(int rank) {
    int master_rank = 0;
    Log::set_id("worker_" + to_string(rank));

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request_to(master_rank);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(master_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_from(master_rank);
            break;

        } else if (tag == JOB_TAG) {
            Log::debug("received genome!\n");
            int current_job = receive_job_from(master_rank);

            ResultSet result = handle_job(rank, current_job);

            Log::debug("calculated_result: %s\n", result_to_string(result).c_str());

            send_result_to(master_rank, result);

        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    Log::release_id("worker_" + to_string(rank));
}



int main(int argc, char **argv) {
    int rank, max_rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    vector<string> arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));

    Log::debug("process %d of %d\n", rank, max_rank);
    Log::restrict_to_rank(0);


    get_argument(arguments, "--time_offset", true, time_offset);

    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    get_argument(arguments, "--output_directory", true, output_directory);

    get_argument(arguments, "--repeats", true, repeats);

    get_argument(arguments, "--fold_size", true, fold_size);

    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);

    weight_initialize = get_enum_from_string(weight_initialize_string);

    if (weight_initialize < 0 || weight_initialize >= NUM_WEIGHT_TYPES - 1) {
        Log::fatal("weight initialization method %s is set wrong \n", weight_initialize_string.c_str());
    }


    if (rank == 0) {
        //only print verbose info from the master process
        time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    } else {
        time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    }

    //MPI_Barrier(MPI_COMM_WORLD);

    Log::clear_rank_restriction();

    if (rank == 0) {
        master(max_rank);
    } else {
        worker(rank);
    }

    Log::release_id("main_" + to_string(rank));
    MPI_Finalize();
}
