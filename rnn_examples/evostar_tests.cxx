#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <iomanip>
using std::setw;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

//for mkdir
#include <sys/stat.h>

#include "mpi.h"


#include "common/arguments.hxx"

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

string process_name;

vector<string> rnn_types({"one_layer_ff", "two_layer_ff", "jordan", "elman", "one_layer_lstm", "two_layer_lstm" });

vector<string> input_parameter_names({
        "Cyclone-Conditioner_Inlet_Temp" ,"Cyclone-Conditioner_Outlet_Temp",
        "Cyclone-Lignite_Feeder_Rate","Cyclone-Primary_Air_Flow","Cyclone-Primary_Air_Split",
        "Cyclone-Secondary_Air_Ratio","Cyclone-System_Cyc_Secondary_Air_Flow_Total",
        "Cyclone-Secondary_Air_Flow","Cyclone-Secondary_Air_Split",
        "Cyclone-Tertiary_Air_Flow","Cyclone-Tertiary_Air_Split",
        "Cyclone-Total_Comb_Air_Flow","Cyclone-Main_Oil_Flow",
        "Cyclone-Main_Flm_Int"
        });

vector<string> output_parameter_names({"Cyclone-Main_Flm_Int"});
//vector<string> output_parameter_names({"Cyclone-Main_Oil_Flow"});

vector<TimeSeriesSet*> input_series;


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

    cout << "[" << setw(10) << process_name << "] sending job " << current_job << " of " << results.size() << " to: " << target << endl;
    MPI_Send(job_message, 1, MPI_INT, target, JOB_TAG, MPI_COMM_WORLD);
}


int receive_job_from(int source) {
    MPI_Status status;
    int job_message[1];
    MPI_Recv(job_message, 1, MPI_INT, source, JOB_TAG, MPI_COMM_WORLD, &status);

    int current_job = job_message[0];

    cout << "[" << setw(10) << process_name << "] receiving current_job: " << current_job << " from: " << source << endl;

    return current_job;
}

string result_to_string(ResultSet result) {
    return "[result, job: " + to_string(result.job) + ", training mae: " + to_string(result.training_mae) + ", training mse: " + to_string(result.training_mse) + ", test mae: " + to_string(result.test_mae) + ", test mse: " + to_string(result.test_mae) + ", millis: " + to_string(result.milliseconds) + "]";
}

void send_result_to(int target, ResultSet result) {
    cout << "sizeof(ResultSet): " << sizeof(ResultSet) << endl;
    cout << "sizeof(result): " << sizeof(result) << endl;

    cout << "sending: " << result_to_string(result) << endl;

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

    cout << "received: " << result_to_string(result) << endl;

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


void master(int max_rank) {
    process_name = "master";

    if (output_directory != "") {
        cout << "[" << setw(10) << process_name << "] creating directory: " << output_directory << endl;
        mkdir(output_directory.c_str(), 0777);
    }

    results = vector<ResultSet>(rnn_types.size() * input_series.size() * repeats);

    cout << "MAX INT: " << numeric_limits<int>::max() << endl;

    int terminates_sent = 0;
    int current_job = 0;

    while (true) {
        //wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int message_source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        cout << "[" << setw(10) << process_name << "] probe returned message from: " << message_source << " with tag: " << tag << endl;

        //if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request_from(message_source);

            if (current_job >= results.size()) {
                //no more jobs to process if the current job is >= the result vector
                //send terminate message
                cout << "[" << setw(10) << process_name << "] terminating worker: " << message_source << endl;
                send_terminate_to(message_source);
                terminates_sent++;

                cout << "[" << setw(10) << process_name << "] sent: " << terminates_sent << " terminates of: " << (max_rank - 1) << endl;
                if (terminates_sent >= max_rank - 1) return;

            } else {
                //send genome
                cout << "[" << setw(10) << process_name << "] sending genome to: " << message_source << endl;
                send_job_to(message_source, current_job);

                //increment the current job for the next worker
                current_job++;
            }
        } else if (tag == RESULT_TAG) {
            cout << "[" << setw(10) << process_name << "] receiving job from: " << message_source << endl;
            ResultSet result = receive_result_from(message_source);
            results[result.job] = result;

        } else {
            cerr << "[" << setw(10) << process_name << "] ERROR: received message with unknown tag: " << tag << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

ResultSet handle_job(int current_job) {
    int32_t jobs_per_rnn = input_series.size() * repeats;

    //get rnn_type
    string rnn_type = rnn_types[ current_job / jobs_per_rnn ] ;
    //get j, k
    int32_t jobs_per_j = repeats;
    int32_t j = (current_job % jobs_per_rnn) / jobs_per_j;

    //get repeat
    int32_t repeat = current_job % jobs_per_j;

    cout << "[" << setw(10) << process_name << "] evaluating rnn type '" << rnn_type << "' with j: " << j << ", repeat: " << repeat << endl;

    vector<TimeSeriesSet*> training_series;
    vector<TimeSeriesSet*> validation_series;

    for (uint32_t k = 0; k < input_series.size(); k++) {
        if (j == k) {
            validation_series.push_back(input_series[k]);
        } else {
            training_series.push_back(input_series[k]);
        }
    }

    vector< vector< vector<double> > > training_inputs;
    vector< vector< vector<double> > > training_outputs;
    vector< vector< vector<double> > > validation_inputs;
    vector< vector< vector<double> > > validation_outputs;

    export_time_series(training_series, input_parameter_names, output_parameter_names, time_offset, training_inputs, training_outputs);
    export_time_series(validation_series, input_parameter_names, output_parameter_names, time_offset, validation_inputs, validation_outputs);

    int number_inputs = input_parameter_names.size();
    int number_outputs = output_parameter_names.size();

    RNN_Genome *genome = NULL;
    if (rnn_type == "one_layer_lstm") {
        genome = create_lstm(number_inputs, 1, number_inputs, number_outputs, 1);

    } else if (rnn_type == "two_layer_lstm") {
        genome = create_lstm(number_inputs, 2, number_inputs, number_outputs, 1);

    } else if (rnn_type == "one_layer_ff") {
        genome = create_ff(number_inputs, 1, number_inputs, number_outputs, 1);

    } else if (rnn_type == "two_layer_ff") {
        genome = create_ff(number_inputs, 2, number_inputs, number_outputs, 1);

    } else if (rnn_type == "jordan") {
        genome = create_jordan(number_inputs, 1, number_inputs, number_outputs, 1);

    } else if (rnn_type == "elman") {
        genome = create_elman(number_inputs, 1, number_inputs, number_outputs, 1);
    }

    RNN* rnn = genome->get_rnn();

    uint32_t number_of_weights = genome->get_number_weights();

    vector<double> min_bound(number_of_weights, -1.0); 
    vector<double> max_bound(number_of_weights, 1.0); 

    vector<double> best_parameters;

    genome->initialize_randomly();
    genome->set_bp_iterations(bp_iterations);

    string log_filename = output_directory + "/" + rnn_type + "_" + to_string(j) + "_" + to_string(repeat) + ".txt";
    genome->set_log_filename(log_filename);


    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();

    long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "backpropagation took: " << milliseconds << " ms." << endl;

    genome->get_weights(best_parameters);
    rnn->set_weights(best_parameters);
    double training_mse = genome->get_mse(best_parameters, training_inputs, training_outputs, true);
    double training_mae = genome->get_mae(best_parameters, training_inputs, training_outputs, true);

    double test_mse = genome->get_mse(best_parameters, validation_inputs, validation_outputs, true);
    double test_mae = genome->get_mae(best_parameters, validation_inputs, validation_outputs, true);

    ResultSet result;
    result.job = current_job;
    result.training_mse = training_mse;
    result.training_mae = training_mae;
    result.test_mse = test_mse;
    result.test_mae = test_mae;
    result.milliseconds = milliseconds;

    cout << "[" << setw(10) << process_name << "] finished job, result: " << result_to_string(result) << endl;

    return result;
}

void worker(int rank) {
    int master_rank = 0;
    process_name = "worker_" + to_string(rank);

    while (true) {
        cout << "[" << setw(10) << process_name << "] sending work request!" << endl;
        send_work_request_to(master_rank);
        cout << "[" << setw(10) << process_name << "] sent work request!" << endl;

        MPI_Status status;
        MPI_Probe(master_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;

        cout << "[" << setw(10) << process_name << "] probe received message with tag: " << tag << endl;

        if (tag == TERMINATE_TAG) {
            cout << "[" << setw(10) << process_name << "] received terminate tag!" << endl;
            receive_terminate_from(master_rank);
            break;

        } else if (tag == JOB_TAG) {
            cout << "[" << setw(10) << process_name << "] received genome!" << endl;
            int current_job = receive_job_from(master_rank);

            ResultSet result = handle_job(current_job);

            cout << "[" << setw(10) << process_name << "] calculated_result: " << result_to_string(result) << endl;

            send_result_to(master_rank, result);

        } else {
            cerr << "[" << setw(10) << process_name << "] ERROR: received message with unknown tag: " << tag << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}



int main(int argc, char **argv) {
    int rank, max_rank;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    vector<string> arguments = vector<string>(argv, argv + argc);

    vector<string> input_filenames;
    get_argument_vector(arguments, "--input_files", true, input_filenames);

    get_argument(arguments, "--time_offset", true, time_offset);

    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    get_argument(arguments, "--output_directory", true, output_directory);

    get_argument(arguments, "--repeats", true, repeats);

    for (int i = 0; i < input_filenames.size(); i++) {
        input_series.push_back(new TimeSeriesSet(input_filenames[i]));
    }
    if (rank == 0) {
        normalize_time_series_sets(input_series, true);
    } else {
        normalize_time_series_sets(input_series, false);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        master(max_rank);

        int32_t current = 0;
        for (int i = 0; i < rnn_types.size(); i++) {
            for (int j = 0; j < input_series.size(); j++) {
                for (int k = 0; k < repeats; k++) {
                    cout << rnn_types[i] << ", tested on series[" << j << "], repeat: " << k << ", result: " << result_to_string(results[current]) << endl;
                    current++;
                }
            }
        }

    } else {
        worker(rank);
    }

    MPI_Finalize();
}
