#include <chrono>

#include <iomanip>
using std::setw;
using std::fixed;
using std::setprecision;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

//for mkdir
#include <sys/stat.h>


#include "mpi.h"

#include "common/arguments.hxx"

#include "rnn/exalt.hxx"

#include "time_series/time_series.hxx"

#define WORK_REQUEST_TAG 1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG 3
#define TERMINATE_TAG 4

mutex exalt_mutex;

vector<string> arguments;

EXALT *exalt;

vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

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

RNN_Genome* receive_genome_from(string name, int source) {
    MPI_Status status;
    int length_message[1];
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);

    int length = length_message[0];

    cout << "[" << setw(10) << name << "] receiving genome of length: " << length << " from: " << source << endl;

    char* genome_str = new char[length + 1];

    cout << "[" << setw(10) << name << "] receiving genome from: " << source << endl;
    MPI_Recv(genome_str, length, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);

    genome_str[length] = '\0';

    //cout << "genome_str:" << endl << genome_str << endl;

    RNN_Genome* genome = new RNN_Genome(genome_str, length, false);

    delete [] genome_str;
    return genome;
}

void send_genome_to(string name, int target, RNN_Genome* genome) {
    char *byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    cout << "[" << setw(10) << name << "] sending genome of length: " << length << " to: " << target << endl;

    int length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    cout << "[" << setw(10) << name << "] sending genome to: " << target << endl;
    MPI_Send(byte_array, length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);
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

void master(int max_rank) {
    string name = "master";

    int terminates_sent = 0;

    while (true) {
        //wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        cout << "[" << setw(10) << name << "] probe returned message from: " << source << " with tag: " << tag << endl;


        //if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);

            exalt_mutex.lock();
            RNN_Genome *genome = exalt->generate_genome();
            exalt_mutex.unlock();

            if (genome == NULL) { //search was completed if it returns NULL for an individual
                //send terminate message
                cout << "[" << setw(10) << name << "] terminating worker: " << source << endl;
                send_terminate_message(source);
                terminates_sent++;

                cout << "[" << setw(10) << name << "] sent: " << terminates_sent << " terminates of: " << (max_rank - 1) << endl;
                if (terminates_sent >= max_rank - 1) return;

            } else {
                //genome->write_to_file( exalt->get_output_directory() + "/before_send_gen_" + to_string(genome->get_generation_id()) );

                //send genome
                cout << "[" << setw(10) << name << "] sending genome to: " << source << endl;
                send_genome_to(name, source, genome);

                //delete this genome as it will not be used again
                delete genome;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            cout << "[" << setw(10) << name << "] received genome from: " << source << endl;
            RNN_Genome *genome = receive_genome_from(name, source);

            exalt_mutex.lock();
            exalt->insert_genome(genome);
            exalt_mutex.unlock();

            //this genome will be deleted if/when removed from population
        } else {
            cerr << "[" << setw(10) << name << "] ERROR: received message with unknown tag: " << tag << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(int rank) {
    string name = "worker_" + to_string(rank);

    while (true) {
        cout << "[" << setw(10) << name << "] sending work request!" << endl;
        send_work_request(0);
        cout << "[" << setw(10) << name << "] sent work request!" << endl;

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int tag = status.MPI_TAG;

        cout << "[" << setw(10) << name << "] probe received message with tag: " << tag << endl;

        if (tag == TERMINATE_TAG) {
            cout << "[" << setw(10) << name << "] received terminate tag!" << endl;
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            cout << "[" << setw(10) << name << "] received genome!" << endl;
            RNN_Genome* genome = receive_genome_from(name, 0);

            genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);

            send_genome_to(name, 0, genome);

            delete genome;
        } else {
            cerr << "[" << setw(10) << name << "] ERROR: received message with unknown tag: " << tag << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    arguments = vector<string>(argv, argv + argc);

    vector<string> input_filenames;
    get_argument_vector(arguments, "--input_filenames", true, input_filenames);

    vector<TimeSeriesSet*> input_series;
    for (int i = 0; i < input_filenames.size(); i++) {
        input_series.push_back(new TimeSeriesSet(input_filenames[i]));
    }

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    bool normalize = argument_exists(arguments, "--normalize");
    if (normalize) {
        if (rank == 0) {
            normalize_time_series_sets(input_series, true);
            //write_time_series_sets(input_series, "./series_");
        } else {
            normalize_time_series_sets(input_series, false);
        }
    }

    vector<string> input_parameter_names;
    get_argument_vector(arguments, "--input_parameter_names", true, input_parameter_names);

    vector<string> output_parameter_names;
    get_argument_vector(arguments, "--output_parameter_names", true, output_parameter_names);

    int32_t population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int32_t number_islands;
    get_argument(arguments, "--number_islands", true, number_islands);

    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

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

    int32_t repeats;
    get_argument(arguments, "--repeats", true, repeats);


    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    mkdir(output_directory.c_str(), 0777);

    uint32_t i = 0;
    bool first = true;
    /*
    if (output_parameter_names[0].compare("Pitch") == 0) {
        i = 7;
    }
    */

    for (; i < input_series.size(); i++) {
        vector<TimeSeriesSet*> training_series;
        vector<TimeSeriesSet*> validation_series;

        for (uint32_t j = 0; j < input_series.size(); j++) {
            if (j == i) {
                validation_series.push_back(input_series[j]);
            } else {
                training_series.push_back(input_series[j]);
            }
        }

        export_time_series(training_series, input_parameter_names, output_parameter_names, time_offset, training_inputs, training_outputs);
        export_time_series(validation_series, input_parameter_names, output_parameter_names, time_offset, validation_inputs, validation_outputs);

        string slice_output_directory = output_directory + "/slice_" + to_string(i);
        mkdir(slice_output_directory.c_str(), 0777);
        ofstream slice_times_file(output_directory + "/slice_" + to_string(i) + "_runtimes.csv");

        int k = 0;
        /*
        if (output_parameter_names[0].compare("Pitch") == 0 && first == true) {
            first = false;
            k = 4;
        }
        */
        for (; k < repeats; k++) {
            string current_output_directory = slice_output_directory + "/repeat_" + to_string(k);
            mkdir(current_output_directory.c_str(), 0777);

            if (rank == 0) {
                exalt = new EXALT(population_size, number_islands, max_genomes, input_parameter_names, output_parameter_names, bp_iterations, learning_rate, use_high_threshold, high_threshold, use_low_threshold, low_threshold, use_dropout, dropout_probability, current_output_directory);

                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
                master(max_rank);
                std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
                long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                slice_times_file << milliseconds << endl;

                RNN_Genome *best_genome = exalt->get_best_genome();

                string binary_file = slice_output_directory + "/repeat_best_" + to_string(k) + ".bin";
                string graphviz_file = slice_output_directory + "/repeat_best_" + to_string(k) + ".gv";

                cout << "writing best genome to '" << binary_file << "' and '" << graphviz_file << "'" << endl;
                best_genome->write_to_file(binary_file, false);
                best_genome->write_graphviz(graphviz_file);

                delete exalt;
            } else {
                worker(rank);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            cout << "rank " << rank << " completed slice " << i << " of " << input_series.size() << " repeat " << k << " of " << repeats << endl;
        }

        slice_times_file.close();
    }

    MPI_Finalize();

    return 0;
}
