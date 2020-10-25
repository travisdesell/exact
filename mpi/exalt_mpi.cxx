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

bool finished = false;

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

    cout << "MAX INT: " << numeric_limits<int>::max() << endl;

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
                cerr << "[" << setw(10) << name << "] terminating worker: " << source << endl;
                send_terminate_message(source);
                terminates_sent++;

                cout << "[" << setw(10) << name << "] sent: " << terminates_sent << " terminates of: " << (max_rank - 1) << endl;
                cerr << "[" << setw(10) << name << "] sent: " << terminates_sent << " terminates of: " << (max_rank - 1) << endl;
                if (terminates_sent >= max_rank - 1) return;

            } else {
                //genome->write_to_file( exalt->get_output_directory() + "/before_send_gen_" + to_string(genome->get_generation_id()) );

                //send genome
                cout << "[" << setw(10) << name << "] sending genome to: " << source << endl;
                cerr << "[" << setw(10) << name << "] sending genome to: " << source << endl;
                send_genome_to(name, source, genome);

                //delete this genome as it will not be used again
                delete genome;
				cerr << "FINISHED SENDING GEMONE: " << genome->get_generation_id() << " TO WORKER AND DELETED LOCAL COPY" << endl ;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            cout << "[" << setw(10) << name << "] received genome from: " << source << endl;
            cerr << "[" << setw(10) << name << "] received genome from: " << source << endl;
            RNN_Genome *genome = receive_genome_from(name, source);

            exalt_mutex.lock();
            exalt->insert_genome(genome);
            exalt_mutex.unlock();
			cerr << "FINISHED INSERTING GENOME: " << genome->get_generation_id() << endl ;

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
			cerr << "WORKER SAYING: Starting BP for Genome: "<< genome->get_generation_id() << endl ;
			//genome->write_graphviz(to_string(genome->get_generation_id()) +".gv") ;
			//genome->write_to_file(to_string(genome->get_generation_id())  +".bin") ;
            genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
			cerr << "WORKER SAYING: Finished BP for Genome: "<< genome->get_generation_id() << endl ;

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

    TimeSeriesSets *time_series_sets = NULL;
    if (rank == 0) {
        //only have the master process print TSS info
        time_series_sets = TimeSeriesSets::generate_from_arguments(arguments, true);

        if (argument_exists(arguments, "--write_time_series")) {
            string base_filename;
            get_argument(arguments, "--write_time_series", true, base_filename);
            time_series_sets->write_time_series_sets(base_filename);
        }
    } else {
        time_series_sets = TimeSeriesSets::generate_from_arguments(arguments, false);
    }

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    int number_inputs = time_series_sets->get_number_inputs();
    int number_outputs = time_series_sets->get_number_outputs();

    cout << "number_inputs: " << number_inputs << ", number_outputs: " << number_outputs << endl;

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

    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    if (rank == 0) {
        exalt = new EXALT(population_size, number_islands, max_genomes, time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), bp_iterations, learning_rate, use_high_threshold, high_threshold, use_low_threshold, low_threshold, use_dropout, dropout_probability, output_directory);

        master(max_rank);
    } else {
        worker(rank);
    }

    finished = true;

    cout << "rank " << rank << " completed!" << endl;

    MPI_Finalize();

    return 0;
}
