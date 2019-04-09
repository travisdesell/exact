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

#include <vector>
using std::vector;

#include "mpi.h"

#include "common/arguments.hxx"

#include "rnn/acnnto.hxx"

#include "time_series/time_series.hxx"

#include <sys/stat.h>

#define WORK_REQUEST_TAG 1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG 3
#define TERMINATE_TAG 4

mutex acnnto_mutex;

vector<string> arguments;

ACNNTO *acnnto;

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

            acnnto_mutex.lock();
            RNN_Genome *genome = acnnto->ants_march();
            acnnto_mutex.unlock();


            // vector<RNN_Edge*> eedges = genome->get_edges();
            // for (int i=0; i< eedges.size(); i++){
            //     cout << "\tMPI-Master:: Edge[" << i << "] Weight:" << eedges[i]->get_weight() << endl;
            // }
            // cout << "\t MPI-Master:: genome->get_generation_id(): " << genome->get_generation_id() << endl;
            // cout << "\t MPI-Master:: genome->edges: " << &eedges << endl;

            if (genome == NULL) { //search was completed if it returns NULL for an individual
                //send terminate message
                cout << "[" << setw(10) << name << "] terminating worker: " << source << endl;
                send_terminate_message(source);
                terminates_sent++;

                cout << "[" << setw(10) << name << "] sent: " << terminates_sent << " terminates of: " << (max_rank - 1) << endl;
                if (terminates_sent >= max_rank - 1) return;

                acnnto->print_last_population();

            } else {
                //genome->write_to_file( acnnto->get_output_directory() + "/before_send_gen_" + to_string(genome->get_generation_id()) );

                //send genome
                cout << "[" << setw(10) << name << "] sending genome to: " << source << endl;
                send_genome_to(name, source, genome);

                //delete this genome as it will not be used again
                delete genome;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            cout << "[" << setw(10) << name << "] received genome from: " << source << endl;
            RNN_Genome *genome = receive_genome_from(name, source);

            acnnto_mutex.lock();
            acnnto->insert_genome(genome);
            acnnto_mutex.unlock();

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
            cout << "Worker=> GENOME weights #: " << genome->get_number_weights() << endl;

            // vector<RNN_Edge*> eedges = genome->get_edges();
            // for (int i=0; i< eedges.size(); i++){
            //     cout << "\tMPI-Worker (Before BackProp):: Edge[" << i << "] Weight:" << eedges[i]->get_weight() << endl;
            // }
            // cout << "\t MPI-Worker:: genome->get_generation_id(): " << genome->get_generation_id() << endl;
            // cout << "\t MPI-Worker:: genome->edges: " << &eedges << endl;


            genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);

            // eedges = genome->get_edges();
            // for (int i=0; i< eedges.size(); i++){
            //     cout << "\tMPI-Worker (After BackProp):: Edge[" << i << "] Weight:" << eedges[i]->get_weight() << endl;
            // }

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

    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    int32_t max_recurrent_depth = 3;
    get_argument(arguments, "--max_recurrent_depth", true, max_recurrent_depth);

    int32_t number_of_ants = 50;
    get_argument(arguments, "--ants", false, number_of_ants);

    int32_t hidden_layers_depth = 0;
    get_argument(arguments, "--hidden_layers_depth", false, hidden_layers_depth);

    int32_t hidden_layer_nodes = 0;
    get_argument(arguments, "--hidden_layer_nodes", false, hidden_layer_nodes);

    double pheromone_decay_parameter = 1.0;
    get_argument(arguments, "--pheromone_decay_parameter", false, pheromone_decay_parameter);

    double pheromone_update_strength = 0.7;
    get_argument(arguments, "--pheromone_update_strength", true, pheromone_update_strength);

    double pheromone_heuristic = 0.3;
    get_argument(arguments, "--pheromone_heuristic", false, pheromone_heuristic);

    double weight_reg_parameter = 0.3;
    get_argument(arguments, "--weight_reg_parameter", false, weight_reg_parameter);

    int32_t bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    double learning_rate = 0.001;
    get_argument(arguments, "--learning_rate", false, learning_rate);

    double high_threshold = 1.0;
    bool use_high_threshold = get_argument(arguments, "--high_threshold", false, high_threshold);

    double low_threshold = 0.05;
    bool use_low_threshold = get_argument(arguments, "--low_threshold", false, low_threshold);

    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    string reward_type = "";
    get_argument(arguments, "--reward_type", true, reward_type);

    if (rank == 0) {
        cout << "NUMBER OF ANTS:: " << number_of_ants << endl;
        cout << "DECAY         :: " << pheromone_decay_parameter << endl;
        cout << "UPDATE        :: " << pheromone_update_strength << endl;
        string log_dir_str;
        ostringstream dum;
        dum << number_of_ants;
            dum << "_";
        dum << pheromone_decay_parameter;
        dum << "_";
        dum << pheromone_update_strength;
        log_dir_str = output_directory;
        log_dir_str += "/";
        log_dir_str += dum.str();
        if (mkdir(log_dir_str.c_str(), 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;
        else
        cout << "Directory created: " << log_dir_str.c_str() << endl;
        output_directory = log_dir_str.c_str();
        acnnto = new ACNNTO(population_size, max_genomes, time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), bp_iterations, learning_rate, use_high_threshold, high_threshold, use_low_threshold, low_threshold, output_directory, number_of_ants, hidden_layers_depth, hidden_layer_nodes, pheromone_decay_parameter, pheromone_update_strength, pheromone_heuristic, max_recurrent_depth, weight_reg_parameter, reward_type );
        master(max_rank);
    } else {
        worker(rank);
    }

    finished = true;

    cout << "rank " << rank << " completed!" << endl;

    MPI_Finalize();

    return 0;
}
