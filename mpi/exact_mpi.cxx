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

#include "image_tools/image_set.hxx"

#include "cnn/exact.hxx"

#define WORK_REQUEST_TAG 1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG 3
#define TERMINATE_TAG 4

mutex exact_mutex;

vector<string> arguments;

EXACT *exact;

bool finished = false;

int images_resize;

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

CNN_Genome* receive_genome_from(string name, int source) {
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

    istringstream iss(genome_str);

    CNN_Genome* genome = new CNN_Genome(iss, false);

    delete [] genome_str;
    return genome;
}

void send_genome_to(string name, int target, CNN_Genome* genome) {
    ostringstream oss;

    genome->write(oss);

    string genome_str = oss.str();
    int length = genome_str.size();

    cout << "[" << setw(10) << name << "] sending genome of length: " << length << " to: " << target << endl;

    int length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    cout << "[" << setw(10) << name << "] sending genome to: " << target << endl;
    MPI_Send(genome_str.c_str(), length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);
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

void master(const Images &training_images, const Images &validation_images, const Images &testing_images, int max_rank) {
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

            exact_mutex.lock();
            CNN_Genome *genome = exact->generate_individual();
            exact_mutex.unlock();

            if (genome == NULL) { //search was completed if it returns NULL for an individual
                //send terminate message
                cout << "[" << setw(10) << name << "] terminating worker: " << source << endl;
                send_terminate_message(source);
                terminates_sent++;

                cout << "[" << setw(10) << name << "] sent: " << terminates_sent << " terminates of: " << (max_rank - 1) << endl;
                if (terminates_sent >= max_rank - 1) return;

            } else {
                ofstream outfile(exact->get_output_directory() + "/gen_" + to_string(genome->get_generation_id()));
                genome->write(outfile);
                outfile.close();

                //send genome
                cout << "[" << setw(10) << name << "] sending genome to: " << source << endl;
                send_genome_to(name, source, genome);

                //delete this genome as it will not be used again
                delete genome;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            cout << "[" << setw(10) << name << "] received genome from: " << source << endl;
            CNN_Genome *genome = receive_genome_from(name, source);

            exact_mutex.lock();
            exact->insert_genome(genome);
            exact_mutex.unlock();

            //this genome will be deleted if/when removed from population
        } else {
            cerr << "[" << setw(10) << name << "] ERROR: received message with unknown tag: " << tag << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(const Images &training_images, const Images &validation_images, const Images &testing_images, int rank) {
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
            CNN_Genome* genome = receive_genome_from(name, 0);

            genome->set_name(name);
            genome->initialize();
            genome->stochastic_backpropagation(training_images, images_resize, validation_images);
            genome->evaluate_test(testing_images);

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

    string training_filename;
    get_argument(arguments, "--training_file", true, training_filename);

    string testing_filename;
    get_argument(arguments, "--testing_file", true, testing_filename);

    string validation_filename;
    get_argument(arguments, "--validation_file", true, validation_filename);

    int padding;
    get_argument(arguments, "--padding", true, padding);

    int population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int max_epochs;
    get_argument(arguments, "--max_epochs", true, max_epochs);

    bool use_sfmp;
    get_argument(arguments, "--use_sfmp", true, use_sfmp);

    bool use_node_operations;
    get_argument(arguments, "--use_node_operations", true, use_node_operations);

    int max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string search_name;
    get_argument(arguments, "--search_name", true, search_name);

    bool reset_edges;
    get_argument(arguments, "--reset_edges", true, reset_edges);

    get_argument(arguments, "--images_resize", true, images_resize);

    Images training_images(training_filename, padding);
    Images validation_images(validation_filename, padding, training_images.get_average(), training_images.get_std_dev());
    Images testing_images(testing_filename, padding, training_images.get_average(), training_images.get_std_dev());

    if (rank == 0) {
        exact = new EXACT(training_images, validation_images, testing_images, padding, population_size, max_epochs, use_sfmp, use_node_operations, max_genomes, output_directory, search_name, reset_edges);

        master(training_images, validation_images, testing_images, max_rank);
    } else {
        worker(training_images, validation_images, testing_images, rank);
    }

    finished = true;

    cout << "rank " << rank << " completed!" << endl;

    MPI_Finalize();

    return 0;
}
