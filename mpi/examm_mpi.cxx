#include <chrono>
#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "examm/examm.hxx"
#include "mpi.h"
#include "rnn/generate_nn.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

#define WORK_REQUEST_TAG  1
#define GENOME_LENGTH_TAG 2
#define GENOME_TAG        3
#define TERMINATE_TAG     4

mutex examm_mutex;

vector<string> arguments;

EXAMM* examm;
WeightUpdate* weight_update_method;

bool finished = false;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > validation_inputs;
vector<vector<vector<double> > > validation_outputs;

// bool random_sequence_length;
// int32_t sequence_length_lower_bound = 30;
// int32_t sequence_length_upper_bound = 100;

void send_work_request(int32_t target) {
    int32_t work_request_message[1];
    work_request_message[0] = 0;
    MPI_Send(work_request_message, 1, MPI_INT, target, WORK_REQUEST_TAG, MPI_COMM_WORLD);
}

void receive_work_request(int32_t source) {
    MPI_Status status;
    int32_t work_request_message[1];
    MPI_Recv(work_request_message, 1, MPI_INT, source, WORK_REQUEST_TAG, MPI_COMM_WORLD, &status);
}

RNN_Genome* receive_genome_from(int32_t source) {
    MPI_Status status;
    int32_t length_message[1];
    MPI_Recv(length_message, 1, MPI_INT, source, GENOME_LENGTH_TAG, MPI_COMM_WORLD, &status);

    int32_t length = length_message[0];

    Log::debug("receiving genome of length: %d from: %d\n", length, source);

    char* genome_str = new char[length + 1];

    Log::debug("receiving genome from: %d\n", source);
    MPI_Recv(genome_str, length, MPI_CHAR, source, GENOME_TAG, MPI_COMM_WORLD, &status);

    genome_str[length] = '\0';

    Log::trace("genome_str:\n%s\n", genome_str);

    RNN_Genome* genome = new RNN_Genome(genome_str, length);

    delete[] genome_str;
    return genome;
}

void send_genome_to(int32_t target, RNN_Genome* genome) {
    char* byte_array;
    int32_t length;

    genome->write_to_array(&byte_array, length);

    Log::debug("sending genome of length: %d to: %d\n", length, target);

    int32_t length_message[1];
    length_message[0] = length;
    MPI_Send(length_message, 1, MPI_INT, target, GENOME_LENGTH_TAG, MPI_COMM_WORLD);

    Log::debug("sending genome to: %d\n", target);
    MPI_Send(byte_array, length, MPI_CHAR, target, GENOME_TAG, MPI_COMM_WORLD);

    free(byte_array);
}

void send_terminate_message(int32_t target) {
    int32_t terminate_message[1];
    terminate_message[0] = 0;
    MPI_Send(terminate_message, 1, MPI_INT, target, TERMINATE_TAG, MPI_COMM_WORLD);
}

void receive_terminate_message(int32_t source) {
    MPI_Status status;
    int32_t terminate_message[1];
    MPI_Recv(terminate_message, 1, MPI_INT, source, TERMINATE_TAG, MPI_COMM_WORLD, &status);
}

void master_sync(int32_t max_rank) {
    max_rank -= 1;
    int32_t generation = 0;
    while (true) {
    
        // Wait for N work requests
        int32_t nreqs = 0;
        while (nreqs < max_rank) {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            int32_t source = status.MPI_SOURCE;
            int32_t tag = status.MPI_TAG;
            // Log::info("probe returned message from: %d with tag: %d\n", source, tag);
            
            if (tag == WORK_REQUEST_TAG) {
                receive_work_request(source);
                nreqs++;
            } else if (tag == GENOME_LENGTH_TAG) {
                Log::debug("received genome from: %d\n", source);
                RNN_Genome* genome = receive_genome_from(source);
                
                examm->insert_genome(genome);

                // delete the genome as it won't be used again, a copy was inserted
                delete genome;
            } else {
                Log::fatal("ERROR: received message from %d with unknown tag: %d", source, tag);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        vector<RNN_Genome *> genomes(max_rank);
        for (int32_t i = 1; i <= max_rank; i++) {
            RNN_Genome* genome = examm->generate_genome();
            if (genome == NULL)
                break;
            genomes[i - 1] = genome;
        }

        if (genomes.size() != max_rank) {
            break;
        }

        for (int i = 1; i <= max_rank; i++) {
            send_genome_to(i, genomes[i - 1]);
            delete genomes[i - 1];
        }
    }
  
    for (int i = 1; i <= max_rank; i++) {
        send_terminate_message(i);
    }
}

void master(int32_t max_rank) {
    int32_t terminates_sent = 0;

    while (true) {
        // wait for a incoming message
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int32_t source = status.MPI_SOURCE;
        int32_t tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        // if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            receive_work_request(source);

            RNN_Genome* genome = examm->generate_genome();

            if (genome == NULL) {  // search was completed if it returns NULL for an individual
                // send terminate message
                Log::info("terminating worker: %d\n", source);
                send_terminate_message(source);
                terminates_sent++;

                Log::debug("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
                if (terminates_sent >= max_rank - 1) {
                    return;
                }

            } else {
                // genome->write_to_file( examm->get_output_directory() + "/before_send_gen_" +
                // to_string(genome->get_generation_id()) );

                // send genome
                Log::debug("sending genome to: %d\n", source);
                send_genome_to(source, genome);

                // delete this genome as it will not be used again
                delete genome;
            }
        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome from: %d\n", source);
            RNN_Genome* genome = receive_genome_from(source);

            examm->insert_genome(genome);

            // delete the genome as it won't be used again, a copy was inserted
            delete genome;
            // this genome will be deleted if/when removed from population
        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

void worker(int32_t rank) {
    Log::set_id("worker_" + to_string(rank));

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        Log::debug("sent work request!\n");

        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int32_t tag = status.MPI_TAG;

        Log::debug("probe received message with tag: %d\n", tag);

        if (tag == TERMINATE_TAG) {
            Log::debug("received terminate tag!\n");
            receive_terminate_message(0);
            break;

        } else if (tag == GENOME_LENGTH_TAG) {
            Log::debug("received genome!\n");
            RNN_Genome* genome = receive_genome_from(0);

            // have each worker write the backproagation to a separate log file
            string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
            Log::set_id(log_id);

            vector<double> params;
            genome->get_weights(params);

            genome->backpropagate_stochastic(
                training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method
            );

            Log::release_id(log_id);

            // go back to the worker's log for MPI communication
            Log::set_id("worker_" + to_string(rank));

            send_genome_to(0, genome);

            delete genome;
        } else {
            Log::fatal("ERROR: received message with unknown tag: %d\n", tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank));
}

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

    TimeSeriesSets* time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    get_train_validation_data(
        arguments, time_series_sets, training_inputs, training_outputs, validation_inputs, validation_outputs
    );

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    WeightRules* weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    RNN_Genome* seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);

    bool synchronous = argument_exists(arguments, "--synchronous");
    Log::warning("synchronous? %d\n", synchronous); 

    Log::clear_rank_restriction();

    if (rank == 0) {
        write_time_series_to_file(arguments, time_series_sets);
        examm = generate_examm_from_arguments(arguments, time_series_sets, weight_rules, seed_genome);
        
        if (synchronous) {
            master_sync(max_rank);
        } else {
            master(max_rank);
        }
    } else {
        worker(rank);
    }
    Log::set_id("main_" + to_string(rank));
    finished = true;
    Log::debug("rank %d completed!\n");
    Log::release_id("main_" + to_string(rank));
    MPI_Finalize();

    delete time_series_sets;
    return 0;
}
