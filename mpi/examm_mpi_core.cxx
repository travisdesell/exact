#include "rnn/work/work.hxx"

#include <assert.h>

#define WORK_REQUEST_TAG 1
#define WORK_TAG 2
#define RESULT_TAG 4
#define TERMINATE_TAG 8
#define INITIALIZE_TAG 16

// mutex examm_mutex;

vector<string> arguments;

EXAMM *examm;

bool finished = false;
int terminates_sent = 0;

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

#define send_result_to(rank, work) send_work_to(rank, work, RESULT_TAG)
void send_work_to(int target, Work *work, int tag=WORK_TAG) {
    Log::info("C0\n");
    ostringstream oss;
    Log::info("C1\n");
    
    work->write_to_stream(oss);
    Log::info("C2\n");

    string value = oss.str();
    int32_t length = value.size();
    const char *buf = value.c_str();

    Log::info("C3\n");
    MPI_Send(buf, length, MPI_CHAR, target, tag, MPI_COMM_WORLD);
}

#define receive_result_from(source) receive_work_from(source, RESULT_TAG)
#define receive_initialize() receive_work_from(0, INITIALIZE_TAG)
Work *receive_work_from(int source, int tag=WORK_TAG) {
    Log::debug("receiving work from: %d\n", source);
    MPI_Status status;
    MPI_Probe(source, tag, MPI_COMM_WORLD, &status);

    int length = -1;
    MPI_Get_count(&status, MPI_CHAR, &length);

    char *work_buffer = new char[length + 1];
    MPI_Recv(work_buffer, length, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
    work_buffer[length] = '\0';

    Work *work = Work::read_from_array(work_buffer, length);

    delete [] work_buffer;
    return work;
}

void generate_and_send_work(int32_t dst, int32_t max_rank) {
    assert(dst != 0);

    Work *work = examm->generate_work();
    assert(work != NULL);

    int class_id = work->get_class_id();

    assert( class_id == MutationWork::class_id
         || class_id == CrossoverWork::class_id
         || class_id == TerminateWork::class_id);

    switch (work->get_class_id()) {
        case MutationWork::class_id:
        case CrossoverWork::class_id:
            Log::debug("sending work to: %d\n", dst);
            break;

        case TerminateWork::class_id:
            //send terminate message
            Log::info("terminating worker: %d\n", dst);
            terminates_sent++;
            Log::debug("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
            break;

        // Unreachable
        default:
            Log::fatal("Received unexpected work type from examm (class_id = %d", work->get_class_id());
            exit(1);
    }

    send_work_to(dst, work);

    delete work;
}

void master(int max_rank, GenomeOperators genome_operators) {
    Log::set_id("master");
    //the "main" id will have already been set by the main function so we do not need to re-set it here
    Log::debug("MAX INT: %d\n", numeric_limits<int>::max());

    Work *initialize_work = examm->get_initialize_work();

    // Send an init message to each worker
    // There are max_rank - 1 workers, and worker ranks start from 1 (0 is master)
    for (int i = 0; i < max_rank - 1; i++) {
        int worker_rank = i + 1;
        send_work_to(worker_rank, initialize_work, INITIALIZE_TAG);
        Log::debug("Sent init to rank %d\n", worker_rank);
    }

    while (terminates_sent < max_rank - 1) {
        //wait for a incoming message
        Log::debug("probing...\n");
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        //if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            Log::info("Received work request from %d\n", source);
            receive_work_request(source);
            generate_and_send_work(source, max_rank);
            Log::info("Sent work to %d\n", source);
        } else if (tag == RESULT_TAG) {
            Log::info("Received results from %d\n", source);
            Work *work = receive_work_from(source, RESULT_TAG);
            RNN_Genome *genome = work->get_genome(genome_operators);
           
            int class_id = work->get_class_id();
            assert(class_id == WorkResult::class_id);
            
            examm->insert_genome(genome);
            
            delete genome;
            delete work;
        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d\n", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    cout << "MASTER FINISHED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
}

void worker_init(int rank, GenomeOperators &genome_operators) {
    Work *work = receive_initialize();

    if (InitializeWork *init_work = dynamic_cast<InitializeWork*>(work)) {
        init_work->update_genome_operators(rank, genome_operators); 
    } else {
        Log::fatal("Received wrong type of work for initialization");
        exit(1);
    }
}

void worker(int rank, GenomeOperators genome_operators, string id="") {
    Log::set_id("worker_" + to_string(rank) + "_" + id);
    
    Log::info("starting worker initialization process (rank = %d)\n", rank);
    worker_init(rank, genome_operators);
    Log::info("worker %d initialized\n", rank);

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);
        
        Work* work = receive_work_from(0);
        Log::info("Received work; class_id = %d\n", work->get_class_id());

        RNN_Genome *genome = work->get_genome(genome_operators);
        delete work;
    
        Log::debug("gid = %d\n", genome->get_generation_id());

        // if genome is null we're done.
        if (genome == NULL) {
            Log::debug("Terminating worker %d %s\n", rank, id.c_str());
            break;
        }
        
        // have each worker write to a separate log file
        string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
        Log::set_id(log_id);       

        if (genome_operators.training_parameters.bp_iterations > 0) {
            genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
        } else {
            genome->calculate_fitness(validation_inputs, validation_outputs);
        }

        Log::release_id(log_id);
        Log::set_id("worker_" + to_string(rank) + "_" + id);
        Log::info("Done training\n");

        //go back to the worker's log for MPI communication
        
        // Ownership of genome has been transfered to result (when result is deleted so will the genome
        Log::info("Creating result\n");
        Work *result = new WorkResult(genome);
        Log::info("Sending result\n");
        send_result_to(0, result);
        Log::info("Sent result\n");
        delete result;
        Log::info("result deleted\n");
    }
    
    Log::info("Worker finished\n");
    //release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank) + id);

    cout << "RANK " << rank << " FINISHED #######################\n";
}

// void stop(int rank) {
//     std::cout<<"RANK: " << rank <<" -- AAAA:: XXXXXXXXXXXXXXXXXXXX\n";
//     MPI_Barrier(MPI_COMM_WORLD);
//     getchar();
// }


