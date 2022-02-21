#include <assert.h>
#include "../rnn/examm.hxx"
#include "../common/log.hxx"
#include <chrono>
#include <mpi.h>

using namespace std::literals;

#define WORK_REQUEST_TAG 1
#define WORK_TAG 2
#define RESULT_TAG 4
#define TERMINATE_TAG 8
#define INITIALIZE_TAG 16

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

#define send_result_to(rank, work) send_msg_to(rank, work, RESULT_TAG)
void send_msg_to(int target, unique_ptr<Msg> msg, int tag=WORK_TAG) {
    ostringstream oss;
    
    msg->write_to_stream(oss);

    string value = oss.str();
    int32_t length = value.size();
    const char *buf = value.c_str();

    MPI_Send(buf, length, MPI_CHAR, target, tag, MPI_COMM_WORLD);
}

#define receive_result_from(source) receive_msg_from(source, RESULT_TAG)
#define receive_initialize() receive_msg_from(0, INITIALIZE_TAG)
unique_ptr<Msg> receive_msg_from(int source, int tag=WORK_TAG) {
    Log::debug("receiving work from: %d\n", source);
    MPI_Status status;
    MPI_Probe(source, tag, MPI_COMM_WORLD, &status);

    int length = -1;
    MPI_Get_count(&status, MPI_CHAR, &length);

    char *work_buffer = new char[length + 1];
    MPI_Recv(work_buffer, length, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
    work_buffer[length] = '\0';

    unique_ptr<Msg> msg(Msg::read_from_array(work_buffer, length));

    delete [] work_buffer;
    return msg;
}

#ifdef MASTER_PERFORMS_OPERATORS
void generate_and_send_work(GenomeOperators& go, int32_t dst, int32_t max_rank) {
#else
void generate_and_send_work(int32_t dst, int32_t max_rank) {
#endif
    assert(dst != 0);

    unique_ptr<Msg> msg = examm->generate_work();
    assert(msg != NULL);

    // Do the mutation / crossover (if master performs operations)
    switch (msg->get_msg_ty()) {
        case Msg::WORK: {
            // If we should perform the operators on the master process.
            // This code will perform the operator then create a MutationMsg
            // object with zero mutations, meaning nothing will be done on the
            // workers.
#ifdef MASTER_PERFORMS_OPERATORS 
            unique_ptr<Msg> before_operator = move(work);
    
            unique_ptr<RNN_Genome> genome = dynamic_cast<WorkMsg *>(before_operator.get())->get_genome(genome_operators);
            if (genome == nullptr) {
                Log::fatal("This should never happen - examm_mpi_core::generate_and_send_work\n");
                exit(1);
            }
            
            work = make_unique<WorkMsg>(genome);
#endif
            Log::debug("sending work to: %d\n", dst);
            break;
        }
        case Msg::TERMINATE:
            //send terminate message
            Log::info("terminating worker: %d\n", dst);
            terminates_sent++;
            Log::debug("sent: %d terminates of %d\n", terminates_sent, (max_rank - 1));
            break;

        // Unreachable
        default:
            Log::fatal("Received unexpected work type from examm (class_id = %d", msg->get_msg_ty());
            exit(1);
    }

    send_msg_to(dst, move(msg));
}

void write_time_log(string path, vector<long> ls) {
    auto log_file = new ofstream(path, std::ios_base::app);

    for (size_t i = 0; i < ls.size(); i++) {
        (*log_file) << ls[i] << endl;
    }

    log_file->close();
}

void master(int max_rank, GenomeOperators genome_operators) {
    Log::set_id("master");
    // the "main" id will have already been set by the main function so we do not need to re-set it here
    Log::debug("MAX INT: %d\n", numeric_limits<int>::max());

    // vector<long> gen_genome_times;
    // vector<long> insert_genome_times;
    // vector<long> recv_result_times;
    // vector<long> probe_times;

#define diff_as_nanos(start, end) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()

    while (terminates_sent < max_rank - 1) {
        // wait for a incoming message
        Log::debug("probing...\n");
        MPI_Status status;
        // chrono::time_point<chrono::system_clock> start_probe = chrono::system_clock::now();
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // chrono::time_point<chrono::system_clock> end_probe = chrono::system_clock::now();

        // long probe_nanos = diff_as_nanos(start_probe, end_probe);
        // probe_times.push_back(probe_nanos);

        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;
        Log::debug("probe returned message from: %d with tag: %d\n", source, tag);

        // if the message is a work request, send a genome

        if (tag == WORK_REQUEST_TAG) {
            Log::info("Received work request from %d\n", source);
            receive_work_request(source);

            // chrono::time_point<chrono::system_clock> start_gen = chrono::system_clock::now();
#ifdef MASTER_PERFORMS_OPERATORS
            generate_and_send_work(genome_operators, source, max_rank);
#else
            generate_and_send_work(source, max_rank);
#endif
            // chrono::time_point<chrono::system_clock> end_gen = chrono::system_clock::now();

            // long gen_nanos = diff_as_nanos(start_gen, end_gen);
            // gen_genome_times.push_back(gen_nanos);

            Log::info("Sent work to %d\n", source);
        } else if (tag == RESULT_TAG) {
            Log::info("Received results from %d\n", source);

            // chrono::time_point<chrono::system_clock> start_recv_result = chrono::system_clock::now();
            unique_ptr<Msg> msg = receive_msg_from(source, RESULT_TAG);
            // chrono::time_point<chrono::system_clock> end_recv_result = chrono::system_clock::now();

            // long recv_result_nanos = diff_as_nanos(start_recv_result, end_recv_result);
            // recv_result_times.push_back(recv_result_nanos);           

            ResultMsg *result = dynamic_cast<ResultMsg *>(msg.get());
            // RNN_Genome *genome = work->get_genome(genome_operators);
            unique_ptr<RNN_Genome> genome = result->get_genome();

            // chrono::time_point<chrono::system_clock> start_insert = chrono::system_clock::now();
            examm->insert_genome(move(genome));
            // chrono::time_point<chrono::system_clock> end_insert = chrono::system_clock::now();
            // long insert_nanos = diff_as_nanos(start_insert, end_insert);
            // insert_genome_times.push_back(insert_nanos);

        } else {
            Log::fatal("ERROR: received message from %d with unknown tag: %d\n", source, tag);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // vector<long> gen_genome_times;
    // vector<long> insert_genome_times;
    // vector<long> recv_result_times;
    // vector<long> probe_times;
    // string output_directory = examm->get_output_directory();
    // write_time_log(output_directory + "/gen_genomes.dat", gen_genome_times);
    // write_time_log(output_directory + "/insert_genome_times.dat", insert_genome_times);
    // write_time_log(output_directory + "/recv_result_times.dat", recv_result_times);
    // write_time_log(output_directory + "/probe_times.dat", probe_times);

    cout << "MASTER FINISHED >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";
}
 
void worker(int rank, GenomeOperators genome_operators, string id="") {
    Log::set_id("worker_" + to_string(rank) + "_" + id);
    Log::info("worker %d initialized\n", rank);

    while (true) {
        Log::debug("sending work request!\n");
        send_work_request(0);

        unique_ptr<Msg> msg = receive_msg_from(0);
        Log::info("Received work; class_id = %d\n", msg->get_msg_ty());
        // RNN_Genome *genome = work->get_genome(genome_operators);

        switch (msg->get_msg_ty()) {
            case Msg::WORK:
                break;
            default:
            case Msg::TERMINATE:
                goto done;
        }

        WorkMsg *work = static_cast<WorkMsg *>(msg.get());
        unique_ptr<RNN_Genome> genome = work->get_genome(genome_operators);

        // if genome is null we're done.
        if (genome == NULL) {
            Log::debug("Terminating worker %d %s\n", rank, id.c_str());
            break;
        }

        Log::debug("gid = %d\n", genome->get_generation_id());
        
        // have each worker write to a separate log file
        string log_id = "genome_" + to_string(genome->get_generation_id()) + "_worker_" + to_string(rank);
        Log::set_id(log_id);       

        if (genome_operators.training_parameters.bp_iterations > 0)
            genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
        else
            genome->calculate_fitness(training_inputs, training_outputs, validation_inputs, validation_outputs);

        Log::release_id(log_id);
        Log::set_id("worker_" + to_string(rank) + "_" + id);
        Log::info("Done training\n");

        // Ownership of genome has been transfered to result (when result is deleted so will the genome
        Log::info("Creating result\n");
        unique_ptr<Msg> result = unique_ptr<Msg>(static_cast<Msg*>(new ResultMsg(genome)));
        Log::info("Sending result\n");
        send_result_to(0, move(result));
        Log::info("Sent result\n");
    }

done:
    Log::info("Worker %d finished\n", rank);
    //release the log file for the worker communication
    Log::release_id("worker_" + to_string(rank) + id);
    cout << "RANK " << rank << " FINISHED #######################\n";
}

// void stop(int rank) {
//     std::cout<<"RANK: " << rank <<" -- AAAA:: XXXXXXXXXXXXXXXXXXXX\n";
//     MPI_Barrier(MPI_COMM_WORLD);
//     getchar();
// }


