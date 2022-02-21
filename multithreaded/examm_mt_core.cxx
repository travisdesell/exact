#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "rnn/examm.hxx"
#include "rnn/msg.hxx"
#include "rnn/training_parameters.hxx"
#include "rnn/genome_operators.hxx"

mutex examm_mutex;

vector<string> arguments;

EXAMM *examm;

bool finished = false;

vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

void examm_thread(int id, GenomeOperators genome_operators, bool random_sequence_length, int lower_length_bound, int upper_length_bound) {

    unique_ptr<RNN_Genome> g;
    string main_id = "examm_thread_" + to_string(id);
    while (true) {
        Log::set_id(main_id);
        examm_mutex.lock();

        if (g != nullptr)
            examm->insert_genome(move(g));
        unique_ptr<Msg> work = examm->generate_work();

        examm_mutex.unlock();

        switch (work->get_msg_ty()) {
            case Msg::TERMINATE:
                goto done;
                break;

            case Msg::WORK: {
                    WorkMsg *wm = dynamic_cast<WorkMsg*>(work.get());
                    g = wm->get_genome(genome_operators);
                    break;
                }
            default:
                Log::fatal("Should never recieve a message of type %d in examm_mt\n", work->get_msg_ty());
                exit(1);
                break;
        }

        string log_id = "genome_" + to_string(g->get_generation_id()) + "_thread_" + to_string(id);
        Log::set_id(log_id);
        if (genome_operators.training_parameters.bp_iterations > 0)
            g->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
        else
            g->calculate_fitness(training_inputs, training_outputs, validation_inputs, validation_outputs);
        Log::release_id(log_id);
    }

    done:
    Log::info("Thread %d terminating\n", id);
    Log::release_id(main_id);
}

void get_individual_inputs(string str, vector<string>& tokens) {
   string word = "";
   for (auto x : str) {
       if (x == ',') {
           tokens.push_back(word);
           word = "";
       } else {
           word = word + x;
        }
   }
   tokens.push_back(word);
}
