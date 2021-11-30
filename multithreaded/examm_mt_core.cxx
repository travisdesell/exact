#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "rnn/examm.hxx"
#include "rnn/work/work.hxx"
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

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> d01(0.0, 1.0);
    while (true) {
        examm_mutex.lock();
        Log::set_id("main");
        fflush(0);
        Work *work = examm->generate_work();
        examm_mutex.unlock();

        // RNN_Genome *genome = work->get_genome(genome_operators);
        RNN_Genome *genome = Work::get_genome(work, genome_operators);

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        string log_id = "genome_" + to_string(genome->get_generation_id()) + "_thread_" + to_string(id);
        Log::set_id(log_id);
        if (d01(rng) < 0.5) {
            for (int i = 0; i < 100; i++) {
                Log::set_id(log_id + to_string(i));

                for (int j = 0; j < 100; j++)
                    Log::info("CONTENTION\n");

                Log::release_id(log_id + to_string(i));
            }
        } else {
            for (int i = 0; i < 10000; i++) {
                Log::info("CONTENTION\n");
            }
        }
        Log::release_id(log_id);

        examm_mutex.lock();
        Log::set_id("main");
        examm->insert_genome(genome);
        examm_mutex.unlock();

        delete genome;
    }

}

void get_individual_inputs(string str, vector<string>& tokens) {
   string word = "";
   for (auto x : str) {
       if (x == ',') {
           tokens.push_back(word);
           word = "";
       }else
           word = word + x;
   }
   tokens.push_back(word);
}
