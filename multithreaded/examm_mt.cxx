#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
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
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"
#include "rnn/generate_nn.hxx"
#include "examm/examm.hxx"

#include "time_series/time_series.hxx"


mutex examm_mutex;

vector<string> arguments;

EXAMM *examm;

WeightUpdate *weight_update_method;

bool finished = false;

vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

void examm_thread(int32_t id) {

    while (true) {
        examm_mutex.lock();
        Log::set_id("main");
        RNN_Genome *genome = examm->generate_genome();
        examm_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        string log_id = "genome_" + to_string(genome->get_generation_id()) + "_thread_" + to_string(id);
        Log::set_id(log_id);
        //genome->backpropagate(training_inputs, training_outputs, validation_inputs, validation_outputs);
        genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method);
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

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    int32_t number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    TimeSeriesSets *time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    get_train_validation_data(arguments, time_series_sets, training_inputs, training_outputs, validation_inputs, validation_outputs);

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    WeightRules *weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    RNN_Genome *seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);

    examm = generate_examm_from_arguments(arguments, time_series_sets, weight_rules, seed_genome);
    
    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(examm_thread, i) );
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    Log::info("completed!\n");
    Log::release_id("main");

    return 0;
}
