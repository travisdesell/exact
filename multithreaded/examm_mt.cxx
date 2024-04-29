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

#include <map>
using std::map;

#include <cmath>
using std::isnan;

#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "examm/examm.hxx"
#include "rnn/generate_nn.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

mutex examm_mutex;

vector<string> arguments;

EXAMM* examm;

WeightUpdate* weight_update_method;

bool finished = false;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > validation_inputs;
vector<vector<vector<double> > > validation_outputs;

void examm_thread(int32_t id) {
    while (true) {
        examm_mutex.lock();
        Log::set_id("main");
        RNN_Genome* genome = examm->generate_genome();
        examm_mutex.unlock();

        if (genome == NULL) {
            break;  // generate_individual returns NULL when the search is done
        }
        
        string log_id = "genome_" + to_string(genome->get_generation_id()) + "_thread_" + to_string(id);
        Log::set_id(log_id);
        // genome->backpropagate(training_inputs, training_outputs, validation_inputs, validation_outputs);
        int32_t initial_genome_size = examm->get_island_size() * examm->get_number_islands();
        if (examm->get_mutate_rl() && !genome->get_is_initializing()){
          //double validation_mse_before = genome->get_mse(genome->get_best_parameters(), validation_inputs, validation_outputs);
            genome->backpropagate_stochastic(
                training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method
            );
            double validation_mse_after = genome->get_mse(genome->get_best_parameters(), validation_inputs, validation_outputs);
            double reward = genome->get_best_parent_mse() - validation_mse_after; 
            for (const auto& pair : *genome->get_generated_by_map()){
                if (pair.first.compare("initial") == 0 || pair.first.compare("crossover") == 0 || pair.first.compare("island_crossover") == 0 ){
                    break;
                } 
                
                if (!isnan(reward) && (genome->get_best_parent_mse() < EXAMM_MAX_DOUBLE)) {
                    examm->update_mutation_to_rewards(pair.first, reward); 
                }
                
            }
            if (genome->get_generation_id() > initial_genome_size){
                double new_epsilon = examm->get_epsilon() + (1.0 / (examm->get_max_genomes() - initial_genome_size));
                examm->set_epsilon(new_epsilon);
                Log::info("New Epsilon: %f\n", examm->get_epsilon());
                Log::info("Mutation Counts:\n");
                for (const auto& pair : examm->get_mutation_to_count()) {
                    Log::info("%s: %0.f\n", pair.first.c_str(), round(pair.second));
                }
                Log::info("Mutation Rewards:\n");
                for (const auto& pair : examm->get_mutation_to_rewards()) {
                    Log::info("%s: %f\n", pair.first.c_str(), pair.second);
                }
            }            
        } else {
            Log::info("DID NOT UPDATE REWARDS!\n");
            genome->backpropagate_stochastic(
                training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method
            );    
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
        } else {
            word = word + x;
        }
    }
    tokens.push_back(word);
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    int32_t number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

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

    examm = generate_examm_from_arguments(arguments, time_series_sets, weight_rules, seed_genome);

    string mutate_function_type = ""; 
    get_argument(arguments, "--mutate_function_type", false, mutate_function_type);
    double epsilon = 0;
    get_argument(arguments, "--epsilon", false, epsilon);
    if (mutate_function_type.compare("") != 0){
        examm->set_mutate_function_type(mutate_function_type);
        examm->set_epsilon(epsilon);
        examm->initialize_mutation_to_rewards(mutate_function_type);
    } 

    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back(thread(examm_thread, i));
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    Log::info("completed!\n");
    Log::release_id("main");

    return 0;
}
