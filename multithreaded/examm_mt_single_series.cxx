#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;
using std::fixed;

#include <iostream>
using std::endl;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

//for mkdir
#include <sys/stat.h>

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

string output_directory = "";

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
        genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs, weight_update_method);        Log::release_id(log_id);

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
    get_argument(arguments, "--output_directory", true, output_directory);
    string output_filename;
    get_argument(arguments, "--output_filename", true, output_filename);
    int32_t number_slices;
    get_argument(arguments, "--number_slices", true, number_slices);
    
    if (output_directory != "") {
        mkdir(output_directory.c_str(), 0777);
    }

    TimeSeriesSets *time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    time_series_sets->split_all(number_slices);
    get_train_validation_data(arguments, time_series_sets, training_inputs, training_outputs, validation_inputs, validation_outputs);

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    WeightRules *weight_rules = new WeightRules();
    weight_rules->initialize_from_args(arguments);

    RNN_Genome *seed_genome = get_seed_genome(arguments, time_series_sets, weight_rules);

    weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    int32_t repeats = 5;
    ofstream overall_results(output_directory + "/overall_results.txt");

    for (int32_t i = 0; i < time_series_sets->get_number_series(); i++) {
        vector<int32_t> training_indexes;
        vector<int32_t> test_indexes;

        for (int32_t j = 0; j < time_series_sets->get_number_series(); j++) {
            if (j == i) {
                test_indexes.push_back(j);
            } else {
                training_indexes.push_back(j);
            }

        }
        time_series_sets->set_training_indexes(training_indexes);
        time_series_sets->set_test_indexes(training_indexes);

        get_train_validation_data(arguments, time_series_sets, training_inputs, training_outputs, validation_inputs, validation_outputs);

        overall_results << "results for slice " << i << " of " << time_series_sets->get_number_series() << " as test data." << endl;

        for (int32_t k = 0; k < repeats; k++) {
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

            RNN_Genome *best_genome = examm->get_best_genome();

            vector<double> best_parameters = best_genome->get_best_parameters();
            Log::info("training MSE: %lf\n", best_genome->get_mse(best_parameters, training_inputs, training_outputs));
            Log::info("training MSE: %lf\n", best_genome->get_mae(best_parameters, training_inputs, training_outputs));
            Log::info("validation MSE: %lf\n", best_genome->get_mse(best_parameters, validation_inputs, validation_outputs));
            Log::info("validation MSE: %lf\n", best_genome->get_mae(best_parameters, validation_inputs, validation_outputs));

            overall_results << setw(15) << fixed << best_genome->get_mse(best_parameters, training_inputs, training_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mae(best_parameters, training_inputs, training_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mse(best_parameters, validation_inputs, validation_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mae(best_parameters, validation_inputs, validation_outputs) << endl;

            best_genome->write_to_file(output_directory + "/" + output_filename + "_slice_" + to_string(i) + "_repeat_" + to_string(k) + ".bin");
            best_genome->write_graphviz(output_directory + "/" + output_filename + "_slice_" + to_string(i) + "_repeat_" + to_string(k) + ".gv");

            Log::debug("deleting genome\n");
            delete best_genome;
            Log::debug("deleting exact\n");
            delete examm;
            Log::debug("deleted exact\n");
        }
        overall_results << endl;
    }

    Log::release_id("main");

    return 0;
}
