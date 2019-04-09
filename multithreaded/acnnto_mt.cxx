#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;

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

#include "common/arguments.hxx"

#include "rnn/acnnto.hxx"

#include "time_series/time_series.hxx"


mutex acnnto_mutex;

vector<string> arguments;

ACNNTO *acnnto;


bool finished = false;


vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

void acnnto_thread(int id) {

    while (true) {
        acnnto_mutex.lock();
        RNN_Genome *genome = acnnto->ants_march();
        acnnto_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        //genome->backpropagate(training_inputs, training_outputs, validation_inputs, validation_outputs);
        // genome->write_graphviz("./test.gv");
        genome->print_information();
        genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
        acnnto_mutex.lock();
        acnnto->insert_genome(genome);
        acnnto_mutex.unlock();

        delete genome;
        cout<<"ONE MORE GENOME COMING!\n";
        getchar();
    }
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    int number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    cout << "exported time series." << endl;

    int number_inputs = time_series_sets->get_number_inputs();
    int number_outputs = time_series_sets->get_number_outputs();

    cout << "number_inputs: " << number_inputs << ", number_outputs: " << number_outputs << endl;

    int32_t population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    int32_t bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);

    double learning_rate = 0.001;
    get_argument(arguments, "--learning_rate", false, learning_rate);

    double high_threshold = 1.0;
    bool use_high_threshold = get_argument(arguments, "--high_threshold", false, high_threshold);

    double low_threshold = 0.05;
    bool use_low_threshold = get_argument(arguments, "--low_threshold", false, low_threshold);

    int32_t number_of_ants = 50;
    get_argument(arguments, "--ants", false, number_of_ants);

    int32_t hidden_layers_depth = 0;
    get_argument(arguments, "--hidden_layers_depth", false, hidden_layers_depth);

    int32_t hidden_layer_nodes = 0;
    get_argument(arguments, "--hidden_layer_nodes", false, hidden_layer_nodes);

    double pheromone_decay_parameter = 1.0;
    get_argument(arguments, "--pheromone_decay_parameter", false, pheromone_decay_parameter);

    double pheromone_update_strength = 0.7;
    get_argument(arguments, "--pheromone_update_strength", false, pheromone_update_strength);

    double pheromone_heuristic = 0.3;
    get_argument(arguments, "--pheromone_heuristic", false, pheromone_heuristic);

    double weight_reg_parameter = 0.0;
    get_argument(arguments, "--weight_reg_parameter", false, weight_reg_parameter);

    int32_t max_recurrent_depth = 3;
    get_argument(arguments, "--max_recurrent_depth", true, max_genomes);

    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    acnnto = new ACNNTO(population_size, max_genomes, time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), bp_iterations, learning_rate, use_high_threshold, high_threshold, use_low_threshold, low_threshold, output_directory, number_of_ants, hidden_layers_depth, hidden_layer_nodes, pheromone_decay_parameter, pheromone_update_strength, pheromone_heuristic, max_recurrent_depth, weight_reg_parameter );


    vector<thread> threads;
    cout<<"NUMBER OF THREADS: "<<number_threads<<endl;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(acnnto_thread, i) );
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    cout << "completed!" << endl;

    return 0;
}
