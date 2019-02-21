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
// cout<<"XYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYO\nXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXYOXY\n";

void acnnto_thread(int id) {

    while (true) {
        acnnto_mutex.lock();
        RNN_Genome *genome = acnnto->ants_march();
        acnnto_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        //genome->backpropagate(training_inputs, training_outputs, validation_inputs, validation_outputs);
        genome->write_graphviz("./test.gv");
        genome->print_information();
        genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
        acnnto_mutex.lock();
        acnnto->insert_genome(genome);
        acnnto_mutex.unlock();

        delete genome;
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

    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    acnnto = new ACNNTO(population_size, max_genomes, time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), bp_iterations, learning_rate, use_high_threshold, high_threshold, use_low_threshold, low_threshold, output_directory);


    vector<thread> threads;
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
