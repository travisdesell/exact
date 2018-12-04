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

#include "rnn/rnn_genome.hxx"

#include "time_series/time_series.hxx"


vector<string> arguments;

vector< vector< vector<double> > > testing_inputs;
vector< vector< vector<double> > > testing_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    RNN_Genome *genome = new RNN_Genome(genome_filename, true);

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_test(testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names());
    time_series_sets->normalize(genome->get_normalize_mins(), genome->get_normalize_maxs());

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);


    vector<double> best_parameters = genome->get_best_parameters();
    cout << "MSE: " << genome->get_mse(best_parameters, testing_inputs, testing_outputs) << endl;
    cout << "MAE: " << genome->get_mae(best_parameters, testing_inputs, testing_outputs) << endl;
    genome->write_predictions(testing_filenames, best_parameters, testing_inputs, testing_outputs);

    int length;
    char *byte_array;

    genome->write_to_array(&byte_array, length, true);

    cout << endl << endl << "WROTE TO BYTE ARRAY WITH LENGTH: " << length << endl << endl;

    RNN_Genome *duplicate_genome = new RNN_Genome(byte_array, length, true);

    vector<double> best_parameters_2 = duplicate_genome->get_best_parameters();
    cout << "MSE: " << duplicate_genome->get_mse(best_parameters_2, testing_inputs, testing_outputs) << endl;
    cout << "MAE: " << duplicate_genome->get_mae(best_parameters_2, testing_inputs, testing_outputs) << endl;
    duplicate_genome->write_predictions(testing_filenames, best_parameters_2, testing_inputs, testing_outputs);

    return 0;
}
