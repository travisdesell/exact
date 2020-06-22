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

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "rnn/rnn_genome.hxx"

#include "time_series/time_series.hxx"


vector<string> arguments;

vector< vector< vector<double> > > testing_inputs;
vector< vector< vector<double> > > testing_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    RNN_Genome *genome = new RNN_Genome(genome_filename);

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_test(testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names());
    Log::debug("got time series sets.\n");

    string normalize_type = genome->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        time_series_sets->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        time_series_sets->normalize_avg_std_dev(genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(), genome->get_normalize_maxs());
    }
    
    Log::info("normalized type: %s \n", normalize_type.c_str());

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);

    vector<double> best_parameters = genome->get_best_parameters();
    Log::info("MSE: %lf\n", genome->get_mse(best_parameters, testing_inputs, testing_outputs));
    Log::info("MAE: %lf\n", genome->get_mae(best_parameters, testing_inputs, testing_outputs));
    genome->write_predictions(output_directory, testing_filenames, best_parameters, testing_inputs, testing_outputs, time_series_sets);

    if (Log::at_level(Log::DEBUG)) {
        int length;
        char *byte_array;

        genome->write_to_array(&byte_array, length);

        Log::debug("WROTE TO BYTE ARRAY WITH LENGTH: %d\n", length);

        RNN_Genome *duplicate_genome = new RNN_Genome(byte_array, length);

        vector<double> best_parameters_2 = duplicate_genome->get_best_parameters();
        Log::debug("duplicate MSE: %lf\n", duplicate_genome->get_mse(best_parameters_2, testing_inputs, testing_outputs));
        Log::debug("duplicate MAE: %lf\n", duplicate_genome->get_mae(best_parameters_2, testing_inputs, testing_outputs));
        duplicate_genome->write_predictions(output_directory, testing_filenames, best_parameters_2, testing_inputs, testing_outputs, time_series_sets);
    }


    Log::release_id("main");
    return 0;
}
