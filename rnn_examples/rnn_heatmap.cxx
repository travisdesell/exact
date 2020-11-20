//example usage:
//  ./rnn_examples/rnn_heatmap --std_message_level info --file_message_level none --output_directory ./ --input_directory ~/Dropbox/microbeam_cbm/Cyclone_binaries/120_min/ --testing_directory ~/Dropbox/microbeam_cbm/2019_08-09_data/ --time_offset 120
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

    string input_directory;
    get_argument(arguments, "--input_directory", true, input_directory);

    string testing_directory;
    get_argument(arguments, "--testing_directory", true, testing_directory);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    Log::info("input directory: '%s'\n", input_directory.c_str());
    Log::info("testing directory: '%s'\n", testing_directory.c_str());
    Log::info("output directory: '%s'\n", output_directory.c_str());

    string output_filename = output_directory + "heatmap_output.csv";
    ofstream output_file(output_filename);

    for (int cyclone = 1; cyclone <= 12; cyclone++) {
        string cyclone_directory = input_directory + "cyclone_" + to_string(cyclone);
        Log::info("analyzing cyclone %d with directory: '%s'\n", cyclone, cyclone_directory.c_str());

        for (int target_cyclone = 1; target_cyclone <= 12; target_cyclone++) {
            double average_mae = 0.0;

            for (int repeat = 0; repeat < 20; repeat++) {
                string repeat_directory = cyclone_directory + "/" + to_string(repeat);
                Log::trace("\tgetting genome file from repeat directory: '%s'\n", repeat_directory.c_str());

                string genome_filename = "";
                for (const auto &entry : fs::directory_iterator(repeat_directory)) {
                    Log::trace("\t\trepeat directory entry: '%s'\n", entry.path().c_str());

                    string path = entry.path();
                    if (path.find("rnn_genome") != std::string::npos) {
                        Log::trace("\t\tgot genome file: '%s'\n", path.c_str());
                        genome_filename = path;
                        break;
                    }
                }

                Log::info("\tgenome filename: '%s'\n", genome_filename.c_str());
                RNN_Genome *genome = new RNN_Genome(genome_filename);


                string testing_filename = testing_directory + "/cyclone_" + to_string(target_cyclone) + "_test.csv";

                vector<string> testing_filenames;
                testing_filenames.push_back(testing_filename);

                TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_test(testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names());
                Log::debug("got time series sets.\n");

                string normalize_type = genome->get_normalize_type();
                if (normalize_type.compare("min_max") == 0) {
                    time_series_sets->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
                } else if (normalize_type.compare("avg_std_dev") == 0) {
                    time_series_sets->normalize_avg_std_dev(genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(), genome->get_normalize_maxs());
                }

                Log::info("normalized type: %s \n", normalize_type.c_str());

                time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);

                vector<double> best_parameters = genome->get_best_parameters();

                //Log::info("MSE: %lf\n", genome->get_mse(best_parameters, testing_inputs, testing_outputs));
                //Log::info("MAE: %lf\n", genome->get_mae(best_parameters, testing_inputs, testing_outputs));
                double mae = genome->get_mae(best_parameters, testing_inputs, testing_outputs);

                cout << "MAE: " << mae << endl;

                average_mae += mae;

                delete time_series_sets;
                delete genome;
            }

            if (target_cyclone > 1) output_file << ",";
            average_mae /= 20.0;
            output_file << average_mae;

            cout << "average MAE: " << average_mae << endl << endl;
        }
        output_file << endl;
    }


    /*
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
    */

    Log::release_id("main");
    return 0;
}
