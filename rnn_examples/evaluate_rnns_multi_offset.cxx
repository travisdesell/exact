/**
 * Some usage examples:
 *
 *
 * ./rnn_examples/evaluate_rnns_multi_offset --genome_filenames ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/boiler_best_bin_gv_files/nose_gas_temperature/time_offset_*.bin --time_offsets 1 2 4 8 --testing_filenames ~/Dropbox/1537\ MTI-RIT/boiler_hourly/database_l_minus_10.csv --output_filename ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/nose_gas_temperature_predictons.csv
 * 
 * ./rnn_examples/evaluate_rnns_multi_offset --genome_filenames ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/boiler_best_bin_gv_files/net_plant_heat_rate/time_offset_*.bin --time_offsets 1 2 4 8 --testing_filenames ~/Dropbox/1537\ MTI-RIT/boiler_hourly/database_l_minus_10.csv --output_filename ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/net_plant_heat_rate_predictions.csv
 *
 * ./rnn_examples/evaluate_rnns_multi_offset --genome_filenames ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/plant_best_bin_gv_files/cyclone3/extra_plant_parameters/time_offset_*.bin --time_offsets 1 15 30 60 120 240 480 --testing_filenames ~/Dropbox/1537\ MTI-RIT/Field_Test_data/20190910_v0.2/cyclone3_file4.csv --output_filename ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_extra_3.csv
 *
 * ./rnn_examples/evaluate_rnns_multi_offset --genome_filenames ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/plant_best_bin_gv_files/cyclone3/plant_fuel_parameters/time_offset_*.bin --time_offsets 1 15 30 60 120 240 480 --testing_filenames ~/Dropbox/1537\ MTI-RIT/Field_Test_data/20190910_v0.2/cyclone3_file4.csv --output_filename ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_plant_fuel_3.csv
 *
 * ./rnn_examples/evaluate_rnns_multi_offset --genome_filenames ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/plant_best_bin_gv_files/cyclone3/plant_parameters/time_offset_*.bin --time_offsets 1 15 30 60 120 240 480 --testing_filenames ~/Dropbox/1537\ MTI-RIT/Field_Test_data/20190910_v0.2/cyclone3_file4.csv --output_filename ~/Dropbox/1537\ MTI-RIT/2019_degruyter_results/flame_intensity_plant_3.csv
 *
 * then you can use plot_multi_time_series.py to generate a chart of the time series
 */


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

    vector<string> genome_filenames;
    get_argument_vector(arguments, "--genome_filenames", true, genome_filenames);

    vector<RNN_Genome*> genomes;
    for (int32_t i = 0; i < genome_filenames.size(); i++) {
        Log::info("reading genome filename: %s\n", genome_filenames[i].c_str());
        genomes.push_back(new RNN_Genome(genome_filenames[i]));
    }

    vector<int32_t> time_offsets;
    get_argument_vector(arguments, "--time_offsets", true, time_offsets);

    if (time_offsets.size() != genome_filenames.size()) {
        Log::fatal("ERROR: number of time_offsets (%d) != number of genome_files: (%d)\n", time_offsets.size(), genome_filenames.size());
        exit(1);
    }

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    string output_filename;
    get_argument(arguments, "--output_filename", true, output_filename);


    vector< vector<double> > all_series;

    //TODO: should check that all genomes have the same output parameter name(s)

    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_test(testing_filenames, genomes[0]->get_input_parameter_names(), genomes[0]->get_output_parameter_names());


    string normalize_type = genomes[0]->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        time_series_sets->normalize_min_max(genomes[0]->get_normalize_mins(), genomes[0]->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        time_series_sets->normalize_avg_std_dev(genomes[0]->get_normalize_avgs(), genomes[0]->get_normalize_std_devs(), genomes[0]->get_normalize_mins(), genomes[0]->get_normalize_maxs());
    }

    vector< vector<double> > full_series;

    //TODO: only working with one output type currently
    string output_parameter_name = genomes[0]->get_output_parameter_names()[0];

    time_series_sets->export_series_by_name(output_parameter_name, full_series);

    all_series.push_back(full_series[0]);
    Log::debug("output_parameter_name: %s, full_series.size(): %d, full_series[0].size(): %d\n", output_parameter_name.c_str(), full_series.size(), full_series[0].size());

    delete time_series_sets;

    for (int32_t i = 0; i < genomes.size(); i++) {
        time_series_sets = TimeSeriesSets::generate_test(testing_filenames, genomes[i]->get_input_parameter_names(), genomes[i]->get_output_parameter_names());
        Log::debug("got time series sets.\n");
        string normalize_type = genomes[i]->get_normalize_type();
        if (normalize_type.compare("min_max") == 0) {
            time_series_sets->normalize_min_max(genomes[i]->get_normalize_mins(), genomes[i]->get_normalize_maxs());
        } else if (normalize_type.compare("avg_std_dev") == 0) {
            time_series_sets->normalize_avg_std_dev(genomes[i]->get_normalize_avgs(), genomes[i]->get_normalize_std_devs(), genomes[i]->get_normalize_mins(), genomes[i]->get_normalize_maxs());
        }

       Log::debug("normalized time series.\n");

        time_series_sets->export_test_series(time_offsets[i], testing_inputs, testing_outputs);

        vector<double> best_parameters = genomes[i]->get_best_parameters();
        Log::info("MSE: %lf\n", genomes[i]->get_mse(best_parameters, testing_inputs, testing_outputs));
        Log::info("MAE: %lf\n", genomes[i]->get_mae(best_parameters, testing_inputs, testing_outputs));

        vector< vector<double> > predictions = genomes[i]->get_predictions(best_parameters, testing_inputs, testing_outputs);

        Log::debug("predictions.size(): %d\n", predictions.size());

        if (predictions.size() != 1) {
            Log::fatal("ERROR: had more than one testing file, currently only one supported.\n");
            exit(1);
        }

        Log::debug("genomes[%d] had %d outputs.\n", i, predictions[0].size());

        all_series.push_back(predictions[0]);

        delete time_series_sets;
    }

    ofstream outfile(output_filename);

    //print the column headeers
    outfile << "#" << output_parameter_name;
    for (int32_t i = 1; i < all_series.size(); i++) {
        outfile << "," << output_parameter_name << "_offset" << time_offsets[i-1];
    }
    outfile << endl;

    Log::debug("all_series.size(): %d\n", all_series.size());
    for (int32_t row = 0; row < all_series[0].size(); row++) {
        for (int32_t i = 0; i < all_series.size(); i++) {
            Log::debug("all_series[%d].size(): %d\n", i, all_series[i].size());

            if (i == 0) outfile << all_series[0][row];
            else {
                if (row < time_offsets[i - 1]) outfile << ",";
                else outfile << "," << all_series[i][row - time_offsets[i-1]];
            }

        }
        outfile << endl;
    }

    Log::release_id("main");

    return 0;
}
