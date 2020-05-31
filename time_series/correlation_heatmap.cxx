#include <fstream>
using std::ofstream;

#include <iomanip>
using std::setw;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

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

    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    Log::debug("got time series sets.\n");

    int32_t max_lag = 0;
    get_argument(arguments, "--max_lag", true, max_lag);

    string target_parameter_name;
    get_argument(arguments, "--target_parameter_name", true, target_parameter_name);


    vector<string> parameter_names = time_series_sets->get_input_parameter_names();

    for (uint32_t i = 0; i < time_series_sets->get_number_series(); i++) {
        TimeSeriesSet *tss = time_series_sets->get_set(i);

        string input_filename = tss->get_filename();
        cout << "processing: '" << input_filename << "'" << endl;

        int32_t last_slash = input_filename.find_last_of('/') + 1;
        int32_t last_dot = input_filename.find_last_of('.');
        string prefix = input_filename.substr(last_slash, last_dot - last_slash);
        string suffix = input_filename.substr(last_dot, input_filename.length());

        cout << "prefix: '" << prefix << "'" << endl;
        cout << "suffix: '" << suffix << "'" << endl;

        string correlations_csv_filename = output_directory + "/" + prefix + "_correlations.csv";
        string headers_txt_filename = output_directory + "/" + prefix + "_headers.txt";

        cout << "correlations_csv_filename: '" << correlations_csv_filename << "'" << endl;
        cout << "headers_txt_filename: '" << headers_txt_filename << "'" << endl;

        ofstream correlations_csv(correlations_csv_filename);
        ofstream headers_txt(headers_txt_filename);

        for (uint32_t j = 0; j < parameter_names.size(); j++) {
            if (parameter_names[j].compare(target_parameter_name) == 0) continue;

            for (uint32_t k = 1; k < max_lag; k++) {
                double correlation = tss->get_correlation(target_parameter_name, parameter_names[j], k);

                if (k > 1) correlations_csv << ",";
                correlations_csv << correlation;
            }
            correlations_csv << endl;

            headers_txt << parameter_names[j] << endl;
        }

        correlations_csv.close();
        headers_txt.close();
    }

    Log::release_id("main");
    return 0;
}
