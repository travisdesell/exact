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

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);

    vector<string> input_parameter_names;
    input_parameter_names.push_back("Coyote-GROSS_GENERATOR_OUTPUT");
    input_parameter_names.push_back("Coyote-Net_Unit_Generation");
    input_parameter_names.push_back("Cyclone_-CYC__CONDITIONER_INLET_TEMP");
    input_parameter_names.push_back("Cyclone_-CYC__CONDITIONER_OUTLET_TEMP");
    input_parameter_names.push_back("Cyclone_-LIGNITE_FEEDER__RATE");
    input_parameter_names.push_back("Cyclone_-CYC__TOTAL_COMB_AIR_FLOW");
    input_parameter_names.push_back("Cyclone_-_MAIN_OIL_FLOW");
    input_parameter_names.push_back("Cyclone_-CYCLONE__MAIN_FLM_INT");

    vector<string> output_parameter_names;
    //output_parameter_names.push_back("Cyclone_-_MAIN_OIL_FLOW");
    output_parameter_names.push_back("Cyclone_-CYCLONE__MAIN_FLM_INT");


    vector<TimeSeriesSet*> testing_time_series;
    for (uint32_t i = 0; i < testing_filenames.size(); i++) {
        TimeSeriesSet *tss = new TimeSeriesSet(testing_filenames[i]);
        tss->select_parameters(input_parameter_names, output_parameter_names);

        //tss->normalize_min_max("Coyote-GROSS_GENERATOR_OUTPUT", 200, 500);
        tss->normalize_min_max("Coyote-GROSS_GENERATOR_OUTPUT", 0, 500);

        //tss->normalize_min_max("Coyote-Net_Unit_Generation", 200, 500);
        tss->normalize_min_max("Coyote-Net_Unit_Generation", 0, 500);

        //tss->normalize_min_max("Cyclone_-CYC__CONDITIONER_INLET_TEMP", 150, 600);
        tss->normalize_min_max("Cyclone_-CYC__CONDITIONER_INLET_TEMP", 0, 600);

        //tss->normalize_min_max("Cyclone_-CYC__CONDITIONER_OUTLET_TEMP", 90, 200);
        //tss->normalize_min_max("Cyclone_-CYC__CONDITIONER_OUTLET_TEMP", 0, 200);
        tss->normalize_min_max("Cyclone_-CYC__CONDITIONER_OUTLET_TEMP", 0, 250);

        //tss->normalize_min_max("Cyclone_-LIGNITE_FEEDER__RATE", 30, 80);
        tss->normalize_min_max("Cyclone_-LIGNITE_FEEDER__RATE", 0, 80);

        //tss->normalize_min_max("Cyclone_-CYC__TOTAL_COMB_AIR_FLOW", 190, 400);
        tss->normalize_min_max("Cyclone_-CYC__TOTAL_COMB_AIR_FLOW", 0, 400);

        //tss->normalize_min_max("Cyclone_-_MAIN_OIL_FLOW", 0, 15);
        tss->normalize_min_max("Cyclone_-_MAIN_OIL_FLOW", -1, 15);

        //tss->normalize_min_max("Cyclone_-CYCLONE__MAIN_FLM_INT", 0, 100);
        tss->normalize_min_max("Cyclone_-CYCLONE__MAIN_FLM_INT", 0, 400);

        testing_time_series.push_back(tss);
    }

    export_time_series(testing_time_series, input_parameter_names, output_parameter_names, time_offset, testing_inputs, testing_outputs);

    int number_inputs = testing_inputs[0].size();
    int number_outputs = testing_outputs[0].size();

    cout << "number_inputs: " << number_inputs << ", number_outputs: " << number_outputs << endl;

    RNN_Genome *duplicate_genome = new RNN_Genome(genome_filename, true);

    vector<double> duplicate_parameters = duplicate_genome->get_best_parameters();
    cout << "MSE: " << duplicate_genome->get_mse(duplicate_parameters, testing_inputs, testing_outputs) << endl;
    cout << "MAE: " << duplicate_genome->get_mae(duplicate_parameters, testing_inputs, testing_outputs) << endl;
    duplicate_genome->write_predictions(testing_filenames, duplicate_parameters, testing_inputs, testing_outputs);

    int length;
    char *byte_array;

    duplicate_genome->write_to_array(&byte_array, length, true);

    cout << endl << endl << "WROTE TO BYTE ARRAY WITH LENGTH: " << length << endl << endl;

    RNN_Genome *duplicate_genome_2 = new RNN_Genome(byte_array, length, true);

    vector<double> duplicate_parameters_2 = duplicate_genome_2->get_best_parameters();
    cout << "MSE: " << duplicate_genome_2->get_mse(duplicate_parameters_2, testing_inputs, testing_outputs) << endl;
    cout << "MAE: " << duplicate_genome_2->get_mae(duplicate_parameters_2, testing_inputs, testing_outputs) << endl;
    duplicate_genome_2->write_predictions(testing_filenames, duplicate_parameters_2, testing_inputs, testing_outputs);


    return 0;
}
