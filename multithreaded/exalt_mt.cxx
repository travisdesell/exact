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

#include "rnn/exalt.hxx"

#include "time_series/time_series.hxx"


mutex exalt_mutex;

vector<string> arguments;

EXALT *exalt;


bool finished = false;

int images_resize;


vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;


void exalt_thread(int id) {

    while (true) {
        exalt_mutex.lock();
        RNN_Genome *genome = exalt->generate_genome();
        exalt_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        //genome->backpropagate(training_inputs, training_outputs, validation_inputs, validation_outputs);
        genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);

        exalt_mutex.lock();
        exalt->insert_genome(genome);
        exalt_mutex.unlock();

        delete genome;
    }
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    int number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    vector<string> training_filenames;
    get_argument_vector(arguments, "--training_filenames", true, training_filenames);

    vector<string> validation_filenames;
    get_argument_vector(arguments, "--validation_filenames", true, validation_filenames);

    vector<TimeSeriesSet*> all_time_series;

    int training_rows = 0;
    vector<TimeSeriesSet*> training_time_series;
    cout << "got training time series filenames:" << endl;
    for (uint32_t i = 0; i < training_filenames.size(); i++) {
        cout << "\t" << training_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(training_filenames[i], 1.0);
        training_time_series.push_back( ts );
        all_time_series.push_back( ts );

        training_rows += ts->get_number_rows();
    }
    cout << "number training files: " << training_filenames.size() << ", total rows for training flights: " << training_rows << endl;

    int validation_rows = 0;
    vector<TimeSeriesSet*> validation_time_series;
    cout << "got test time series filenames:" << endl;
    for (uint32_t i = 0; i < validation_filenames.size(); i++) {
        cout << "\t" << validation_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(validation_filenames[i], -1.0);
        validation_time_series.push_back( ts );
        all_time_series.push_back( ts );

        validation_rows += ts->get_number_rows();
    }
    cout << "number test files: " << validation_filenames.size() << ", total rows for test flights: " << validation_rows << endl;

    if (argument_exists(arguments, "--normalize")) {
        normalize_time_series_sets(all_time_series);
        cout << "normalized all time series" << endl;
    } else {
        cout << "not normalizing time series" << endl;
    }


    vector<string> input_parameter_names;
    /*
    input_parameter_names.push_back("indicated_airspeed");
    input_parameter_names.push_back("msl_altitude");
    input_parameter_names.push_back("eng_1_rpm");
    input_parameter_names.push_back("eng_1_fuel_flow");
    input_parameter_names.push_back("eng_1_oil_press");
    input_parameter_names.push_back("eng_1_oil_temp");
    input_parameter_names.push_back("eng_1_cht_1");
    input_parameter_names.push_back("eng_1_cht_2");
    input_parameter_names.push_back("eng_1_cht_3");
    input_parameter_names.push_back("eng_1_cht_4");
    input_parameter_names.push_back("eng_1_egt_1");
    input_parameter_names.push_back("eng_1_egt_2");
    input_parameter_names.push_back("eng_1_egt_3");
    input_parameter_names.push_back("eng_1_egt_4");
    */
    
    /*
    input_parameter_names.push_back("par1");
    input_parameter_names.push_back("par2");
    input_parameter_names.push_back("par3");
    input_parameter_names.push_back("par4");
    input_parameter_names.push_back("par5");
    input_parameter_names.push_back("par6");
    input_parameter_names.push_back("par7");
    input_parameter_names.push_back("par8");
    input_parameter_names.push_back("par9");
    input_parameter_names.push_back("par10");
    input_parameter_names.push_back("par11");
    input_parameter_names.push_back("par12");
    input_parameter_names.push_back("par13");
    input_parameter_names.push_back("par14");
    input_parameter_names.push_back("vib");
    */

    input_parameter_names.push_back("Coyote-GROSS_GENERATOR_OUTPUT");
    input_parameter_names.push_back("Coyote-Net_Unit_Generation");
    input_parameter_names.push_back("Cyclone_01-CYC_01_CONDITIONER_INLET_TEMP");
    input_parameter_names.push_back("Cyclone_01-CYC_01_CONDITIONER_OUTLET_TEMP");
    input_parameter_names.push_back("Cyclone_01-LIGNITE_FEEDER_01_RATE");
    input_parameter_names.push_back("Cyclone_01-CYC_01_TOTAL_COMB_AIR_FLOW");
    input_parameter_names.push_back("Cyclone_01-1_MAIN_OIL_FLOW");
    input_parameter_names.push_back("Cyclone_01-CYCLONE_1_MAIN_FLM_INT");



    vector<string> output_parameter_names;
    output_parameter_names.push_back("Cyclone_01-1_MAIN_OIL_FLOW");

    //output_parameter_names.push_back("vib");

    //output_parameter_names.push_back("indicated_airspeed");
    //output_parameter_names.push_back("eng_1_oil_press");
    /*
    output_parameter_names.push_back("msl_altitude");
    output_parameter_names.push_back("eng_1_rpm");
    output_parameter_names.push_back("eng_1_fuel_flow");
    output_parameter_names.push_back("eng_1_oil_press");
    output_parameter_names.push_back("eng_1_oil_temp");
    output_parameter_names.push_back("eng_1_cht_1");
    output_parameter_names.push_back("eng_1_cht_2");
    output_parameter_names.push_back("eng_1_cht_3");
    output_parameter_names.push_back("eng_1_cht_4");
    output_parameter_names.push_back("eng_1_egt_1");
    output_parameter_names.push_back("eng_1_egt_2");
    output_parameter_names.push_back("eng_1_egt_3");
    output_parameter_names.push_back("eng_1_egt_4");
    */

    int32_t time_offset = 10;

    training_inputs.resize(training_time_series.size());
    training_outputs.resize(training_time_series.size());
    for (uint32_t i = 0; i < training_time_series.size(); i++) {
        training_time_series[i]->export_time_series(training_inputs[i], input_parameter_names, -time_offset);
        training_time_series[i]->export_time_series(training_outputs[i], output_parameter_names, time_offset);
    }

    validation_inputs.resize(validation_time_series.size());
    validation_outputs.resize(validation_time_series.size());
    for (uint32_t i = 0; i < validation_time_series.size(); i++) {
        validation_time_series[i]->export_time_series(validation_inputs[i], input_parameter_names, -time_offset);
        validation_time_series[i]->export_time_series(validation_outputs[i], output_parameter_names, time_offset);
    }


    int number_inputs = training_inputs[0].size();
    int number_outputs = training_outputs[0].size();

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

    double dropout_probability = 0.0;
    bool use_dropout = get_argument(arguments, "--dropout_probability", false, dropout_probability);

    exalt = new EXALT(population_size, max_genomes, number_inputs, number_outputs, bp_iterations, learning_rate, use_high_threshold, high_threshold, use_low_threshold, low_threshold, use_dropout, dropout_probability);


    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(exalt_thread, i) );
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    cout << "completed!" << endl;

    return 0;
}
