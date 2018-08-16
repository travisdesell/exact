#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;
using std::fixed;

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

    string series_filename;
    get_argument(arguments, "--series_filename", true, series_filename);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

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

    string log_filename = "";
    get_argument(arguments, "--log_filename", false, log_filename);

    string output_filename;
    get_argument(arguments, "--output_filename", true, output_filename);

    int32_t number_slices;
    get_argument(arguments, "--number_slices", true, number_slices);

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

    TimeSeriesSet *tss = new TimeSeriesSet(series_filename);
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

    vector<TimeSeriesSet*> slices;
    tss->split(number_slices, slices);

    vector<TimeSeriesSet*> training_series;
    vector<TimeSeriesSet*> validation_series;

    int32_t repeats = 10;

    ofstream overall_results("overall_results.txt");

    for (uint32_t i = 0; i < number_slices; i++) {
        while (training_series.size() > 0) {
            TimeSeriesSet *current = training_series.back();
            training_series.pop_back();
            delete current;
        }

        while (validation_series.size() > 0) {
            TimeSeriesSet *current = validation_series.back();
            validation_series.pop_back();
            delete current;
        }

        for (uint32_t j = 0; j < number_slices; j++) {
            if (j == i) {
                validation_series.push_back(slices[j]);
            } else {
                training_series.push_back(slices[j]);
            }
        }
        export_time_series(training_series, input_parameter_names, output_parameter_names, time_offset, training_inputs, training_outputs);
        export_time_series(validation_series, input_parameter_names, output_parameter_names, time_offset, validation_inputs, validation_outputs);

        overall_results << "results for slice " << i << " as test data." << endl;

        for (uint32_t k = 0; k < repeats; k++) {
            exalt = new EXALT(population_size, max_genomes, input_parameter_names, output_parameter_names, bp_iterations, learning_rate, use_high_threshold, high_threshold, use_low_threshold, low_threshold, use_dropout, dropout_probability, log_filename);

            vector<thread> threads;
            for (int32_t i = 0; i < number_threads; i++) {
                threads.push_back( thread(exalt_thread, i) );
            }

            for (int32_t i = 0; i < number_threads; i++) {
                threads[i].join();
            }

            finished = true;

            cout << "completed!" << endl;

            RNN_Genome *best_genome = exalt->get_best_genome();

            vector<double> best_parameters = best_genome->get_best_parameters();
            cout << "training MSE: " << best_genome->get_mse(best_parameters, training_inputs, training_outputs) << endl;
            cout << "training MSE: " << best_genome->get_mae(best_parameters, training_inputs, training_outputs) << endl;
            cout << "validation MSE: " << best_genome->get_mse(best_parameters, validation_inputs, validation_outputs) << endl;
            cout << "validation MSE: " << best_genome->get_mae(best_parameters, validation_inputs, validation_outputs) << endl;

            overall_results << setw(15) << fixed << best_genome->get_mse(best_parameters, training_inputs, training_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mae(best_parameters, training_inputs, training_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mse(best_parameters, validation_inputs, validation_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mae(best_parameters, validation_inputs, validation_outputs) << endl;

            best_genome->write_to_file(output_filename + "_slice_" + to_string(i) + "_repeat_" + to_string(k) + ".bin", false);
            best_genome->write_graphviz(output_filename + "_slice_" + to_string(i) + "_repeat_" + to_string(k) + ".gv");

            /*
            RNN_Genome *duplicate_genome = new RNN_Genome(output_filename, false);

            vector<double> duplicate_parameters = duplicate_genome->get_best_parameters();
            cout << "training MSE: " << duplicate_genome->get_mse(duplicate_parameters, training_inputs, training_outputs) << endl;
            cout << "training MSE: " << duplicate_genome->get_mae(duplicate_parameters, training_inputs, training_outputs) << endl;
            cout << "validation MSE: " << duplicate_genome->get_mse(duplicate_parameters, validation_inputs, validation_outputs) << endl;
            cout << "validation MSE: " << duplicate_genome->get_mae(duplicate_parameters, validation_inputs, validation_outputs) << endl;
            */

            delete best_genome;
            delete exalt;
        }
        overall_results << endl;
    }

    return 0;
}
