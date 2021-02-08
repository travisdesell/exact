#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;
using std::fixed;

#include <iostream>
using std::endl;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

//for mkdir
#include <sys/stat.h>

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "rnn/examm.hxx"

#include "time_series/time_series.hxx"

#include "examm_mt_core.cxx"

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

#define EXAMM_MT
#include "common/examm_argparse.cxx"

    string output_filename;
    get_argument(arguments, "--output_filename", true, output_filename);

    int32_t number_slices;
    get_argument(arguments, "--number_slices", true, number_slices);

    time_series_sets->split_all(number_slices);

    int32_t repeats = 5;

    if (output_directory != "") {
        mkdir(output_directory.c_str(), 0777);
    }
    ofstream overall_results(output_directory + "/overall_results.txt");

    for (uint32_t i = 0; i < time_series_sets->get_number_series(); i++) {
        vector<int> training_indexes;
        vector<int> test_indexes;

        for (int j = 0; j < time_series_sets->get_number_series(); j++) {
            if (j == i) {
                test_indexes.push_back(j);
            } else {
                training_indexes.push_back(j);
            }

        }
        time_series_sets->set_training_indexes(training_indexes);
        time_series_sets->set_test_indexes(training_indexes);

        time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
        time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

        overall_results << "results for slice " << i << " of " << time_series_sets->get_number_series() << " as test data." << endl;

        for (uint32_t k = 0; k < repeats; k++) {
            examm = make_examm();
            set_innovation_counts(examm);

            vector<thread> threads;
            for (int32_t i = 0; i < number_threads; i++) {
                threads.push_back( thread(examm_thread, i, make_genome_operators(i, edge_innovation_count, node_innovation_count)) );
            }

            for (int32_t i = 0; i < number_threads; i++) {
                threads[i].join();
            }

            finished = true;

            Log::info("completed!\n");

            RNN_Genome *best_genome = examm->get_best_genome();

            vector<double> best_parameters = best_genome->get_best_parameters();
            Log::info("training MSE: %lf\n", best_genome->get_mse(best_parameters, training_inputs, training_outputs));
            Log::info("training MSE: %lf\n", best_genome->get_mae(best_parameters, training_inputs, training_outputs));
            Log::info("validation MSE: %lf\n", best_genome->get_mse(best_parameters, validation_inputs, validation_outputs));
            Log::info("validation MSE: %lf\n", best_genome->get_mae(best_parameters, validation_inputs, validation_outputs));

            overall_results << setw(15) << fixed << best_genome->get_mse(best_parameters, training_inputs, training_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mae(best_parameters, training_inputs, training_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mse(best_parameters, validation_inputs, validation_outputs) << ", "
                << setw(15) << fixed << best_genome->get_mae(best_parameters, validation_inputs, validation_outputs) << endl;

            best_genome->write_to_file(output_directory + "/" + output_filename + "_slice_" + to_string(i) + "_repeat_" + to_string(k) + ".bin");
            best_genome->write_graphviz(output_directory + "/" + output_filename + "_slice_" + to_string(i) + "_repeat_" + to_string(k) + ".gv");

            Log::debug("deleting genome\n");
            delete best_genome;
            Log::debug("deleting exact\n");
            delete examm;
            Log::debug("deleted exact\n");
        }
        overall_results << endl;
    }

    Log::release_id("main");

    return 0;
}
