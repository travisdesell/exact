#include <chrono>

#include <iomanip>
using std::setw;
using std::fixed;
using std::setprecision;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;


#include "mpi.h"

#include "common/arguments.hxx"
#include "common/files.hxx"
#include "common/log.hxx"

#include "rnn/examm.hxx"

#include "time_series/time_series.hxx"
#include "mpi/examm_mpi_core.cxx"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_rank(rank);
    Log::set_id("main_" + to_string(rank));
    Log::restrict_to_rank(0);

#include "common/examm_argparse.cxx"

    //only have the master process print TSS info
    if (rank == 0 && argument_exists(arguments, "--write_time_series")) {
        string base_filename;
        get_argument(arguments, "--write_time_series", true, base_filename);
        time_series_sets->write_time_series_sets(base_filename);
    }

    int32_t repeats;
    get_argument(arguments, "--repeats", true, repeats);

    int fold_size = 2;
    get_argument(arguments, "--fold_size", true, fold_size);

    Log::clear_rank_restriction();

    for (int32_t i = 0; i < time_series_sets->get_number_series(); i += fold_size) {
        vector<int> training_indexes;
        vector<int> test_indexes;

        for (uint32_t j = 0; j < time_series_sets->get_number_series(); j += fold_size) {
            if (j == i) {
                for (int k = 0; k < fold_size; k++) {
                    test_indexes.push_back(j + k);
                }
            } else {
                for (int k = 0; k < fold_size; k++) {
                    training_indexes.push_back(j + k);
                }
            }
        }

        time_series_sets->set_training_indexes(training_indexes);
        time_series_sets->set_test_indexes(test_indexes);

        //time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
        //time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

        string slice_output_directory = output_directory + "/slice_" + to_string(i);
        mkpath(slice_output_directory.c_str(), 0777);
        ofstream slice_times_file(output_directory + "/slice_" + to_string(i) + "_runtimes.csv");

        for (int k = 0; k < repeats; k++) {
            string current_output_directory = slice_output_directory + "/repeat_" + to_string(k);
            mkpath(current_output_directory.c_str(), 0777);

            //set to the master/workers can specify the right log id
            int global_slice = i;
            int global_repeat = k;

            if (rank == 0) {
                string examm_log_id = "examm_slice_" + to_string(global_slice) + "_repeat_" + to_string(global_repeat);
                Log::set_id(examm_log_id);

                examm = make_examm();

                std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
                master(max_rank, make_genome_operators(0, -1, -1));
                std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
                long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

                //examm->write_memory_log(current_output_directory + "/memory_fitness_log.csv");

                slice_times_file << milliseconds << endl;

                RNN_Genome *best_genome = examm->get_best_genome();

                string binary_file = slice_output_directory + "/repeat_best_" + to_string(k) + ".bin";
                string graphviz_file = slice_output_directory + "/repeat_best_" + to_string(k) + ".gv";

                Log::debug("writing best genome to '%s' and '%s'\n", binary_file.c_str(), graphviz_file.c_str());
                best_genome->write_to_file(binary_file);
                best_genome->write_graphviz(graphviz_file);

                delete examm;
                Log::release_id(examm_log_id);
            } else {
                worker(rank, make_genome_operators(rank, -1, -1), "slice_" + to_string(global_slice) + "_repeat_" + to_string(global_repeat));
            }
            Log::set_id("main_" + to_string(rank));

            MPI_Barrier(MPI_COMM_WORLD);
            Log::debug("rank %d completed slice %d of %d repeat %d of %d\n", rank, i, time_series_sets->get_number_series(), k, repeats);
        }

        slice_times_file.close();
    }

    MPI_Finalize();
    Log::release_id("main_" + to_string(rank));

    return 0;
}
