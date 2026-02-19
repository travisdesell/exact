#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_update.hxx"

vector<string> arguments;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > testing_inputs;
vector<vector<vector<double> > > testing_outputs;

// Find the best genome file in a directory by validation MSE
string find_best_genome(string directory) {
    vector<string> genome_files;
    
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        Log::fatal("Error: Could not open directory: %s\n", directory.c_str());
        return "";
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        string filename = entry->d_name;
        if (filename.find(".bin") == filename.length() - 4) {
            string full_path = directory + "/" + filename;
            genome_files.push_back(full_path);
        }
    }
    closedir(dir);
    
    if (genome_files.empty()) {
        Log::fatal("Error: No .bin genome files found in directory: %s\n", directory.c_str());
        return "";
    }
    
    // First, check if there's a "global_best_genome.bin" (without generation ID)
    for (const string& file : genome_files) {
        string filename = file.substr(file.find_last_of("/") + 1);
        if (filename == "global_best_genome.bin") {
            Log::info("Found %d genome files, using global_best_genome.bin\n", (int)genome_files.size());
            return file;
        }
    }
    
    // Otherwise, load each genome and find the one with the best (lowest) validation MSE
    string best_file = genome_files[0];
    double best_mse = EXAMM_MAX_DOUBLE;
    int valid_genomes = 0;
    
    for (const string& file : genome_files) {
        try {
            RNN_Genome* genome = new RNN_Genome(file);
            double validation_mse = genome->get_best_validation_mse();
            
            // Check if this is a valid MSE (not NaN or max value)
            if (validation_mse < best_mse && validation_mse != EXAMM_MAX_DOUBLE && !isnan(validation_mse)) {
                best_mse = validation_mse;
                best_file = file;
            }
            valid_genomes++;
            delete genome;
        } catch (...) {
            // Skip files that can't be loaded
            continue;
        }
    }
    
    if (valid_genomes == 0) {
        Log::fatal("Error: Could not load any valid genome files from directory: %s\n", directory.c_str());
        return "";
    }
    
    Log::info("Found %d genome files, using best (validation MSE: %.6f): %s\n", 
              (int)genome_files.size(), best_mse, best_file.c_str());
    return best_file;
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string genome_directory;
    bool use_directory = get_argument(arguments, "--genome_directory", false, genome_directory);
    
    string genome_filename;
    if (!use_directory) {
        get_argument(arguments, "--genome_file", true, genome_filename);
    } else {
        genome_filename = find_best_genome(genome_directory);
        if (genome_filename.empty()) {
            Log::release_id("main");
            return 1;
        }
    }
    
    Log::info("Loading genome from: %s\n", genome_filename.c_str());
    RNN_Genome* genome = new RNN_Genome(genome_filename);

    // Load training and validation data
    vector<string> training_filenames;
    get_argument_vector(arguments, "--training_filenames", true, training_filenames);
    
    vector<string> validation_filenames;
    get_argument_vector(arguments, "--validation_filenames", true, validation_filenames);

    // Add parameter names from genome to arguments (required by generate_from_arguments)
    if (!argument_exists(arguments, "--input_parameter_names")) {
        vector<string> input_params = genome->get_input_parameter_names();
        arguments.push_back("--input_parameter_names");
        for (const string& param : input_params) {
            arguments.push_back(param);
        }
    }
    
    if (!argument_exists(arguments, "--output_parameter_names")) {
        vector<string> output_params = genome->get_output_parameter_names();
        arguments.push_back("--output_parameter_names");
        for (const string& param : output_params) {
            arguments.push_back(param);
        }
    }



    // Generate time series sets from arguments
    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);

    // Normalize using genome's stored normalization parameters
    string normalize_type = genome->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        time_series_sets->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        time_series_sets->normalize_avg_std_dev(
            genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(),
            genome->get_normalize_maxs()
        );
    }

    Log::info("normalized type: %s \n", normalize_type.c_str());

    // Create weight update method
    WeightUpdate* weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", false, time_offset);

    // Export training and testing series
    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);

    // Evaluate BEFORE fine-tuning
    Log::info("=== BEFORE FINE-TUNING ===\n");
    vector<double> initial_parameters = genome->get_best_parameters();
    double initial_training_mse = genome->get_mse(initial_parameters, training_inputs, training_outputs);
    double initial_testing_mse = genome->get_mse(initial_parameters, testing_inputs, testing_outputs);
    double initial_training_mae = genome->get_mae(initial_parameters, training_inputs, training_outputs);
    double initial_testing_mae = genome->get_mae(initial_parameters, testing_inputs, testing_outputs);
    
    Log::info("Initial Training MSE: %.6f\n", initial_training_mse);
    Log::info("Initial Training MAE: %.6f\n", initial_training_mae);
    Log::info("Initial Testing MSE: %.6f\n", initial_testing_mse);
    Log::info("Initial Testing MAE: %.6f\n", initial_testing_mae);

    // Fine-tuning parameters
    int32_t finetune_iterations = 100;
    get_argument(arguments, "--finetune_iterations", false, finetune_iterations);
    bool use_stochastic = true;
    get_argument(arguments, "--finetune_stochastic", false, use_stochastic);
    
    Log::info("\nStarting fine-tuning with %d iterations, stochastic: %s\n", finetune_iterations, use_stochastic ? "true" : "false");
    
    genome->set_bp_iterations(finetune_iterations);
    
    // Perform fine-tuning
    if (use_stochastic) {
        genome->backpropagate_stochastic(
            training_inputs, training_outputs, testing_inputs, testing_outputs, weight_update_method
        );
    } else {
        genome->backpropagate(
            training_inputs, training_outputs, testing_inputs, testing_outputs, weight_update_method
        );
    }
    
    // Get the updated parameters after backpropagation
    // Backpropagation updates best_parameters internally, but we ensure we get the current weights
    vector<double> final_parameters;
    genome->get_weights(final_parameters);
    
    // Update the genome's best_parameters to ensure they're saved correctly
    genome->set_best_parameters(final_parameters);
    
    // Evaluate AFTER fine-tuning
    Log::info("\n=== AFTER FINE-TUNING ===\n");
    double final_training_mse = genome->get_mse(final_parameters, training_inputs, training_outputs);
    double final_testing_mse = genome->get_mse(final_parameters, testing_inputs, testing_outputs);
    double final_training_mae = genome->get_mae(final_parameters, training_inputs, training_outputs);
    double final_testing_mae = genome->get_mae(final_parameters, testing_inputs, testing_outputs);
    
    Log::info("Final Training MSE: %.6f\n", final_training_mse);
    Log::info("Final Training MAE: %.6f\n", final_training_mae);
    Log::info("Final Testing MSE: %.6f\n", final_testing_mse);
    Log::info("Final Testing MAE: %.6f\n", final_testing_mae);
    Log::info("Training MSE Improvement: %.6f\n", initial_training_mse - final_training_mse);
    Log::info("Testing MSE Improvement: %.6f\n", initial_testing_mse - final_testing_mse);
    
    // Write metrics to CSV file if requested
    string metrics_csv_file;
    if (get_argument(arguments, "--metrics_csv_file", false, metrics_csv_file)) {
        FILE* csv_file = fopen(metrics_csv_file.c_str(), "w");  // write mode (overwrite)
        if (csv_file != nullptr) {
            // Write CSV row: initial_mse,initial_mae,final_mse,final_mae,mse_difference,mae_difference
            fprintf(csv_file, "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    initial_testing_mse, initial_testing_mae,
                    final_testing_mse, final_testing_mae,
                    initial_testing_mse - final_testing_mse,
                    initial_testing_mae - final_testing_mae);
            fclose(csv_file);
            Log::info("Metrics written to CSV file: %s\n", metrics_csv_file.c_str());
        } else {
            Log::error("Could not write metrics to CSV file: %s\n", metrics_csv_file.c_str());
        }
    }
    
    // Save the fine-tuned genome
    string output_genome_filename = output_directory + "/finetuned_genome.bin";
    genome->write_to_file(output_genome_filename);
    Log::info("\nFine-tuned genome saved to: %s\n", output_genome_filename.c_str());
    
    // Write predictions for testing data
    genome->write_predictions(
        output_directory, validation_filenames, final_parameters, testing_inputs, testing_outputs, time_series_sets
    );
    Log::info("Predictions saved to: %s\n", output_directory.c_str());

    Log::release_id("main");
    delete genome;
    delete time_series_sets;
    delete weight_update_method;
    
    return 0;
}
