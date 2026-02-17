#include <chrono>

#include <iomanip>
using std::setw;
using std::fixed;
using std::setprecision;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "common/process_arguments.hxx"
#include "rnn/rnn_genome.hxx"
#include "time_series/time_series.hxx"
#include "weights/weight_rules.hxx"
#include "weights/weight_update.hxx"

vector<string> arguments;

vector<vector<vector<double> > > training_inputs;
vector<vector<vector<double> > > training_outputs;
vector<vector<vector<double> > > testing_inputs;
vector<vector<vector<double> > > testing_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);

    // Load the genome
    Log::info("Loading genome from: %s\n", genome_filename.c_str());
    RNN_Genome* genome = new RNN_Genome(genome_filename);
    
    // Generate time series sets from arguments (will load data compatible with genome)
    TimeSeriesSets* time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    
    // Create weight update method
    WeightUpdate* weight_update_method = new WeightUpdate();
    weight_update_method->generate_from_arguments(arguments);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);
    
    // Export training and testing series
    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);

    // Normalize the training data using the genome's normalization parameters
    string normalize_type = genome->get_normalize_type();
    if (normalize_type.compare("min_max") == 0) {
        time_series_sets->normalize_min_max(genome->get_normalize_mins(), genome->get_normalize_maxs());
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        time_series_sets->normalize_avg_std_dev(
            genome->get_normalize_avgs(), genome->get_normalize_std_devs(), genome->get_normalize_mins(),
            genome->get_normalize_maxs()
        );
    }

    // Evaluate BEFORE fine-tuning
    Log::info("=== BEFORE FINE-TUNING ===\n");
    vector<double> initial_parameters = genome->get_best_parameters();
    double initial_mse = genome->get_mse(initial_parameters, training_inputs, training_outputs);
    double initial_validation_mse = genome->get_mse(initial_parameters, testing_inputs, testing_outputs);
    Log::info("Initial Training MSE: %.6f\n", initial_mse);
    Log::info("Initial Testing MSE: %.6f\n", initial_validation_mse);
    
    int32_t finetune_iterations = 100;
    get_argument(arguments, "--finetune_iterations", true, finetune_iterations);
    bool use_stochastic = true;
    get_argument(arguments, "--finetune_stochastic", true, use_stochastic);
    
    Log::info("Starting fine-tuning with %d iterations, stochastic: %s\n", finetune_iterations, use_stochastic ? "true" : "false");
    
    // Set the number of backpropagation iterations for fine-tuning
    genome->set_bp_iterations(finetune_iterations);
    genome->set_weights(initial_parameters);
    
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
    
    // Evaluate AFTER fine-tuning
    Log::info("\n=== AFTER FINE-TUNING ===\n");
    vector<double> final_parameters = genome->get_best_parameters();
    double final_mse = genome->get_mse(final_parameters, training_inputs, training_outputs);
    double final_validation_mse = genome->get_mse(final_parameters, testing_inputs, testing_outputs);
    Log::info("Final Training MSE: %.6f\n", final_mse);
    Log::info("Final Testing MSE: %.6f\n", final_validation_mse);
    Log::info("Improvement: %.6f\n", initial_validation_mse - final_validation_mse);
    
    // Save the fine-tuned genome
    string output_genome_filename = output_directory + "/finetuned_genome.bin";
    genome->write_to_file(output_genome_filename);
    Log::info("Fine-tuned genome saved to: %s\n", output_genome_filename.c_str());
    
    // Save predictions if requested
    bool save_predictions = argument_exists(arguments, "--save_predictions");
    if (save_predictions) {
        vector<string> testing_filenames;
        get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);
        
        genome->write_predictions(
            output_directory, testing_filenames, final_parameters, testing_inputs, testing_outputs, time_series_sets
        );
        Log::info("Predictions saved to: %s\n", output_directory.c_str());
    }

    Log::release_id("main");
    delete genome;
    delete time_series_sets;
    delete weight_update_method;
    
    return 0;
}
