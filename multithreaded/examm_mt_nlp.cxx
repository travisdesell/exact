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

#include "rnn/examm.hxx"

#include "word_series/word_series.hxx"

mutex examm_mutex;

vector<string> arguments;

EXAMM *examm;


bool finished = false;


vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;


void examm_thread(int id) {

    while (true) {
        examm_mutex.lock();
        Log::set_id("main");
        RNN_Genome *genome = examm->generate_genome();
        examm_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        string log_id = "genome_" + to_string(genome->get_generation_id()) + "_thread_" + to_string(id);
        Log::set_id(log_id);
        genome->backpropagate(training_inputs, training_outputs, validation_inputs, validation_outputs);
        //genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
        Log::release_id(log_id);

        examm_mutex.lock();
        Log::set_id("main");
        examm->insert_genome(genome);
        examm_mutex.unlock();

        delete genome;
    }

}

void get_individual_inputs(string str, vector<string>& tokens) {
    string word = "";
    for (auto x : str) {
        if (x == ',') {
            tokens.push_back(word);
            word = "";
        } else {
            word = word + x;
        }
    }
    tokens.push_back(word);
}


int main(int argc, char  **argv)
{
	vector<string> arguments = vector<string>(argv, argv + argc);

	Log::initialize(arguments);
    Log::set_id("main");

    int number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    int32_t word_offset = 1;
    get_argument(arguments,"--word_offset",true,word_offset);


	Corpus* corpus_sets = Corpus::generate_from_arguments(arguments);

	corpus_sets->export_training_series(word_offset,training_inputs,training_outputs);
	corpus_sets->export_test_series(word_offset,validation_inputs,validation_outputs);

	Log::info("exported word series.\n");
    
	int number_inputs = corpus_sets->get_number_inputs();
    int number_outputs = corpus_sets->get_number_outputs();

    Log::info("The size of the input is :: %d,%d,%d \n",training_inputs.size(),training_inputs[0].size(),training_inputs[0][0].size());

	Log::info("number_inputs: %d, number_outputs: %d\n", number_inputs, number_outputs);

	int32_t population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int32_t number_islands;
    get_argument(arguments, "--number_islands", true, number_islands);

    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    string speciation_method = "";
    get_argument(arguments, "--speciation_method", false, speciation_method);

    int32_t extinction_event_generation_number = 0;
    get_argument(arguments, "--extinction_event_generation_number", false, extinction_event_generation_number);
  
    int32_t islands_to_exterminate;
    get_argument(arguments, "--islands_to_exterminate", false, islands_to_exterminate);

    string island_ranking_method = "";
    get_argument(arguments, "--island_ranking_method", false, island_ranking_method);

    string repopulation_method = "";
    get_argument(arguments, "--repopulation_method", false, repopulation_method);

    int32_t repopulation_mutations = 0;
    get_argument(arguments, "--repopulation_mutations", false, repopulation_mutations);

    double species_threshold = 0.0;
    get_argument(arguments, "--species_threshold", false, species_threshold);
        
    double fitness_threshold = 100;
    get_argument(arguments, "--fitness_threshold", false, fitness_threshold);

    double neat_c1 = 1;
    get_argument(arguments, "--neat_c1", false, neat_c1);

    double neat_c2 = 1;
    get_argument(arguments, "--neat_c2", false, neat_c2);

    double neat_c3 = 1;
    get_argument(arguments, "--neat_c3", false, neat_c3);

    bool repeat_extinction = argument_exists(arguments, "--repeat_extinction");

    int32_t epochs_acc_freq = 0;
    get_argument(arguments, "--epochs_acc_freq", false, epochs_acc_freq);
    
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

    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);

    vector<string> possible_node_types;
    get_argument_vector(arguments, "--possible_node_types", false, possible_node_types);

    int32_t min_recurrent_depth = 1;
    get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);

    int32_t max_recurrent_depth = 10;
    get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);

    //bool use_regression = argument_exists(arguments, "--use_regression");
    bool use_regression = false; //NLP will always use softmax, not regression

    string weight_initialize_string = "random";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);
    WeightType weight_initialize;
    weight_initialize = get_enum_from_string(weight_initialize_string);
    
    string weight_inheritance_string = "lamarckian";
    get_argument(arguments, "--weight_inheritance", false, weight_inheritance_string);
    WeightType weight_inheritance;
    weight_inheritance = get_enum_from_string(weight_inheritance_string);

    string mutated_component_weight_string = "lamarckian";
    get_argument(arguments, "--mutated_component_weight", false, mutated_component_weight_string);
    WeightType mutated_component_weight;
    mutated_component_weight = get_enum_from_string(mutated_component_weight_string);


    RNN_Genome *seed_genome = NULL;
    string genome_file_name = "";
    if (get_argument(arguments, "--genome_bin", false, genome_file_name)) {
        seed_genome = new RNN_Genome(genome_file_name);
        seed_genome->set_normalize_bounds(corpus_sets->get_normalize_type(), corpus_sets->get_normalize_mins(), corpus_sets->get_normalize_maxs(), corpus_sets->get_normalize_avgs(), corpus_sets->get_normalize_std_devs());

        string transfer_learning_version;
        get_argument(arguments, "--transfer_learning_version", true, transfer_learning_version);

        bool epigenetic_weights = argument_exists(arguments, "--epigenetic_weights");

        seed_genome->transfer_to(corpus_sets->get_input_parameter_names(), corpus_sets->get_output_parameter_names(), transfer_learning_version, epigenetic_weights, min_recurrent_depth, max_recurrent_depth);
    }

    bool start_filled = false;
    if (genome_file_name != "") {
        get_argument(arguments, "--start_filled", false, start_filled);
    }

    examm = new EXAMM(population_size, number_islands, max_genomes, extinction_event_generation_number, islands_to_exterminate, island_ranking_method,
            repopulation_method, repopulation_mutations,
            repeat_extinction, epochs_acc_freq, speciation_method,
            species_threshold, fitness_threshold,
            neat_c1, neat_c2, neat_c3,
            corpus_sets->get_input_parameter_names(),
            corpus_sets->get_output_parameter_names(),
            corpus_sets->get_normalize_type(),
            corpus_sets->get_normalize_mins(),
            corpus_sets->get_normalize_maxs(),
            corpus_sets->get_normalize_avgs(),
            corpus_sets->get_normalize_std_devs(),
            weight_initialize, weight_inheritance, mutated_component_weight,
            bp_iterations, learning_rate,
            use_high_threshold, high_threshold,
            use_low_threshold, low_threshold,
            use_dropout, dropout_probability,
            min_recurrent_depth, max_recurrent_depth,
            use_regression,
            output_directory,
            seed_genome,
            start_filled);

    if (possible_node_types.size() > 0)  {
        examm->set_possible_node_types(possible_node_types);
    }

    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(examm_thread, i) );
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    Log::info("completed!\n");
    Log::release_id("main");

    return 0;
}
