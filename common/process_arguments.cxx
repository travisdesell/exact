#include <string>
using std::string;

#include <vector>
using std::vector;

#include "process_arguments.hxx"
#include "rnn/generate_nn.hxx"

EXAMM* generate_examm_from_arguments(
    const vector<string>& arguments, TimeSeriesSets* time_series_sets, WeightRules* weight_rules,
    RNN_Genome* seed_genome
) {
    Log::info("Getting arguments for EXAMM\n");
    int32_t island_size;
    get_argument(arguments, "--island_size", true, island_size);
    int32_t number_islands;
    get_argument(arguments, "--number_islands", true, number_islands);
    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);
    string output_directory = "";
    get_argument(arguments, "--output_directory", false, output_directory);
    vector<string> possible_node_types;
    get_argument_vector(arguments, "--possible_node_types", false, possible_node_types);
    string save_genome_option = "all_best_genomes";
    get_argument(arguments, "--save_genome_option", false, save_genome_option);

    Log::info(
        "Setting up examm with %d islands, island size %d, and max_genome %d\n", number_islands, island_size,
        max_genomes
    );

    // random_sequence_length = argument_exists(arguments, "--random_sequence_length");
    // get_argument(arguments, "--sequence_length_lower_bound", false, sequence_length_lower_bound);
    // get_argument(arguments, "--sequence_length_upper_bound", false, sequence_length_upper_bound);

    vector<string> dnas_node_type_strings;
    get_argument_vector(arguments, "--dnas_node_types", false, dnas_node_type_strings);
    if (dnas_node_type_strings.size() != 0) {
        dnas_node_types.clear();
        for (auto node_type : dnas_node_type_strings) {
            dnas_node_types.push_back(node_type_from_string(node_type));
        }

        Log::info("Using following node types for dnas: ");
        for (auto s : dnas_node_type_strings) {
            Log::info_no_header("%s", s.c_str());
        }
        Log::info_no_header("\n");
    }

    GenomeProperty* genome_property = new GenomeProperty();
    genome_property->generate_genome_property_from_arguments(arguments);
    genome_property->get_time_series_parameters(time_series_sets);

    SpeciationStrategy* speciation_strategy = generate_speciation_strategy_from_arguments(arguments, seed_genome);

    EXAMM* examm = new EXAMM(
        island_size, number_islands, max_genomes, speciation_strategy, weight_rules, genome_property, output_directory,
        save_genome_option
    );
    if (possible_node_types.size() > 0) {
        examm->set_possible_node_types(possible_node_types);
    }

    return examm;
}

SpeciationStrategy* generate_speciation_strategy_from_arguments(
    const vector<string>& arguments, RNN_Genome* seed_genome
) {
    SpeciationStrategy* speciation_strategy = NULL;
    string speciation_method = "";
    get_argument(arguments, "--speciation_method", false, speciation_method);

    if (is_island_strategy(speciation_method)) {
        Log::info("Using Island speciation strategy\n");
        speciation_strategy = generate_island_speciation_strategy_from_arguments(arguments, seed_genome);
    } else if (is_neat_strategy(speciation_method)) {
        Log::info("Using Neat speciation strategy\n");
        speciation_strategy = generate_neat_speciation_strategy_from_arguments(arguments, seed_genome);
    } else {
        Log::fatal("Wrong speciation strategy method %s\n", speciation_method.c_str());
        exit(1);
    }
    return speciation_strategy;
}

IslandSpeciationStrategy* generate_island_speciation_strategy_from_arguments(
    const vector<string>& arguments, RNN_Genome* seed_genome
) {
    int32_t island_size;
    get_argument(arguments, "--island_size", true, island_size);
    int32_t number_islands;
    get_argument(arguments, "--number_islands", true, number_islands);
    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);
    int32_t extinction_event_generation_number = 0;
    get_argument(arguments, "--extinction_event_generation_number", false, extinction_event_generation_number);
    int32_t islands_to_exterminate = 0;
    get_argument(arguments, "--islands_to_exterminate", false, islands_to_exterminate);
    string island_ranking_method = "";
    get_argument(arguments, "--island_ranking_method", false, island_ranking_method);
    string repopulation_method = "";
    get_argument(arguments, "--repopulation_method", false, repopulation_method);
    int32_t num_mutations = 1;
    get_argument(arguments, "--num_mutations", false, num_mutations);

    double mutation_rate = 0.70, intra_island_co_rate = 0.20, inter_island_co_rate = 0.10;

    if (number_islands == 1) {
        inter_island_co_rate = 0.0;
        intra_island_co_rate = 0.30;
    }

    bool repeat_extinction = argument_exists(arguments, "--repeat_extinction");
    string transfer_learning_version = "";
    bool transfer_learning = get_argument(arguments, "--transfer_learning_version", false, transfer_learning_version);
    int32_t seed_stirs = 0;
    get_argument(arguments, "--seed_stirs", false, seed_stirs);
    bool start_filled = argument_exists(arguments, "--start_filled");
    bool tl_epigenetic_weights = argument_exists(arguments, "--tl_epigenetic_weights");

    IslandSpeciationStrategy* island_strategy = new IslandSpeciationStrategy(
        number_islands, island_size, mutation_rate, intra_island_co_rate, inter_island_co_rate, seed_genome,
        island_ranking_method, repopulation_method, extinction_event_generation_number, num_mutations,
        islands_to_exterminate, max_genomes, repeat_extinction, start_filled, transfer_learning,
        transfer_learning_version, seed_stirs, tl_epigenetic_weights
    );

    return island_strategy;
}

NeatSpeciationStrategy* generate_neat_speciation_strategy_from_arguments(
    const vector<string>& arguments, RNN_Genome* seed_genome
) {
    // bool seed_genome_was_minimal = true;
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
    double mutation_rate = 0.70, intra_island_co_rate = 0.20, inter_island_co_rate = 0.10;

    NeatSpeciationStrategy* neat_strategy = new NeatSpeciationStrategy(
        mutation_rate, intra_island_co_rate, inter_island_co_rate, seed_genome, species_threshold, fitness_threshold,
        neat_c1, neat_c2, neat_c3
    );
    return neat_strategy;
}

bool is_island_strategy(string strategy_name) {
    if (strategy_name.compare("") == 0 || strategy_name.compare("island") == 0) {
        return true;
    } else {
        return false;
    }
}

bool is_neat_strategy(string strategy_name) {
    if (strategy_name.compare("neat") == 0) {
        return true;
    } else {
        return false;
    }
}

void write_time_series_to_file(const vector<string>& arguments, TimeSeriesSets* time_series_sets) {
    if (argument_exists(arguments, "--write_time_series")) {
        string base_filename;
        get_argument(arguments, "--write_time_series", true, base_filename);
        time_series_sets->write_time_series_sets(base_filename);
    }
}

void get_train_validation_data(
    const vector<string>& arguments, TimeSeriesSets* time_series_sets, vector<vector<vector<double> > >& train_inputs,
    vector<vector<vector<double> > >& train_outputs, vector<vector<vector<double> > >& validation_inputs,
    vector<vector<vector<double> > >& validation_outputs
) {
    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_training_series(time_offset, train_inputs, train_outputs);
    time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    int32_t sequence_length = 0;
    if (get_argument(arguments, "--test_sequence_length", false, sequence_length)) {
        Log::info("Slicing input training data with time sequence length: %d\n", sequence_length);
        slice_input_data(train_inputs, train_outputs, sequence_length);
    }

    int32_t validation_sequence_length = 0;
    if (get_argument(arguments, "--validation_sequence_length", false, validation_sequence_length)) {
        Log::info("Slicing input validation data with time sequence length: %d\n", validation_sequence_length);
        slice_input_data(validation_inputs, validation_outputs, validation_sequence_length);
    }

    Log::info("Generating time series data finished! \n");
}

void slice_input_data(
    vector<vector<vector<double> > >& inputs, vector<vector<vector<double> > >& outputs, int32_t sequence_length
) {
    vector<vector<vector<double> > > sliced_inputs;
    vector<vector<vector<double> > > sliced_outputs;
    for (int32_t n = 0; n < (int32_t) inputs.size(); n++) {
        int32_t num_row = (int32_t) inputs[n][0].size();
        int32_t num_inputs = (int32_t) inputs[n].size();
        int32_t num_outputs = (int32_t) outputs[n].size();
        int32_t current_row = 0;
        while (current_row + sequence_length <= num_row) {
            vector<vector<double> > current_time_series_input;   // <each parameter <time series values>>
            vector<vector<double> > current_time_series_output;  // <each parameter <time series values>>
            current_time_series_input = slice_time_series(current_row, sequence_length, num_inputs, inputs[n]);
            current_time_series_output = slice_time_series(current_row, sequence_length, num_outputs, outputs[n]);
            sliced_inputs.push_back(current_time_series_input);
            sliced_outputs.push_back(current_time_series_output);
            current_row = current_row + sequence_length;
        }
        Log::info("Before slicing, original time series %d has %d parameters, and %d length\n", n, num_inputs, num_row);
    }

    inputs.clear();
    outputs.clear();

    inputs.assign(sliced_inputs.begin(), sliced_inputs.end());
    outputs.assign(sliced_outputs.begin(), sliced_outputs.end());
    Log::info(
        "After slicing, sliced training input data has %d sets, and %d parameters and length %d \n", inputs.size(),
        inputs[0].size(), inputs[0][0].size()
    );
    Log::info(
        "After slicing, sliced training output data has %d sets, and %d parameters and length %d \n", outputs.size(),
        outputs[0].size(), outputs[0][0].size()
    );
}

vector<vector<double> > slice_time_series(
    int32_t start_index, int32_t sequence_length, int32_t num_parameter, const vector<vector<double> >& time_series
) {
    vector<vector<double> > current_time_series;
    for (int32_t j = 0; j < num_parameter; j++) {
        vector<double> current_parameter_slice;
        current_parameter_slice.assign(
            time_series[j].begin() + start_index, time_series[j].begin() + start_index + sequence_length
        );
        current_time_series.push_back(current_parameter_slice);
    }
    return current_time_series;
}
