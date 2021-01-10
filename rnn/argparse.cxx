    TimeSeriesSets *time_series_sets = NULL;
    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
    time_series_sets->export_test_series(time_offset, validation_inputs, validation_outputs);

    int number_inputs = time_series_sets->get_number_inputs();
    int number_outputs = time_series_sets->get_number_outputs();

    Log::debug("number_inputs: %d, number_outputs: %d\n", number_inputs, number_outputs);

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
    bool use_regression = true; //time series will always use regression

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

        string transfer_learning_version;
        get_argument(arguments, "--transfer_learning_version", true, transfer_learning_version);

        bool epigenetic_weights = argument_exists(arguments, "--epigenetic_weights");

        seed_genome->transfer_to(time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), transfer_learning_version, epigenetic_weights, min_recurrent_depth, max_recurrent_depth);
    }


    bool start_filled = false;
    get_argument(arguments, "--start_filled", false, start_filled);

