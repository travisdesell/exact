#if !defined WORK_HXX || !defined DATASET_META_HXX || !defined TRAINING_PARAMETERS_HXX || !defined GENOME_OPERATORS_HXX
#error "You must import rnn/genome_operators.hxx, rnn/work/work.hxx, time_series/dataset_meta.hxx, and rnn/training_parameters.hxx"
#endif

#ifndef EXAMM_NLP
    
    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_from_arguments(arguments);
    int32_t time_offset = time_offset;
    get_argument(arguments, "--time_offset", true, time_offset);

    TimeSeriesSets *dataset = time_series_sets;

#else

    Corpus* corpus_sets = Corpus::generate_from_arguments(arguments);
    Corpus *dataset = corpus_sets;

    int32_t word_offset = 1;
    get_argument(arguments, "--word_offset", true, word_offset);
    int32_t time_offset = word_offset;

    Log::info("exported word series.\n");

#endif

#ifdef EXAMM_MT

    int32_t number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);
    int32_t number_workers = number_threads;

#else

    int worker_id, number_workers;
    MPI_Comm_rank(MPI_COMM_WORLD, &worker_id);
    // This is actually n workers + 1, since master is counted.
    // This function gets the n of processes in total
    MPI_Comm_size(MPI_COMM_WORLD, &number_workers);
    number_workers -= 1;

#endif

    int number_inputs = dataset->get_number_inputs();
    int number_outputs = dataset->get_number_outputs();

    dataset->export_training_series(time_offset, training_inputs, training_outputs);
    dataset->export_test_series(time_offset, validation_inputs, validation_outputs);

    Log::debug("number_inputs: %d, number_outputs: %d\n", number_inputs, number_outputs);

    int32_t population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int32_t number_islands;
    get_argument(arguments, "--number_islands", true, number_islands);

    int32_t max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    string speciation_method = "";
    get_argument(arguments, "--speciation_method", false, speciation_method);

    int32_t extinction_event_generation_number = max_genomes + 1;
    get_argument(arguments, "--extinction_event_generation_number", false, extinction_event_generation_number);
  
    int32_t islands_to_exterminate = 0;
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

    // The stress test is used to see what throughput the EXAMM master process can
    // give when the workers are as fast as possible. This would mean all workers
    // do is evaluate the fitness of the genomes they receive (and perform mutation
    // or crossover)
#ifndef EXAMM_MPI_STRESS_TEST
    int32_t bp_iterations;
    get_argument(arguments, "--bp_iterations", true, bp_iterations);
#else
    int32_t bp_iterations = 0;
#endif

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
    Log::info("WI=%d\n", weight_initialize);
    
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

        seed_genome->transfer_to(dataset->get_input_parameter_names(), dataset->get_output_parameter_names(), transfer_learning_version, epigenetic_weights, min_recurrent_depth, max_recurrent_depth);
    }


    bool start_filled = false;
    get_argument(arguments, "--start_filled", false, start_filled);
    
    DatasetMeta dataset_meta = dataset->get_dataset_meta();
    TrainingParameters training_parameters(
            bp_iterations,
            high_threshold,
            low_threshold,
            learning_rate,
            dropout_probability,
            use_high_threshold,
            use_low_threshold,
            use_regression,
            use_dropout);
    GenomeOperators genome_operators(
            number_workers,
            0,
            number_inputs,
            number_outputs,
            0, 0,
            min_recurrent_depth,
            max_recurrent_depth,
            weight_initialize,
            weight_inheritance,
            mutated_component_weight,
            dataset_meta,
            training_parameters,
            possible_node_types);
    function<EXAMM *()> make_examm = [&]() {
            EXAMM *examm = new EXAMM(
                    population_size,
                    number_islands,
                    max_genomes,
                    extinction_event_generation_number, 
                    islands_to_exterminate, 
                    island_ranking_method,
                    repopulation_method, 
                    repopulation_mutations, 
                    repeat_extinction,
                    speciation_method,
                    species_threshold, 
                    fitness_threshold,
                    neat_c1, neat_c2, neat_c3,
                    weight_initialize, weight_inheritance, mutated_component_weight,
                    output_directory,
                    genome_operators,
                    dataset_meta,
                    training_parameters,
                    seed_genome,
                    start_filled);
            return examm;
    };

    int32_t edge_innovation_count;
    int32_t node_innovation_count;
    
    function<void (EXAMM *)> set_innovation_counts = 
        [&](EXAMM *examm) {
            RNN_Genome *genome = examm->get_seed_genome();
            edge_innovation_count = genome->get_max_edge_innovation_number() + 1;
            node_innovation_count = genome->get_max_node_innovation_number() + 1;
        };

    function<GenomeOperators (int32_t, int32_t, int32_t)> make_genome_operators =
        [&](int32_t worker_id, int32_t edge_innovation_count, int32_t node_innovation_count) {
            GenomeOperators go(
                number_workers,
                worker_id,
                number_inputs,
                number_outputs,
                edge_innovation_count, node_innovation_count,
                min_recurrent_depth,
                max_recurrent_depth,
                weight_initialize,
                weight_inheritance,
                mutated_component_weight,
                dataset_meta,
                training_parameters,
                possible_node_types);
            return go;
        };
