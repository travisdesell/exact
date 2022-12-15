#include <string>
using std::string;

#include <vector>
using std::vector;

#include "process_arguments.hxx"


SpeciationStrategy* generate_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome) {
    SpeciationStrategy *speciation_strategy = NULL;
    string speciation_method = "";
    get_argument(arguments, "--speciation_method", false, speciation_method);

    if (is_island_strategy(speciation_method)) {
        Log::info("Using Island speciation strategy\n");
        speciation_strategy= generate_island_speciation_strategy_from_arguments(arguments, seed_genome);
    }
    else if (is_neat_strategy(speciation_method)) {
        Log::info("Using Neat speciation strategy\n");
        speciation_strategy = generate_neat_speciation_strategy_from_arguments(arguments, seed_genome);
    }
    else {
        Log::fatal("Wrong speciation strategy method %s\n", speciation_method.c_str());
        exit(1);
    }
    return speciation_strategy;
}

IslandSpeciationStrategy* generate_island_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome) {

    int32_t island_size;
    get_argument(arguments, "--population_size", true, island_size);

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

    // RNN_Genome *seed_genome = NULL;
    bool seed_genome_was_minimal = true;
    
    bool start_filled = false;
    get_argument(arguments, "--start_filled", false, start_filled);

    // if (start_filled) {
    //     // Only used if start_filled is enabled
    //     function<void (RNN_Genome *)> apply_stir_mutations = [this](RNN_Genome *genome) {
    //         RNN_Genome *copy = genome->copy();
    //         this->mutate(num_mutations, copy);
    //         return copy;
    //     };

    //     speciation_strategy = new IslandSpeciationStrategy(
    //             number_islands, island_size, mutation_rate, intra_island_co_rate, inter_island_co_rate,
    //             seed_genome, island_ranking_method, repopulation_method, extinction_event_generation_number, num_mutations, islands_to_exterminate, seed_genome_was_minimal, apply_stir_mutations);
    // }
    IslandSpeciationStrategy *island_strategy = new IslandSpeciationStrategy(
        number_islands, island_size, mutation_rate, intra_island_co_rate, inter_island_co_rate,
        seed_genome, island_ranking_method, repopulation_method, extinction_event_generation_number, 
        num_mutations, islands_to_exterminate, max_genomes, repeat_extinction, seed_genome_was_minimal);
    
    
    return island_strategy;
}

NeatSpeciationStrategy* generate_neat_speciation_strategy_from_arguments(const vector<string> &arguments, RNN_Genome *seed_genome) {

    bool seed_genome_was_minimal = true;
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

    NeatSpeciationStrategy *neat_strategy = new NeatSpeciationStrategy(mutation_rate, intra_island_co_rate, inter_island_co_rate, seed_genome, species_threshold, fitness_threshold, neat_c1, neat_c2, neat_c3);
    return neat_strategy;
}

bool is_island_strategy (string strategy_name) {
    if (strategy_name.compare("") == 0 || strategy_name.compare("island") == 0) return true;
    else return false;
}

bool is_neat_strategy (string strategy_name) {
    if (strategy_name.compare("neat") == 0) return true;
    else return false;
}