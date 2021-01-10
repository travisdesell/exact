#include <algorithm>
using std::sort;

#include <chrono>
#include <cstring>

#include <functional>
using std::bind;
using std::function;

#include <fstream>
using std::ofstream;

#include <iomanip>
using std::setw;
using std::setprecision;

#include <iostream>
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;
using std::uniform_int_distribution;

#include <string>
using std::string;
using std::to_string;

#include "examm.hxx"
#include "rnn_genome.hxx"
#include "generate_nn.hxx"
#include "speciation_strategy.hxx"
#include "island_speciation_strategy.hxx"
#include "neat_speciation_strategy.hxx"

//INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING
#include "rnn.hxx"
#include "rnn_node.hxx"
#include "lstm_node.hxx"
#include "gru_node.hxx"
#include "delta_node.hxx"
#include "ugrnn_node.hxx"
#include "mgu_node.hxx"

#include "common/files.hxx"
#include "common/log.hxx"


EXAMM::~EXAMM() {
    RNN_Genome *genome;
    for (uint32_t i = 0; i < genomes.size(); i++) {
        while (genomes[i].size() > 0) {
            genome = genomes[i].back();
            genomes[i].pop_back();
            delete genome;
        }
    }
}

EXAMM::EXAMM(
        int32_t _population_size, 
        int32_t _number_islands, 
        int32_t _max_genomes, 
        int32_t _extinction_event_generation_number,
        int32_t islands_to_exterminate,
        string _island_ranking_method, 
        string _repopulation_method, 
        int32_t _repopulation_mutations,
        bool _repeat_extinction,
        string _speciation_method,
        double _species_threshold, 
        double _fitness_threshold,
        double _neat_c1, 
        double _neat_c2, 
        double _neat_c3,
        WeightType _weight_initialize,
        WeightType _weight_inheritance,
        WeightType _mutated_component_weight,
        string _output_directory,
        RNN_Genome *seed_genome,
        bool _start_filled) :
                        population_size(_population_size),
                        number_islands(_number_islands),
                        max_genomes(_max_genomes),
                        extinction_event_generation_number(_extinction_event_generation_number),
                        island_ranking_method(_island_ranking_method), 
                        speciation_method(_speciation_method),
                        repopulation_method(_repopulation_method), 
                        repopulation_mutations(_repopulation_mutations),
                        repeat_extinction(_repeat_extinction),
                        output_directory(_output_directory),
                        weight_initialize(_weight_initialize),
                        weight_inheritance(_weight_inheritance),
                        mutated_component_weight(_mutated_component_weight),
                        start_filled(_start_filled) {
    total_bp_epochs = 0;

    //update to now have islands of genomes
    genomes = vector< vector<RNN_Genome*> >(number_islands);

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    //rng_crossover_weight = uniform_real_distribution<double>(0.0, 0.0);
    //rng_crossover_weight = uniform_real_distribution<double>(-0.10, 0.1);
    rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);
    //rng_crossover_weight = uniform_real_distribution<double>(0.45, 0.55);

    check_weight_initialize_validity();

    Log::info("weight initialize: %s\n", WEIGHT_TYPES_STRING[weight_initialize].c_str());
    Log::info("weight inheritance: %s \n", WEIGHT_TYPES_STRING[weight_inheritance].c_str());
    Log::info("mutated component weight: %s\n", WEIGHT_TYPES_STRING[mutated_component_weight].c_str());

    Log::info("Speciation method is: \"%s\" (Default is the island-based speciation strategy).\n", speciation_method.c_str());
    if (speciation_method.compare("island") == 0 || speciation_method.compare("") == 0) {
        //generate a minimal feed foward network as the seed genome
        
        bool seed_genome_was_minimal = false;
        if (seed_genome == NULL) {
            seed_genome_was_minimal = true;
            seed_genome = create_ff(input_parameter_names, 0, 0, output_parameter_names, 0, weight_initialize, weight_inheritance, mutated_component_weight);
            seed_genome->initialize_randomly();
        } //otherwise the seed genome was passed into EXAMM

        //make sure we don't duplicate node or edge innovation numbers
        edge_innovation_count = seed_genome->get_max_edge_innovation_count() + 1;
        node_innovation_count = seed_genome->get_max_node_innovation_count() + 1;

        seed_genome->set_generated_by("initial");

        //insert a copy of it into the population so
        //additional requests can mutate it

        seed_genome->enable_use_regression(use_regression);
        seed_genome->best_validation_mse = EXAMM_MAX_DOUBLE;

        seed_genome->best_validation_mse = EXAMM_MAX_DOUBLE;
        seed_genome->best_validation_mae = EXAMM_MAX_DOUBLE;
        //seed_genome->best_parameters.clear();
        
        double mutation_rate = 0.70, intra_island_co_rate = 0.20, inter_island_co_rate = 0.10;
        
        if (number_islands == 1) {
            inter_island_co_rate = 0.0;
            intra_island_co_rate = 0.30;
        }
       
        // Only difference here is that the apply_stir_mutations lambda is passed if the island is supposed to start filled.
        if (start_filled) {
            // Only used if start_filled is enabled
            function<void (RNN_Genome *)> apply_stir_mutations = [this](RNN_Genome *genome) {
                RNN_Genome *copy = genome->copy();
                this->mutate(repopulation_mutations, copy);
                return copy;
            };

            speciation_strategy = new IslandSpeciationStrategy(
                    number_islands, population_size, mutation_rate, intra_island_co_rate, inter_island_co_rate,
                    seed_genome, island_ranking_method, repopulation_method, extinction_event_generation_number, repopulation_mutations, islands_to_exterminate, seed_genome_was_minimal, apply_stir_mutations);
        } else {
            speciation_strategy = new IslandSpeciationStrategy(
                    number_islands, population_size, mutation_rate, intra_island_co_rate, inter_island_co_rate,
                    seed_genome, island_ranking_method, repopulation_method, extinction_event_generation_number, repopulation_mutations, islands_to_exterminate, max_genomes, repeat_extinction, seed_genome_was_minimal);
        }
    }
    else if (speciation_method.compare("neat") == 0) {
        
        bool seed_genome_was_minimal = false;
        if (seed_genome == NULL) {
            seed_genome_was_minimal = true;
            seed_genome = create_ff(input_parameter_names, 0, 0, output_parameter_names, 0, weight_initialize, weight_inheritance, mutated_component_weight);
            seed_genome->initialize_randomly();
        } //otherwise the seed genome was passed into EXAMM

        //make sure we don't duplicate node or edge innovation numbers
        edge_innovation_count = seed_genome->get_max_edge_innovation_count() + 1;
        node_innovation_count = seed_genome->get_max_node_innovation_count() + 1;

        seed_genome->set_generated_by("initial");

        //insert a copy of it into the population so
        //additional requests can mutate it

        seed_genome->best_validation_mse = EXAMM_MAX_DOUBLE;
        seed_genome->best_validation_mae = EXAMM_MAX_DOUBLE;
        //seed_genome->best_parameters.clear();
        
        double mutation_rate = 0.70, intra_island_co_rate = 0.20, inter_island_co_rate = 0.10;
        
        if (number_islands == 1) {
            inter_island_co_rate = 0.0;
            intra_island_co_rate = 0.30;
        }
        // no transfer learning for NEAT
        speciation_strategy = new NeatSpeciationStrategy(mutation_rate, intra_island_co_rate, inter_island_co_rate, seed_genome,  max_genomes, species_threshold, fitness_threshold, neat_c1, neat_c2, neat_c3, generator);
        
    }

    if (output_directory != "") {
        mkpath(output_directory.c_str(), 0777);
        log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
        (*log_file) << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";
        //memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";

        (*log_file) << speciation_strategy->get_strategy_information_headers();
        //(memory_log) << speciation_strategy->get_strategy_information_headers();

        (*log_file) << endl;
        //memory_log << endl;

        op_log_file = new ofstream(output_directory + "/op_log.csv");
        
        op_log_ordering = {
            "genomes",
            "crossover",
            "island_crossover",
            "clone",
            "add_edge",
            "add_recurrent_edge",
            "enable_edge",
            "disable_edge",
            "enable_node",
            "disable_node",
        };

        // To get data about these ops without respect to node type,
        // you'll have to calculate the sum, e.g. sum split_node(x) for all node types x
        // to get information about split_node as a whole.
        vector<string> ops_with_node_type = {
            "add_node",
            "split_node",
            "merge_node",
            "split_edge"
        };

        for (int i = 0; i < ops_with_node_type.size(); i++) {
            string op = ops_with_node_type[i];
            for (int j = 0; j < possible_node_types.size(); j++)
                op_log_ordering.push_back(op + "(" + NODE_TYPES[possible_node_types[j]] + ")");
        }

        for (int i = 0; i < op_log_ordering.size(); i++) {
            string op = op_log_ordering[i];
            (*op_log_file) << op;
            (*op_log_file) << " Generated, ";
            (*op_log_file) << op;
            (*op_log_file) << " Inserted, ";
            
            inserted_counts[op] = 0;
            generated_counts[op] = 0;
        }

        map<string, int>::iterator it;

        (*op_log_file) << endl;

    } else {
        log_file = NULL;
        op_log_file = NULL;
    }

    startClock = std::chrono::system_clock::now();
}

void EXAMM::print() {
    if (Log::at_level(Log::INFO)) {
        speciation_strategy->print();
    }
}

void EXAMM::update_log() {
    if (log_file != NULL) {

        //make sure the log file is still good
        if (!log_file->good()) {
            log_file->close();
            delete log_file;

            string output_file = output_directory + "/fitness_log.csv";
            log_file = new ofstream(output_file, std::ios_base::app);

            if (!log_file->is_open()) {
                Log::error("could not open EXAMM output log: '%s'\n", output_file.c_str());
                exit(1);
            }
        }

        if (!op_log_file->good()) {
            op_log_file->close();
            delete op_log_file;

            string output_file = output_directory + "/op_log.csv";
            op_log_file = new ofstream(output_file, std::ios_base::app);

            if (!op_log_file->is_open()) {
                Log::error("could not open EXAMM output log: '%s'\n", output_file.c_str());
                exit(1);
            }
        }

        RNN_Genome *best_genome = get_best_genome();
        if (best_genome == NULL) {
            best_genome = speciation_strategy->get_global_best_genome();
        }

        std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

        (*log_file) << speciation_strategy->get_inserted_genomes()
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count()
            << speciation_strategy->get_strategy_information_values()
            << endl;

        /*
        memory_log << speciation_strategy->get_inserted_genomes()
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count()
            << speciation_strategy->get_strategy_information_values()
            << endl;
        */

        for (int i = 0; i < op_log_ordering.size(); i++) {
            string op = op_log_ordering[i];
            (*op_log_file) << generated_counts[op] << ", " << inserted_counts[op]  << ", ";
        }

        (*op_log_file) << endl;

    }
}

void EXAMM::write_memory_log(string filename) {
    ofstream log_file(filename);
    log_file << memory_log.str();
    log_file.close();
}

void EXAMM::set_possible_node_types(vector<string> possible_node_type_strings) {
    possible_node_types.clear();

    for (uint32_t i = 0; i < possible_node_type_strings.size(); i++) {
        string node_type_s = possible_node_type_strings[i];

        bool found = false;

        for (int32_t j = 0; j < NUMBER_NODE_TYPES; j++) {
            if (NODE_TYPES[j].compare(node_type_s) == 0) {
                found = true;
                possible_node_types.push_back(j);
            }
        }

        if (!found) {
            Log::error("unknown node type: '%s'\n", node_type_s.c_str());
            exit(1);
        }
    }

    if (possible_node_types.size() == 0) {
        Log::fatal("failed to specify any node types: there must be at least one node type specified");
        exit(1);
    }
}

string EXAMM::get_output_directory() const {
    return output_directory;
}

double EXAMM::get_best_fitness() {
    return speciation_strategy->get_best_fitness();
}

double EXAMM::get_worst_fitness() {
    return speciation_strategy->get_worst_fitness();
}

RNN_Genome* EXAMM::get_best_genome() {
    return speciation_strategy->get_best_genome();
}

RNN_Genome* EXAMM::get_worst_genome() {
    return speciation_strategy->get_worst_genome();
}

//this will insert a COPY, original needs to be deleted
bool EXAMM::insert_genome(RNN_Genome* genome) {
    total_bp_epochs += genome->get_bp_iterations();

    // Log::info("genomes evaluated: %10d , attempting to insert: %s\n", (speciation_strategy->get_inserted_genomes() + 1), parse_fitness(genome->get_fitness()).c_str());

    if (!genome->sanity_check()) {
        Log::error("genome failed sanity check on insert!\n");
        exit(1);
    }

    //updates EXAMM's mapping of which genomes have been generated by what
    genome->update_generation_map(generated_from_map);

    int32_t insert_position = speciation_strategy->insert_genome(genome);
    //write this genome to disk if it was a new best found genome
    if (insert_position == 0) {
        genome->normalize_type = normalize_type;
        genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(genome->get_generation_id()) + ".gv");
        genome->write_to_file(output_directory + "/rnn_genome_" + to_string(genome->get_generation_id()) + ".bin");
    }

    // Name of the operator
    const map<string, int> *generated_by_map = genome->get_generated_by_map();

    for (auto it = generated_by_map->begin(); it != generated_by_map->end(); it++) {
        string generated_by = it->first;

        if (generated_counts.count(generated_by) > 0) {
            generated_counts["nenomes"] += 1;

            // Add one to the number of genomes generated by this operator
            generated_counts[generated_by] += 1;

            // If it was inserted add one to the number of genomes generated AND inserted by this operator
            if (insert_position >= 0) {
                inserted_counts["genomes"] += 1;
                inserted_counts[generated_by] += 1;
            }
        } else {
            if (generated_by != "initial")
                Log::error("unrecognized generated_by string '%s'\n", generated_by.c_str());
        }
    }

    speciation_strategy->print();
    update_log();

    return insert_position >= 0;
}

Work* EXAMM::generate_work() {
    if (speciation_strategy->get_inserted_genomes() > max_genomes)
        return new TerminateWork();

    return speciation_strategy->generate_genome(rng_0_1, generator);
    
    // function<void (int32_t, RNN_Genome*)> mutate_function =
    //     [=](int32_t max_mutations, RNN_Genome *genome) {
    //         this->mutate(max_mutations, genome);
    //     };

    // function<RNN_Genome* (RNN_Genome*, RNN_Genome*)> crossover_function =
    //     [=](RNN_Genome *parent1, RNN_Genome *parent2) {
    //         return this->crossover(parent1, parent2);
    //     };

    // genome->set_parameter_names(input_parameter_names, output_parameter_names);
    // genome->set_normalize_bounds(normalize_type, normalize_mins, normalize_maxs, normalize_avgs, normalize_std_devs);
    // genome->set_bp_iterations(bp_iterations);
    // genome->set_learning_rate(learning_rate);
    // genome->enable_use_regression(use_regression);

    // if (use_high_threshold) genome->enable_high_threshold(high_threshold);
    // if (use_low_threshold) genome->enable_low_threshold(low_threshold);
    // if (use_dropout) genome->enable_dropout(dropout_probability);

    // if (!epigenetic_weights) genome->initialize_randomly();

    //this is just a sanity check, can most likely comment out (checking to see
    //if all the paramemters are sane)
    // Log::debug("getting mu/sigma after random initialization of copy!\n");
    // double _mu, _sigma;
    // genome->get_mu_sigma(genome->best_parameters, _mu, _sigma);

}

int EXAMM::get_random_node_type() {
    return possible_node_types[rng_0_1(generator) * possible_node_types.size()];
}

void EXAMM::attempt_node_insert(vector<RNN_Node_Interface*> &child_nodes, const RNN_Node_Interface *node, const vector<double> &new_weights) {
    for (int32_t i = 0; i < (int32_t)child_nodes.size(); i++) {
        if (child_nodes[i]->get_innovation_number() == node->get_innovation_number()) return;
    }

    RNN_Node_Interface *node_copy = node->copy();
    node_copy->set_weights(new_weights);

    child_nodes.insert( upper_bound(child_nodes.begin(), child_nodes.end(), node_copy, sort_RNN_Nodes_by_depth()), node_copy);
}

void EXAMM::attempt_edge_insert(vector<RNN_Edge*> &child_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Edge *edge, RNN_Edge *second_edge, bool set_enabled) {
    for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
        if (child_edges[i]->get_innovation_number() == edge->get_innovation_number()) {
            Log::fatal("ERROR in crossover! trying to push an edge with innovation_number: %d and it already exists in the vector!\n", edge->get_innovation_number());

            Log::fatal("vector innovation numbers: ");
            for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                Log::fatal("\t%d", child_edges[i]->get_innovation_number());
            }

            Log::fatal("This should never happen!\n");
            exit(1);

            return;
        } else if (child_edges[i]->get_input_innovation_number() == edge->get_input_innovation_number() &&
                child_edges[i]->get_output_innovation_number() == edge->get_output_innovation_number()) {

            Log::debug("Not inserting edge in crossover operation as there was already an edge with the same input and output innovation numbers!\n");
            return;
        }
    }

    vector<double> new_input_weights, new_output_weights;
    double new_weight = 0.0;
    if (second_edge != NULL) {
        double crossover_value = rng_crossover_weight(generator);
        new_weight = crossover_value * (second_edge->weight - edge->weight) + edge->weight;

        Log::trace("EDGE WEIGHT CROSSOVER :: better: %lf, worse: %lf, crossover_value: %lf, new_weight: %lf\n", edge->weight, second_edge->weight, crossover_value, new_weight);

        vector<double> input_weights1, input_weights2, output_weights1, output_weights2;
        edge->get_input_node()->get_weights(input_weights1);
        edge->get_output_node()->get_weights(output_weights1);

        second_edge->get_input_node()->get_weights(input_weights2);
        second_edge->get_output_node()->get_weights(output_weights2);

        new_input_weights.resize(input_weights1.size());
        new_output_weights.resize(output_weights1.size());

        //can check to see if input weights lengths are same
        //can check to see if output weights lengths are same

        for (int32_t i = 0; i < (int32_t)new_input_weights.size(); i++) {
            new_input_weights[i] = crossover_value * (input_weights2[i] - input_weights1[i]) + input_weights1[i];
            Log::trace("\tnew input weights[%d]: %lf\n", i, new_input_weights[i]);
        }

        for (int32_t i = 0; i < (int32_t)new_output_weights.size(); i++) {
            new_output_weights[i] = crossover_value * (output_weights2[i] - output_weights1[i]) + output_weights1[i];
            Log::trace("\tnew output weights[%d]: %lf\n", i, new_output_weights[i]);
        }

    } else {
        new_weight = edge->weight;
        edge->get_input_node()->get_weights(new_input_weights);
        edge->get_output_node()->get_weights(new_output_weights);
    }

    attempt_node_insert(child_nodes, edge->get_input_node(), new_input_weights);
    attempt_node_insert(child_nodes, edge->get_output_node(), new_output_weights);

    RNN_Edge *edge_copy = edge->copy(child_nodes);

    edge_copy->enabled = set_enabled;
    edge_copy->weight = new_weight;

    //edges have already been copied
    child_edges.insert( upper_bound(child_edges.begin(), child_edges.end(), edge_copy, sort_RNN_Edges_by_depth()), edge_copy);
}

void EXAMM::attempt_recurrent_edge_insert(vector<RNN_Recurrent_Edge*> &child_recurrent_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Recurrent_Edge *recurrent_edge, RNN_Recurrent_Edge *second_edge, bool set_enabled) {
    for (int32_t i = 0; i < (int32_t)child_recurrent_edges.size(); i++) {
        if (child_recurrent_edges[i]->get_innovation_number() == recurrent_edge->get_innovation_number()) {
            Log::fatal("ERROR in crossover! trying to push an recurrent_edge with innovation_number: %d  and it already exists in the vector!\n", recurrent_edge->get_innovation_number());
            Log::fatal("vector innovation numbers:\n");
            for (int32_t i = 0; i < (int32_t)child_recurrent_edges.size(); i++) {
                Log::fatal("\t %d", child_recurrent_edges[i]->get_innovation_number());
            }

            Log::fatal("This should never happen!\n");
            exit(1);

            return;
        } else if (child_recurrent_edges[i]->get_input_innovation_number() == recurrent_edge->get_input_innovation_number() &&
                child_recurrent_edges[i]->get_output_innovation_number() == recurrent_edge->get_output_innovation_number()) {

            Log::debug("Not inserting recurrent_edge in crossover operation as there was already an recurrent_edge with the same input and output innovation numbers!\n");
            return;
        }
    }


    vector<double> new_input_weights, new_output_weights;
    double new_weight = 0.0;
    if (second_edge != NULL) {
        double crossover_value = rng_crossover_weight(generator);
        new_weight = crossover_value * (second_edge->weight - recurrent_edge->weight) + recurrent_edge->weight;

        Log::debug("RECURRENT EDGE WEIGHT CROSSOVER :: better: %lf, worse: %lf, crossover_value: %lf, new_weight: %lf\n", recurrent_edge->weight, second_edge->weight, crossover_value, new_weight);

        vector<double> input_weights1, input_weights2, output_weights1, output_weights2;
        recurrent_edge->get_input_node()->get_weights(input_weights1);
        recurrent_edge->get_output_node()->get_weights(output_weights1);

        second_edge->get_input_node()->get_weights(input_weights2);
        second_edge->get_output_node()->get_weights(output_weights2);

        new_input_weights.resize(input_weights1.size());
        new_output_weights.resize(output_weights1.size());

        for (int32_t i = 0; i < (int32_t)new_input_weights.size(); i++) {
            new_input_weights[i] = crossover_value * (input_weights2[i] - input_weights1[i]) + input_weights1[i];
            Log::trace("\tnew input weights[%d]: %lf\n", i, new_input_weights[i]);
        }

        for (int32_t i = 0; i < (int32_t)new_output_weights.size(); i++) {
            new_output_weights[i] = crossover_value * (output_weights2[i] - output_weights1[i]) + output_weights1[i];
            Log::trace("\tnew output weights[%d]: %lf\n", i, new_output_weights[i]);
        // }

    } else {
        new_weight = recurrent_edge->weight;
        recurrent_edge->get_input_node()->get_weights(new_input_weights);
        recurrent_edge->get_output_node()->get_weights(new_output_weights);
    }

    attempt_node_insert(child_nodes, recurrent_edge->get_input_node(), new_input_weights);
    attempt_node_insert(child_nodes, recurrent_edge->get_output_node(), new_output_weights);

    RNN_Recurrent_Edge *recurrent_edge_copy = recurrent_edge->copy(child_nodes);

    recurrent_edge_copy->enabled = set_enabled;
    recurrent_edge_copy->weight = new_weight;


    //recurrent_edges have already been copied
    child_recurrent_edges.insert( upper_bound(child_recurrent_edges.begin(), child_recurrent_edges.end(), recurrent_edge_copy, sort_RNN_Recurrent_Edges_by_depth()), recurrent_edge_copy);
}

void EXAMM::check_weight_initialize_validity() {
    if (weight_initialize < 0) {
        Log::fatal("Weight initalization is set to NONE, this should not happen! \n");
        exit(1);    
    }
    if (weight_inheritance < 0) {
        Log::fatal("Weight inheritance is set to NONE, this should not happen! \n");
        exit(1);
    }
    if (mutated_component_weight < 0) {
        Log::fatal("Mutated component weight is set to NONE, this should not happen! \n");
        exit(1);    
    }
    if (weight_initialize == WeightType::LAMARCKIAN) {
        Log::fatal("Weight initialization method is set to Lamarckian! \n");
        exit(1);
    }
    if (weight_inheritance != weight_initialize && weight_inheritance != WeightType::LAMARCKIAN) {
        Log::fatal("Weight initialize is %s, weight inheritance is %s\n", WEIGHT_TYPES_STRING[weight_initialize].c_str(), WEIGHT_TYPES_STRING[weight_inheritance].c_str());
        exit(1);
    }
    if (mutated_component_weight != weight_initialize && mutated_component_weight != WeightType::LAMARCKIAN) {
        Log::fatal("Weight initialize is %s, new component weight is %s\n", WEIGHT_TYPES_STRING[weight_initialize].c_str(), WEIGHT_TYPES_STRING[mutated_component_weight].c_str());
        exit(1);
    }

}
