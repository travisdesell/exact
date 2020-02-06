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

#include <string>
using std::string;
using std::to_string;

#include "examm.hxx"
#include "rnn_genome.hxx"
#include "generate_nn.hxx"
#include "rec_depth_dist.hxx"
#include "speciation_strategy.hxx"
#include "island_speciation_strategy.hxx"


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
    for (int32_t i = 0; i < genomes.size(); i++) {
        while (genomes[i].size() > 0) {
            genome = genomes[i].back();
            genomes[i].pop_back();
            delete genome;
        }
    }
}

EXAMM::EXAMM(int32_t _population_size, int32_t _number_islands, int32_t _max_genomes, int32_t _num_genomes_check_on_island, string _speciation_method,
                const vector<string> &_input_parameter_names,
                const vector<string> &_output_parameter_names,
                const map<string,double> &_normalize_mins,
                const map<string,double> &_normalize_maxs,
                int32_t _bp_iterations, double _learning_rate,
                bool _use_high_threshold, double _high_threshold,
                bool _use_low_threshold, double _low_threshold,
                bool _use_dropout, double _dropout_probability,
                int32_t _min_recurrent_depth, int32_t _max_recurrent_depth,
                string _rec_sampling_population, string _rec_sampling_distribution, string _output_directory,
                string _genome_file_name,
                int _no_extra_inputs, int _no_extra_outputs,
                vector<string> &_inputs_to_remove, vector<string> &_outputs_to_remove,
                bool _tl_ver1, bool _tl_ver2, bool _tl_ver3 ) :
                                        population_size(_population_size),
                                        number_islands(_number_islands),
                                        max_genomes(_max_genomes),
                                        number_inputs(_input_parameter_names.size()),
                                        number_outputs(_output_parameter_names.size()),
                                        bp_iterations(_bp_iterations),
                                        learning_rate(_learning_rate),
                                        use_high_threshold(_use_high_threshold),
                                        high_threshold(_high_threshold),
                                        use_low_threshold(_use_low_threshold),
                                        low_threshold(_low_threshold),
                                        use_dropout(_use_dropout),
                                        dropout_probability(_dropout_probability),
                                        output_directory(_output_directory),
                                        genome_file_name(_genome_file_name),
                                        no_extra_inputs(_no_extra_inputs),
                                        no_extra_outputs(_no_extra_outputs),
                                        tl_ver1(_tl_ver1),
                                        tl_ver2(_tl_ver2),
                                        tl_ver3(_tl_ver3) {

    input_parameter_names = _input_parameter_names;
    output_parameter_names = _output_parameter_names;
    normalize_mins = _normalize_mins;
    normalize_maxs = _normalize_maxs;
    inputs_to_remove  = _inputs_to_remove;

    outputs_to_remove = _outputs_to_remove ;

    total_bp_epochs = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    //update to now have islands of genomes
    genomes = vector< vector<RNN_Genome*> >(number_islands);

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    //rng_crossover_weight = uniform_real_distribution<double>(0.0, 0.0);
    //rng_crossover_weight = uniform_real_distribution<double>(-0.10, 0.1);
    rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);
    //rng_crossover_weight = uniform_real_distribution<double>(0.45, 0.55);

    min_recurrent_depth = _min_recurrent_depth;
    max_recurrent_depth = _max_recurrent_depth;

    Log::error("Speciation method is: %s\n", speciation_method.c_str());
    if (speciation_method.compare("island") == 0 || speciation_method.compare("") == 0) {
        //generate a minimal feed foward network as the seed genome
        RNN_Genome *seed_genome = NULL;
        printf("<%s>\n", genome_file_name.c_str());
        if (genome_file_name.compare("") == 0) {
            seed_genome = create_ff(number_inputs, 0, 0, number_outputs, 0);
            seed_genome->initialize_randomly();
            edge_innovation_count = seed_genome->edges.size() + seed_genome->recurrent_edges.size();
            node_innovation_count = seed_genome->nodes.size();
        }
        else {
            Log::error("doing transfer!\n");
            seed_genome = generate_for_transfer_learning(genome_file_name, no_extra_inputs, no_extra_outputs );
            Log::error("generated seed genome, number of inputs: %d, number of outputs: %d\n", seed_genome->get_number_inputs(), seed_genome->get_number_outputs());

            //printf("Hello2\n");
        }

        seed_genome->set_generated_by("initial");

        //insert a copy of it into the population so
        //additional requests can mutate it


        seed_genome->best_validation_mse = EXAMM_MAX_DOUBLE;
        seed_genome->best_validation_mae = EXAMM_MAX_DOUBLE;
        //seed_genome->best_parameters.clear();

        speciation_strategy = new IslandSpeciationStrategy(number_islands, population_size, 0.70, 0.20, 0.10, seed_genome);
    }

    if (_rec_sampling_population.compare("global") == 0) {
        rec_sampling_population = GLOBAL_POPULATION;
    } else if (_rec_sampling_population.compare("island") == 0) {
        rec_sampling_population = ISLAND_POPULATION;
    } else {
        Log::warning("value passed to --rec_sampling_population is not valid ('%s'), defaulting to global.\n", _rec_sampling_population.c_str());
        rec_sampling_population = GLOBAL_POPULATION;
    }

    if (_rec_sampling_distribution.compare("global") == 0) {
        rec_sampling_population = NORMAL_DISTRIBUTION;
    } else if (_rec_sampling_distribution.compare("histogram") == 0) {
        rec_sampling_population = HISTOGRAM_DISTRIBUTION;
    } else if (_rec_sampling_distribution.compare("uniform") == 0) {
        rec_sampling_population = UNIFORM_DISTRIBUTION;
    } else {
        Log::warning("value passed to --rec_sampling_distribution is not valid ('%s'), defaulting to uniform.\n", _rec_sampling_distribution.c_str());
        rec_sampling_distribution = UNIFORM_DISTRIBUTION;
    }

    epigenetic_weights = true;

    more_fit_crossover_rate = 1.00;
    less_fit_crossover_rate = 0.50;
    //more_fit_crossover_rate = 0.75;
    //less_fit_crossover_rate = 0.25;

    clone_rate = 1.0;

    add_edge_rate = 1.0;
    //add_recurrent_edge_rate = 3.0;
    add_recurrent_edge_rate = 1.0;
    enable_edge_rate = 1.0;
    //disable_edge_rate = 3.0;
    disable_edge_rate = 1.0;
    //split_edge_rate = 1.0;
    split_edge_rate = 0.0;

    possible_node_types.clear();
    possible_node_types.push_back(SIMPLE_NODE);
    possible_node_types.push_back(JORDAN_NODE);
    possible_node_types.push_back(ELMAN_NODE);
    possible_node_types.push_back(UGRNN_NODE);
    possible_node_types.push_back(MGU_NODE);
    possible_node_types.push_back(GRU_NODE);
    possible_node_types.push_back(LSTM_NODE);
    possible_node_types.push_back(DELTA_NODE);

    bool node_ops = true;
    if (node_ops) {
        add_node_rate = 1.0;
        enable_node_rate = 1.0;
        //disable_node_rate = 3.0;
        disable_node_rate = 1.0;
        split_node_rate = 1.0;
        merge_node_rate = 1.0;

    } else {
        add_node_rate = 0.0;
        enable_node_rate = 0.0;
        disable_node_rate = 0.0;
        split_node_rate = 0.0;
        merge_node_rate = 0.0;
    }

    if (output_directory != "") {
        mkpath(output_directory.c_str(), 0777);
        log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
        (*log_file) << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";
        memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";
        for (int i = 0; i < (int32_t)genomes.size(); i++)
        {
            (*log_file) << "," << "Island_" << i << "_best_fitness" ;
            (*log_file) << "," << "Island_" << i << "_worst_fitness";
            memory_log << "," << "Island_" << i << "_best_fitness";
            memory_log << "," << "Island_" << i << "_worst_fitness";
        }
        (*log_file) << endl;
        memory_log << endl;
    } else {
        log_file = NULL;
    }

    startClock = std::chrono::system_clock::now();

}

void EXAMM::print() {
    speciation_strategy->print();
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
        RNN_Genome *best_genome = get_best_genome();

        std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

        (*log_file) << speciation_strategy->get_inserted_genomes()
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count();

        // log best fitness
        for (int i = 0; i < (int32_t)genomes.size(); i++) {
            double best_fitness = EXAMM_MAX_DOUBLE;
            double worst_fitness = -EXAMM_MAX_DOUBLE;
            for (int32_t j = 0; j < (int32_t)genomes[i].size(); j++) {
                if (genomes[i][j]->get_fitness() < best_fitness) {
                    best_fitness = genomes[i][j]->get_fitness();
                }
                if (genomes[i][j]->get_fitness() > worst_fitness)
                {
                    worst_fitness = genomes[i][j]->get_fitness();
                }
            }
            (*log_file) << "," << best_fitness << "," << worst_fitness;
        }

        (*log_file) << endl;

        memory_log << speciation_strategy->get_inserted_genomes()
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count();

        // log best fitness
        for (int i = 0; i < (int32_t)genomes.size(); i++)
        {
            double best_fitness = EXAMM_MAX_DOUBLE;
            double worst_fitness = -EXAMM_MAX_DOUBLE;
            for (int32_t j = 0; j < (int32_t)genomes[i].size(); j++)
            {
                if (genomes[i][j]->get_fitness() < best_fitness)
                {
                    best_fitness = genomes[i][j]->get_fitness();
                }
                if (genomes[i][j]->get_fitness() > worst_fitness)
                {
                    worst_fitness = genomes[i][j]->get_fitness();
                }
            }
            memory_log << "," << best_fitness << "," << worst_fitness;
        }

        memory_log << endl;

    }
}

void EXAMM::write_memory_log(string filename) {
    ofstream log_file(filename);
    log_file << memory_log.str();
    log_file.close();
}

void EXAMM::set_possible_node_types(vector<string> possible_node_type_strings) {
    possible_node_types.clear();

    for (int32_t i = 0; i < possible_node_type_strings.size(); i++) {
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

    Log::info("genomes evaluated: %10d , attempting to insert: %s\n", (speciation_strategy->get_inserted_genomes() + 1), parse_fitness(genome->get_fitness()).c_str());

    if (!genome->sanity_check()) {
        Log::error("genome failed sanity check on insert!\n");
        exit(1);
    }

    //updates EXAMM's mapping of which genomes have been generated by what
    genome->update_generation_map(generated_from_map);

    int32_t insert_position = speciation_strategy->insert_genome(genome) >= 0;

    //write this genome to disk if it was a new best found genome
    if (insert_position == 0) {
        genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(genome->get_generation_id()) + ".gv");
        genome->write_to_file(output_directory + "/rnn_genome_" + to_string(genome->get_generation_id()) + ".bin");
    }
    speciation_strategy->print();
    update_log();

    return insert_position >= 0;
}

RNN_Genome* EXAMM::generate_for_transfer_learning(string file_name, int extra_inputs, int extra_outputs) {
    RNN_Genome* genome = new RNN_Genome(file_name);
    vector<RNN_Node_Interface*> output_nodes;
    vector<RNN_Node_Interface*> input_nodes;
    vector<RNN_Node_Interface*> new_output_nodes;
    vector<RNN_Node_Interface*> new_input_nodes;

    uniform_real_distribution<double> rng(-0.5, 0.5);

    double mu, sigma;
    genome->get_mu_sigma(genome->best_parameters, mu, sigma);


    //iterate over all the input parameters and determine which
    //need to be kept
    vector<int> new_input_parameter_id;
    for (int32_t i = 0; i < genome->input_parameter_names.size(); i++) {
        bool keep_parameter = true;
        for (auto removed : inputs_to_remove) {
            if (genome->input_parameter_names[i] == removed ) {
                keep_parameter = false;
                break;
            }
        }

        if (keep_parameter) new_input_parameter_id.push_back(i);
    }

    //iterate over all the output parameters and determine which
    //need to be kept
    vector<int> new_output_parameter_id;
    for (int32_t i = 0; i < genome->output_parameter_names.size(); i++) {
        bool keep_parameter = true;
        for ( auto removed: outputs_to_remove ) {
            if (genome->output_parameter_names[i] == removed) {
                keep_parameter = false;
                break;
            }
        }

        if (keep_parameter) new_output_parameter_id.push_back(i);
    }

    vector<RNN_Node_Interface*> new_nodes;
    int count = 0;
    for (int32_t i = 0; i < genome->nodes.size(); i++) {
        if (genome->nodes[i]->get_layer_type() == INPUT_LAYER) {
            for (auto id : new_input_parameter_id) {
                if (id == i) {
                    new_nodes.push_back(genome->nodes[i]);
                }
            }

        } else if (genome->nodes[i]->get_layer_type() == HIDDEN_LAYER) {
            new_nodes.push_back(genome->nodes[i]);

        } else if (genome->nodes[i]->get_layer_type() == OUTPUT_LAYER) {
            for (auto id : new_output_parameter_id) {
                if (id == count) {
                    new_nodes.push_back(genome->nodes[i]);
                }
            }
            count++;
        } else {
            std::cerr << "ERROR: Layer Type " << genome->nodes[i]->get_layer_type() << " Not Valid\n" ;
            exit(1) ;
        }
    }

    vector<RNN_Edge*> new_edges;
    vector<RNN_Recurrent_Edge*> new_recurrent_edges;

    int weights_count = 0;
    bool flag = false;
    for (auto node : genome->nodes) {
        for (auto new_node : new_nodes) {
            if (node->get_innovation_number() == new_node->get_innovation_number()) {
                flag = true;
                break;
            }
        }

        if (flag) {
            if (node_innovation_count < node->get_innovation_number()) {
                node_innovation_count = node->get_innovation_number();
            }

            if (node->get_layer_type() == OUTPUT_LAYER) {
                output_nodes.push_back(node);
            } else if (node->get_layer_type() == INPUT_LAYER) {
                input_nodes.push_back(node);
            }
        } else {
            for (int32_t j = 0; j < node->get_number_weights(); j++)
                weights_count++;
        }
        flag = false;
    }
    genome->nodes = new_nodes;


    flag = false;
    for (auto edge : genome->edges) {
        for (auto InNode : genome->nodes) {
            if (edge->get_input_innovation_number() == InNode->get_innovation_number()) {
                for (auto OutNode : genome->nodes) {
                    if (edge->get_output_innovation_number() == OutNode->get_innovation_number()) {
                        flag = true;
                        break;
                    }
                }
                break;
            }
        }
        if (flag) {
            if (edge_innovation_count<edge->get_innovation_number())
                edge_innovation_count = edge->get_innovation_number() ;
            new_edges.push_back(edge) ;
        }
        else
        Log::info("Execluding Edge: %d In: %d Out: %d\n", edge->get_innovation_number(), edge->get_input_innovation_number(), edge->get_output_innovation_number());
        flag = false ;
        weights_count++ ;
    }

    flag = false ;
    for (auto recedge : genome->recurrent_edges) {
        for (auto InNode : genome->nodes) {
            if (recedge->get_input_innovation_number() == InNode->get_innovation_number() ) {
                for (auto OutNode : genome->nodes ) {
                    if (recedge->get_output_innovation_number() == OutNode->get_innovation_number()) {
                        flag = true;
                        break;
                    }
                }
                break;
            }
        }
        if (flag) {
            if (edge_innovation_count < recedge->get_innovation_number())
                edge_innovation_count = recedge->get_innovation_number();
            new_recurrent_edges.push_back(recedge);
        }
        else
            Log::info("Execluding Edge: %d In: %d Out: %d\n", recedge->get_innovation_number(), recedge->get_input_innovation_number(), recedge->get_output_innovation_number());
        flag = false;
        weights_count++;
    }

    genome->edges = new_edges;
    genome->recurrent_edges = new_recurrent_edges;

    //*** JUST CHECKING IF THIS WILL FIX THE BUG ***//
    // node_innovation_count+=1000;
    // edge_innovation_count+=1000;
    //*** JUST CHECKING IF THIS WILL FIX THE BUG ***//

    for (int32_t i = 0; i < extra_outputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, 1.0 /*output nodes should be depth 1*/, SIMPLE_NODE);
        node->initialize_randomly(genome->generator, genome->normal_distribution, mu, sigma);
        genome->nodes.push_back(node);
        new_output_nodes.push_back(node) ;
    }

    /* TRANSFER LEARNING VERSIONS:
        - V1: Inputs to Outputs
        - V2: Inputs to Hidden
        - V3: Outputs to Hidden
    */

    for (int32_t i = 0; i < extra_inputs; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, 0, SIMPLE_NODE);
        node->initialize_randomly(genome->generator, genome->normal_distribution, mu, sigma);
        genome->nodes.push_back(node);
        new_input_nodes.push_back(node) ;

        //Connecting New Input Nodes to Old Output Nodes
        if (tl_ver1) {
            for (auto out_node: output_nodes) {
                RNN_Edge *edge = new RNN_Edge(++edge_innovation_count, node, out_node);
                edge->weight = bound(genome->normal_distribution.random(genome->generator, mu, sigma));
            }
        }
    }

    //Connecting Input Nodes to New Output Nodes
    if (tl_ver1) {
        for (auto node : genome->nodes) {
            if (node->get_layer_type() == INPUT_LAYER) {
                for (auto new_output_node : new_output_nodes) {
                    genome->edges.push_back(new RNN_Edge(++edge_innovation_count, node, new_output_node)) ;
                }
            }
        }
    }

    auto rng_ = std::default_random_engine {};

    Distribution *dist = get_recurrent_depth_dist(genome->get_group_id());

    // Connecting New Inputs to Hidden Nodes:
    if (tl_ver2 && new_input_nodes.size()!=0) {
        Log::info("Creating Edges between New-Inputs and Hid!\n");
        std::shuffle(std::begin(new_input_nodes), std::end(new_input_nodes), rng_);
        for (auto node : new_input_nodes) {
            Log::debug("\tBEFORE -- CHECK EDGE INNOVATION COUNT: %d\n", edge_innovation_count);
            genome->connect_new_input_node(mu, sigma, node, dist, edge_innovation_count);
            Log::debug("\tAFTER -- CHECK EDGE INNOVATION COUNT: %d\n", edge_innovation_count);
        }
    }

    // Connecting New Outputs to Hidden Nodes:
    if (tl_ver3 && new_output_nodes.size()!=0) {
        Log::info("Creating Edges between New-Outputs and Hid!\n");
        std::shuffle(std::begin(new_output_nodes), std::end(new_output_nodes), rng_);
        for (auto node : new_output_nodes) {
            Log::debug("\tBEFORE -- CHECK EDGE INNOVATION COUNT: %d\n", edge_innovation_count);
            genome->connect_new_output_node(mu, sigma, node, dist, edge_innovation_count);
            Log::debug("\tAFTER -- CHECK EDGE INNOVATION COUNT: %d\n", edge_innovation_count);
        }
    }

    //need to recalculate the reachability of each node
    genome->assign_reachability();

    //need to make sure that each input and each output has at least one connection
    for (auto node : genome->nodes) {
        Log::info("node[%d], depth: %lf, total_inputs: %d, total_outputs: %d\n", node->get_innovation_number(), node->get_depth(), node->get_total_inputs(), node->get_total_outputs());

        if (node->get_layer_type() == INPUT_LAYER) {
            if (node->get_total_outputs() == 0) {
                Log::info("input node[%d] had no outputs, connecting it!\n", node->get_innovation_number());
                //if an input has no outgoing edges randomly connect it
                genome->connect_new_input_node(mu, sigma, node, dist, edge_innovation_count);
            }

        } else if (node->get_layer_type() == OUTPUT_LAYER) {
            if (node->get_total_inputs() == 0) {
                Log::info("output node[%d] had no inputs, connecting it!\n", node->get_innovation_number());
                //if an output has no incoming edges randomly connect it
                genome->connect_new_output_node(mu, sigma, node, dist, edge_innovation_count);
            }
        }
    }

    delete dist;

    //update the reachabaility again
    genome->assign_reachability();

    Log::info("new_parameters.size() before get weights: %d\n", genome->initial_parameters.size());

    //update the new and best parameter lengths because this will have added edges
    vector<double> updated_genome_parameters;
    genome->get_weights(updated_genome_parameters);
    genome->set_initial_parameters( updated_genome_parameters );
    genome->set_best_parameters( updated_genome_parameters );

    Log::info("new_parameters.size() after get weights: %d\n", updated_genome_parameters.size());

    Log::info("FINISHING PREPARING INITIAL GENOME\n");
    return genome;
}

RNN_Genome* EXAMM::generate_genome() {
    if (speciation_strategy->get_inserted_genomes() > max_genomes) return NULL;

    function<void (int32_t, RNN_Genome*)> mutate_function =
        [=](int32_t max_mutations, RNN_Genome *genome) {
            this->mutate(max_mutations, genome);
        };

    function<RNN_Genome* (RNN_Genome*, RNN_Genome*)> crossover_function =
        [=](RNN_Genome *parent1, RNN_Genome *parent2) {
            return this->crossover(parent1, parent2);
        };

    RNN_Genome *genome = speciation_strategy->generate_genome(rng_0_1, generator, mutate_function, crossover_function);

    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    genome->set_normalize_bounds(normalize_mins, normalize_maxs);
    genome->set_bp_iterations(bp_iterations);
    genome->set_learning_rate(learning_rate);

    if (use_high_threshold) genome->enable_high_threshold(high_threshold);
    if (use_low_threshold) genome->enable_low_threshold(low_threshold);
    if (use_dropout) genome->enable_dropout(dropout_probability);

    if (!epigenetic_weights) genome->initialize_randomly();

    //this is just a sanity check, can most likely comment out (checking to see
    //if all the paramemters are sane)
    Log::debug("getting mu/sigma after random initialization of copy!\n");
    double _mu, _sigma;
    genome->get_mu_sigma(genome->best_parameters, _mu, _sigma);

    return genome;
}

int EXAMM::get_random_node_type() {
    return possible_node_types[rng_0_1(generator) * possible_node_types.size()];
}

Distribution *EXAMM::get_recurrent_depth_dist(int32_t island_index) {
    Distribution *d = NULL;
    if (rec_sampling_distribution != UNIFORM_DISTRIBUTION) {
        if (rec_sampling_distribution == NORMAL_DISTRIBUTION) {
            if (rec_sampling_population == ISLAND_POPULATION)
                d = new RecDepthNormalDist(genomes[island_index], min_recurrent_depth, max_recurrent_depth);
            else
                d = new RecDepthNormalDist(genomes, min_recurrent_depth, max_recurrent_depth);
        } else {
            if (rec_sampling_population == ISLAND_POPULATION)
                d = new RecDepthHistDist(genomes[island_index], min_recurrent_depth, max_recurrent_depth);
            else
                d = new RecDepthHistDist(genomes, min_recurrent_depth, max_recurrent_depth);
        }
    } else {
        d = new RecDepthUniformDist(min_recurrent_depth, max_recurrent_depth);
    }
    return d;
}

void EXAMM::mutate(int32_t max_mutations, RNN_Genome *g) {
    double total = clone_rate + add_edge_rate + add_recurrent_edge_rate + enable_edge_rate + disable_edge_rate + split_edge_rate + add_node_rate + enable_node_rate + disable_node_rate + split_node_rate + merge_node_rate;

    bool modified = false;

    double mu, sigma;

    //g->write_graphviz("rnn_genome_premutate_" + to_string(g->get_generation_id()) + ".gv");

    Log::debug("generating new genome by mutation.\n");
    g->get_mu_sigma(g->best_parameters, mu, sigma);
    g->clear_generated_by();

    //the the weights in the genome to it's best parameters
    //for epigenetic iniitalization
    if (g->best_parameters.size() == 0) {
        g->set_weights(g->initial_parameters);
        g->get_mu_sigma(g->initial_parameters, mu, sigma);
    } else {
        g->set_weights(g->best_parameters);
        g->get_mu_sigma(g->best_parameters, mu, sigma);
    }

    int number_mutations = 0;

    while (!modified) {
        if (modified) {
            modified = false;
            number_mutations++;
            if (number_mutations >= max_mutations) break;
        }

        g->assign_reachability();
        double rng = rng_0_1(generator) * total;
        int new_node_type = get_random_node_type();
        string node_type_str = NODE_TYPES[new_node_type];
        Log::debug( "rng: %lf, total: %lf, new node type: %d (%s)\n", rng, total, new_node_type, node_type_str.c_str());

        if (rng < clone_rate) {
            Log::debug("\tcloned\n");
            g->set_generated_by("clone");
            modified = true;
            continue;
        }
        rng -= clone_rate;

        if (rng < add_edge_rate) {
            modified = g->add_edge(mu, sigma, edge_innovation_count);
            Log::debug("\tadding edge, modified: %d\n", modified);
            if (modified) g->set_generated_by("add_edge");
            continue;
        }
        rng -= add_edge_rate;

        if (rng < add_recurrent_edge_rate) {
            Distribution *dist = get_recurrent_depth_dist(g->get_group_id());
            modified = g->add_recurrent_edge(mu, sigma, dist, edge_innovation_count);
            delete dist;
            Log::debug("\tadding recurrent edge, modified: %d\n", modified);
            if (modified) g->set_generated_by("add_recurrent_edge");
            continue;
        }
        rng -= add_recurrent_edge_rate;

        if (rng < enable_edge_rate) {
            modified = g->enable_edge();
            Log::debug("\tenabling edge, modified: %d\n", modified);
            if (modified) g->set_generated_by("enable_edge");
            continue;
        }
        rng -= enable_edge_rate;

        if (rng < disable_edge_rate) {
            modified = g->disable_edge();
            Log::debug("\tdisabling edge, modified: %d\n", modified);
            if (modified) g->set_generated_by("disable_edge");
            continue;
        }
        rng -= disable_edge_rate;

        if (rng < split_edge_rate) {
            Distribution *dist = get_recurrent_depth_dist(g->get_group_id());
            modified = g->split_edge(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
            Log::debug("\tsplitting edge, modified: %d\n", modified);
            if (modified) g->set_generated_by("split_edge(" + node_type_str + ")");
            continue;
        }
        rng -= split_edge_rate;

        if (rng < add_node_rate) {
            Distribution *dist = get_recurrent_depth_dist(g->get_group_id());
            modified = g->add_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
            Log::debug("\tadding node, modified: %d\n", modified);
            if (modified) g->set_generated_by("add_node(" + node_type_str + ")");
            continue;
        }
        rng -= add_node_rate;

        if (rng < enable_node_rate) {
            modified = g->enable_node();
            Log::debug("\tenabling node, modified: %d\n", modified);
            if (modified) g->set_generated_by("enable_node");
            continue;
        }
        rng -= enable_node_rate;

        if (rng < disable_node_rate) {
            modified = g->disable_node();
            Log::debug("\tdisabling node, modified: %d\n", modified);
            if (modified) g->set_generated_by("disable_node");
            continue;
        }
        rng -= disable_node_rate;

        if (rng < split_node_rate) {
            Distribution *dist = get_recurrent_depth_dist(g->get_group_id());
            modified = g->split_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
            Log::debug("\tsplitting node, modified: %d\n", modified);
            if (modified) g->set_generated_by("split_node(" + node_type_str + ")");
            continue;
        }
        rng -= split_node_rate;

        if (rng < merge_node_rate) {
            Distribution *dist = get_recurrent_depth_dist(g->get_group_id());
            modified = g->merge_node(mu, sigma, new_node_type, dist, edge_innovation_count, node_innovation_count);
            Log::debug("\tmerging node, modified: %d\n", modified);
            if (modified) g->set_generated_by("merge_node(" + node_type_str + ")");
            continue;
        }
        rng -= merge_node_rate;
    }

    //get the new set of parameters (as new paramters may have been
    //added duriung mutation) and set them to the initial parameters
    //for epigenetic_initialization

    vector<double> new_parameters;
    g->get_weights(new_parameters);
    g->initial_parameters = new_parameters;

    if (Log::at_level(Log::DEBUG)) {
        Log::debug("getting mu/sigma before assign reachability\n");
        g->get_mu_sigma(new_parameters, mu, sigma);
    }

    g->assign_reachability();

    //reset the genomes statistics (as these carry over on copy)
    g->best_validation_mse = EXAMM_MAX_DOUBLE;
    g->best_validation_mae = EXAMM_MAX_DOUBLE;

    if (Log::at_level(Log::DEBUG)) {
        Log::debug("checking parameters after mutation\n");
        g->get_mu_sigma(g->initial_parameters, mu, sigma);
    }

    g->best_parameters.clear();
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
        }

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


RNN_Genome* EXAMM::crossover(RNN_Genome *p1, RNN_Genome *p2) {
    Log::debug("generating new genome by crossover!\n");
    Log::debug("p1->island: %d, p2->island: %d\n", p1->get_group_id(), p2->get_group_id());
    Log::debug("p1->number_inputs: %d, p2->number_inputs: %d\n", p1->get_number_inputs(), p2->get_number_inputs());

    for (int i = 0; i < p1->nodes.size(); i++) {
        Log::debug("p1 node[%d], in: %d, depth: %lf, layer_type: %d, node_type: %d, reachable: %d, enabled: %d\n", i, p1->nodes[i]->get_innovation_number(), p1->nodes[i]->get_depth(), p1->nodes[i]->get_layer_type(), p1->nodes[i]->get_node_type(), p1->nodes[i]->is_reachable(), p1->nodes[i]->is_enabled());
    }

    for (int i = 0; i < p2->nodes.size(); i++) {
        Log::debug("p2 node[%d], in: %d, depth: %lf, layer_type: %d, node_type: %d, reachable: %d, enabled: %d\n", i, p2->nodes[i]->get_innovation_number(), p2->nodes[i]->get_depth(), p2->nodes[i]->get_layer_type(), p2->nodes[i]->get_node_type(), p2->nodes[i]->is_reachable(), p2->nodes[i]->is_enabled());
    }

    double _mu, _sigma;
    Log::debug("getting p1 mu/sigma!\n");
    if (p1->best_parameters.size() == 0) {
        p1->set_weights(p1->initial_parameters);
        p1->get_mu_sigma(p1->initial_parameters, _mu, _sigma);
    } else {
        p1->set_weights(p1->best_parameters);
        p1->get_mu_sigma(p1->best_parameters, _mu, _sigma);
    }

    Log::debug("getting p2 mu/sigma!\n");
    if (p2->best_parameters.size() == 0) {
        p2->set_weights(p2->initial_parameters);
        p2->get_mu_sigma(p2->initial_parameters, _mu, _sigma);
    } else {
        p2->set_weights(p2->best_parameters);
        p2->get_mu_sigma(p2->best_parameters, _mu, _sigma);
    }

    //nodes are copied in the attempt_node_insert_function
    vector< RNN_Node_Interface* > child_nodes;
    vector< RNN_Edge* > child_edges;
    vector< RNN_Recurrent_Edge* > child_recurrent_edges;

    //edges are not sorted in order of innovation number, they need to be
    vector< RNN_Edge* > p1_edges = p1->edges;
    vector< RNN_Edge* > p2_edges = p2->edges;

    sort(p1_edges.begin(), p1_edges.end(), sort_RNN_Edges_by_innovation());
    sort(p2_edges.begin(), p2_edges.end(), sort_RNN_Edges_by_innovation());

    Log::debug("\tp1 innovation numbers AFTER SORT:\n");
    for (int32_t i = 0; i < (int32_t)p1_edges.size(); i++) {
        Log::trace("\t\t%d\n", p1_edges[i]->innovation_number);
    }
    Log::debug("\tp2 innovation numbers AFTER SORT:\n");
    for (int32_t i = 0; i < (int32_t)p2_edges.size(); i++) {
        Log::debug("\t\t%d\n", p2_edges[i]->innovation_number);
    }

    vector< RNN_Recurrent_Edge* > p1_recurrent_edges = p1->recurrent_edges;
    vector< RNN_Recurrent_Edge* > p2_recurrent_edges = p2->recurrent_edges;

    sort(p1_recurrent_edges.begin(), p1_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_innovation());
    sort(p2_recurrent_edges.begin(), p2_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_innovation());

    int32_t p1_position = 0;
    int32_t p2_position = 0;

    while (p1_position < (int32_t)p1_edges.size() && p2_position < (int32_t)p2_edges.size()) {
        RNN_Edge* p1_edge = p1_edges[p1_position];
        RNN_Edge* p2_edge = p2_edges[p2_position];

        int p1_innovation = p1_edge->innovation_number;
        int p2_innovation = p2_edge->innovation_number;

        if (p1_innovation == p2_innovation) {
            attempt_edge_insert(child_edges, child_nodes, p1_edge, p2_edge, true);

            p1_position++;
            p2_position++;
        } else if (p1_innovation < p2_innovation) {
            bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
            if (p1_edge->is_reachable()) set_enabled = true;
            else set_enabled = false;

            attempt_edge_insert(child_edges, child_nodes, p1_edge, NULL, set_enabled);

            p1_position++;
        } else {
            bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
            if (p2_edge->is_reachable()) set_enabled = true;
            else set_enabled = false;

            attempt_edge_insert(child_edges, child_nodes, p2_edge, NULL, set_enabled);

            p2_position++;
        }
    }

    while (p1_position < (int32_t)p1_edges.size()) {
        RNN_Edge* p1_edge = p1_edges[p1_position];

        bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
        if (p1_edge->is_reachable()) set_enabled = true;
        else set_enabled = false;

        attempt_edge_insert(child_edges, child_nodes, p1_edge, NULL, set_enabled);

        p1_position++;
    }

    while (p2_position < (int32_t)p2_edges.size()) {
        RNN_Edge* p2_edge = p2_edges[p2_position];

        bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
        if (p2_edge->is_reachable()) set_enabled = true;
        else set_enabled = false;

        attempt_edge_insert(child_edges, child_nodes, p2_edge, NULL, set_enabled);

        p2_position++;
    }

    //do the same for recurrent_edges
    p1_position = 0;
    p2_position = 0;

    while (p1_position < (int32_t)p1_recurrent_edges.size() && p2_position < (int32_t)p2_recurrent_edges.size()) {
        RNN_Recurrent_Edge* p1_recurrent_edge = p1_recurrent_edges[p1_position];
        RNN_Recurrent_Edge* p2_recurrent_edge = p2_recurrent_edges[p2_position];

        int p1_innovation = p1_recurrent_edge->innovation_number;
        int p2_innovation = p2_recurrent_edge->innovation_number;

        if (p1_innovation == p2_innovation) {
            //do weight crossover
            attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p1_recurrent_edge, p2_recurrent_edge, true);

            p1_position++;
            p2_position++;
        } else if (p1_innovation < p2_innovation) {
            bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
            if (p1_recurrent_edge->is_reachable()) set_enabled = true;
            else set_enabled = false;

            attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p1_recurrent_edge, NULL, set_enabled);

            p1_position++;
        } else {
            bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
            if (p2_recurrent_edge->is_reachable()) set_enabled = true;
            else set_enabled = false;

            attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p2_recurrent_edge, NULL, set_enabled);

            p2_position++;
        }
    }

    while (p1_position < (int32_t)p1_recurrent_edges.size()) {
        RNN_Recurrent_Edge* p1_recurrent_edge = p1_recurrent_edges[p1_position];

        bool set_enabled = rng_0_1(generator) < more_fit_crossover_rate;
        if (p1_recurrent_edge->is_reachable()) set_enabled = true;
        else set_enabled = false;

        attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p1_recurrent_edge, NULL, set_enabled);

        p1_position++;
    }

    while (p2_position < (int32_t)p2_recurrent_edges.size()) {
        RNN_Recurrent_Edge* p2_recurrent_edge = p2_recurrent_edges[p2_position];

        bool set_enabled = rng_0_1(generator) < less_fit_crossover_rate;
        if (p2_recurrent_edge->is_reachable()) set_enabled = true;
        else set_enabled = false;

        attempt_recurrent_edge_insert(child_recurrent_edges, child_nodes, p2_recurrent_edge, NULL, set_enabled);

        p2_position++;
    }

    sort(child_nodes.begin(), child_nodes.end(), sort_RNN_Nodes_by_depth());
    sort(child_edges.begin(), child_edges.end(), sort_RNN_Edges_by_depth());
    sort(child_recurrent_edges.begin(), child_recurrent_edges.end(), sort_RNN_Recurrent_Edges_by_depth());

    RNN_Genome *child = new RNN_Genome(child_nodes, child_edges, child_recurrent_edges);
    child->set_parameter_names(input_parameter_names, output_parameter_names);
    child->set_normalize_bounds(normalize_mins, normalize_maxs);


    if (p1->get_group_id() == p2->get_group_id()) {
        child->set_generated_by("crossover");
    } else {
        child->set_generated_by("island_crossover");
    }

    double mu, sigma;

    vector<double> new_parameters;
    child->get_weights(new_parameters);
    Log::debug("getting mu/sigma before assign reachability\n");
    child->get_mu_sigma(new_parameters, mu, sigma);

    child->assign_reachability();

    //reset the genomes statistics (as these carry over on copy)
    child->best_validation_mse = EXAMM_MAX_DOUBLE;
    child->best_validation_mae = EXAMM_MAX_DOUBLE;

    //get the new set of parameters (as new paramters may have been
    //added duriung mutatino) and set them to the initial parameters
    //for epigenetic_initialization
    child->get_weights(new_parameters);
    child->initial_parameters = new_parameters;

    Log::debug("checking parameters after crossover\n");
    child->get_mu_sigma(child->initial_parameters, mu, sigma);

    child->best_parameters.clear();

    return child;
}
