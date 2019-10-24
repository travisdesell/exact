#include <algorithm>
using std::sort;

#include <chrono>
#include <cstring>

#include <fstream>
using std::ofstream;

#include <iomanip>
using std::setw;
using std::setprecision;

#include <iostream>
using std::cerr;
using std::cout;
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

#include "common/files.hxx"

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

EXAMM::EXAMM(int32_t _population_size, int32_t _number_islands, int32_t _max_genomes, int32_t _num_genomes_check_on_island, string _check_on_island_method, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs, int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold, double _low_threshold, bool _use_dropout, double _dropout_probability, string _output_directory) : population_size(_population_size), number_islands(_number_islands), max_genomes(_max_genomes), number_inputs(_input_parameter_names.size()), number_outputs(_output_parameter_names.size()), bp_iterations(_bp_iterations), learning_rate(_learning_rate), use_high_threshold(_use_high_threshold), high_threshold(_high_threshold), use_low_threshold(_use_low_threshold), low_threshold(_low_threshold), use_dropout(_use_dropout), dropout_probability(_dropout_probability), output_directory(_output_directory) {
    num_genomes_check_on_island = _num_genomes_check_on_island;
    check_on_island_method = _check_on_island_method;
    input_parameter_names = _input_parameter_names;
    output_parameter_names = _output_parameter_names;
    normalize_mins = _normalize_mins;
    normalize_maxs = _normalize_maxs;

//TODO: should move this into a util file
int mkpath(const char *path, mode_t mode);

    inserted_genomes = 0;
    generated_genomes = 0;
    total_bp_epochs = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    //update to now have islands of genomes
    genomes = vector< vector<RNN_Genome*> >(number_islands);
    island_states = vector<int32_t>(number_islands, ISLAND_INITIALIZING);

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    //rng_crossover_weight = uniform_real_distribution<double>(0.0, 0.0);
    //rng_crossover_weight = uniform_real_distribution<double>(-0.10, 0.1);
    rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);
    //rng_crossover_weight = uniform_real_distribution<double>(0.45, 0.55);

    max_recurrent_depth = 10;

    epigenetic_weights = true;

    mutation_rate = 0.70;
    crossover_rate = 0.20 + mutation_rate;
    island_crossover_rate = 0.10 + crossover_rate;
    //all three should add up to 1.0

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

void EXAMM::print_population() {
    cout << "POPULATIONS: " << endl;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        cout << "\tPOPULATION " << i << ":" << endl;

        cout << "\t" << RNN_Genome::print_statistics_header() << endl;

        for (int32_t j = 0; j < (int32_t)genomes[i].size(); j++) {
            cout << "\t" << genomes[i][j]->print_statistics() << endl;
        }
    }

    cout << endl << endl;

    if (log_file != NULL) {

        //make sure the log file is still good
        if (!log_file->good()) {
            log_file->close();
            delete log_file;

            string output_file = output_directory + "/fitness_log.csv";
            log_file = new ofstream(output_file, std::ios_base::app);

            if (!log_file->is_open()) {
                cerr << "ERROR, could not open EXAMM output log: '" << output_file << "'" << endl;
                exit(1);
            }
        }

        RNN_Genome *best_genome = get_best_genome();

        std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

        (*log_file) << inserted_genomes
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

            memory_log << inserted_genomes
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
            cerr << "ERROR! unknown node type: '" << node_type_s << "'" << endl;
            exit(1);
        }
    }
}

int32_t EXAMM::population_contains(RNN_Genome* genome, int32_t island) {
    for (int32_t j = 0; j < (int32_t)genomes[island].size(); j++) {
        if (genomes[island][j]->equals(genome)) {
            return j;
        }
    }

    return -1;
}

bool EXAMM::populations_full() const {
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        if (genomes[i].size() < population_size) return false;
    }

    return true;
}

string EXAMM::get_output_directory() const {
    return output_directory;
}

RNN_Genome* EXAMM::get_best_genome() {
    int32_t best_genome_population = -1;
    double best_fitness = EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        if (genomes[i].size() > 0) {
            if (genomes[i][0]->get_fitness() <= best_fitness) {
                best_fitness = genomes[i][0]->get_fitness();
                best_genome_population = i;
            }
        }
    }

    if (best_genome_population < 0) {
        return NULL;
    } else {
        return genomes[best_genome_population][0];
    }
}

RNN_Genome* EXAMM::get_worst_genome() {
    int32_t worst_genome_population = -1;
    double worst_fitness = -EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        if (genomes[i].size() > 0) {
            if (genomes[i].back()->get_fitness() > worst_fitness) {
                worst_fitness = genomes[i].back()->get_fitness();
                worst_genome_population = i;
            }
        }
    }

    if (worst_genome_population < 0) {
        return NULL;
    } else {
        return genomes[worst_genome_population].back();
    }
}


double EXAMM::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double EXAMM::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

//this will insert a COPY, original needs to be deleted
bool EXAMM::insert_genome(RNN_Genome* genome) {
    if (!genome->sanity_check()) {
        cerr << "ERROR, genome failed sanity check on insert!" << endl;
        exit(1);
    }

    int32_t island = genome->get_island();
    double new_fitness = genome->get_fitness();

    bool was_inserted = true;

    inserted_genomes++;
    total_bp_epochs += genome->get_bp_iterations();

    genome->update_generation_map(generated_from_map);

    cout << "genomes evaluated: " << setw(10) << inserted_genomes << ", inserting: " << parse_fitness(genome->get_fitness()) << " to island " << island << endl;

    if (genomes[island].size() >= population_size  && new_fitness > genomes[island].back()->get_fitness()) {
        cout << "ignoring genome, fitness: " << new_fitness << " > worst population[" << island << "] fitness: " << genomes[island].back()->get_fitness() << endl;
        print_population();
        return false;
    }

    int32_t duplicate_genome = population_contains(genome, island);
    if (duplicate_genome >= 0) {
        //if fitness is better, replace this genome with new one
        cout << "found duplicate at position: " << duplicate_genome << endl;

        RNN_Genome *duplicate = genomes[island][duplicate_genome];
        if (duplicate->get_fitness() > new_fitness) {
            //erase the genome with loewr fitness from the vector;
            cout << "REPLACING DUPLICATE GENOME, fitness of genome in search: " << parse_fitness(duplicate->get_fitness()) << ", new fitness: " << parse_fitness(genome->get_fitness()) << endl;
            genomes[island].erase(genomes[island].begin() + duplicate_genome);
            delete duplicate;

        } else {
            cerr << "\tpopulation already contains genome! not inserting." << endl;
            print_population();
            return false;
        }
    }

    if (genomes[island].size() < population_size || genomes[island].back()->get_fitness() > new_fitness) {
        //this genome will be inserted
        was_inserted = true;

        if (genomes[island].size() == 0 || genome->get_fitness() < get_best_genome()->get_fitness()) {
            cout << "new best fitness!" << endl;

            if (genome->get_fitness() != EXAMM_MAX_DOUBLE) {
                //need to set the weights for non-initial genomes so we
                //can generate a proper graphviz file
                vector<double> best_parameters = genome->get_best_parameters();
                genome->set_weights(best_parameters);
            }

            genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".gv");
            genome->write_to_file(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".bin", true);

        }

        genome->update_generation_map(inserted_from_map);

        cout << "inserting new genome to island " << island << endl;
        //inorder insert the new individual
        RNN_Genome *copy = genome->copy();
        cout << "created copy with island: " << copy->get_island() << endl;

        genomes[island].insert( upper_bound(genomes[island].begin(), genomes[island].end(), copy, sort_genomes_by_fitness()), copy);
        cout << "finished insert" << endl;

        if (genomes[island].size() >= population_size) {
            island_states[island] = ISLAND_FILLED;
        }

        //delete the worst individual if we've reached the population size
        if ((int32_t)genomes[island].size() > population_size) {
            cout << "deleting worst genome" << endl;
            RNN_Genome *worst = genomes[island].back();
            genomes[island].pop_back();

            delete worst;
        }
    } else {
        was_inserted = false;
        cout << "not inserting genome due to poor fitness" << endl;
    }

    print_population();

    cout << "printed population!" << endl;

    return was_inserted;
}

int32_t EXAMM::check_on_island() {
    if (check_on_island_method == "") {
        return -1;
    }

    // only run this function every n inserted genomes
    if (   num_genomes_check_on_island == 0 
        || inserted_genomes % num_genomes_check_on_island != 0) {
        return -1;
    }

    if (check_on_island_method == "clear_island_with_worst_best_genome") {
        return clear_island_with_worst_best_genome();
    }
    
    return -1;
}

int32_t EXAMM::clear_island_with_worst_best_genome() {
    int32_t worst_island = -1;
    double worst_island_fitness = -EXAMM_MAX_DOUBLE;

    // find the best genome for each island and identify the 
    //island with the worst best genome
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        // islands that are intializing or repopulating are
        //not valid for clearing
        if (island_states[i] != ISLAND_FILLED) {
            continue;
        }

        // find best fitness genome
        double best_fitness = EXAMM_MAX_DOUBLE;
        for (int32_t j = 0; j < (int32_t)genomes[i].size(); j++) {
            if (genomes[i][j]->get_fitness() < best_fitness) {
                best_fitness = genomes[i][j]->get_fitness();
            }
        }

        if (best_fitness > worst_island_fitness) {
            worst_island = i;
            worst_island_fitness = best_fitness;
        }
    }

    // clear the genomes from the identified island
    if (worst_island != -1) {
        cout << "check_on_island: clearing genomes from a bad island: " << worst_island << endl;

        while (genomes[worst_island].size() > 0) {
            RNN_Genome *first_genome = genomes[worst_island][0];
            genomes[worst_island].erase(genomes[worst_island].begin());
            delete first_genome;
        }

        island_states[worst_island] = ISLAND_REPOPULATING;
    }

    return worst_island;
}

void EXAMM::initialize_genome_parameters(RNN_Genome* genome) {
    genome->set_bp_iterations(bp_iterations);
    genome->set_learning_rate(learning_rate);

    if (use_high_threshold) genome->enable_high_threshold(high_threshold);
    if (use_low_threshold) genome->enable_low_threshold(low_threshold);
    if (use_dropout) genome->enable_dropout(dropout_probability);
}

RNN_Genome* EXAMM::generate_genome() {
    if (inserted_genomes > max_genomes) return NULL;

    int32_t island = generated_genomes % number_islands;
    if (genomes[island].size() >= number_islands) {
        island_states[island] = ISLAND_FILLED;
    }

    //check_on_island returns -1 if no island was killed, or the island number otherwise
    int32_t revisit_island = check_on_island();
    
    generated_genomes++;

    RNN_Genome *genome = NULL;

    if (island_states[island] == ISLAND_INITIALIZING) {

        if (genomes[island].size() == 0) {
            //this is the first genome to be generated
            //generate minimal genome, insert it into the population
            genome = create_ff(number_inputs, 0, 0, number_outputs, 0);
            genome->set_island(island);
            genome->set_parameter_names(input_parameter_names, output_parameter_names);
            genome->set_normalize_bounds(normalize_mins, normalize_maxs);

            edge_innovation_count = genome->edges.size() + genome->recurrent_edges.size();
            node_innovation_count = genome->nodes.size();

            genome->set_generated_by("initial");
            initialize_genome_parameters(genome);

            //insert a copy of it into the population so
            //additional requests can mutate it
            genome->initialize_randomly();
            double _mu, _sigma;
            cout << "getting mu/sigma after random initialization!" << endl;
            genome->get_mu_sigma(genome->best_parameters, _mu, _sigma);

            genome->best_validation_mse = EXAMM_MAX_DOUBLE;
            genome->best_validation_mae = EXAMM_MAX_DOUBLE;
            genome->best_parameters.clear();
            //genome->clear_generated_by();

            insert_genome(genome->copy());
        } else {
            while (genome == NULL) {
                int32_t genome_position = genomes[island].size() * rng_0_1(generator);
                genome = genomes[island][genome_position]->copy();
                mutate(genome);

                genome->set_normalize_bounds(normalize_mins, normalize_maxs);
                genome->set_island(island);

                if (genome->outputs_unreachable()) {
                    //no path from at least one input to the outputs
                    delete genome;
                    genome = NULL;
                }
            }

            //the population hasn't been filled yet, so insert a copy of
            //the genome into the population so it can be further mutated
            RNN_Genome *copy = genome->copy();
            copy->initialize_randomly();
            double _mu, _sigma;
            cout << "getting mu/sigma after random initialization of copy!" << endl;
            genome->get_mu_sigma(genome->best_parameters, _mu, _sigma);

            insert_genome(copy);

            //also randomly initialize this genome as
            //what it was generated from was also randomly
            //initialized as the population hasn't been
            //filled
            genome->initialize_randomly();
            cout << "getting mu/sigma after random initialization due to genomes.size() < population_size!" << endl;
            genome->get_mu_sigma(genome->best_parameters, _mu, _sigma);
        }

    } else if (island_states[island] == ISLAND_FILLED) {
        //generate a genome via crossover or mutation

        cout << "generating new genome, genomes[" << island << "].size(): " << genomes[island].size() << ", population_size: " << population_size << ", crossover_rate: " << crossover_rate << endl;

        while (genome == NULL) {
            //if we haven't filled the island populations yet, only
            //use mutation
            //otherwise do mutation at %, crossover at %, and island crossover at %

            double r = rng_0_1(generator);
            if (!populations_full() || r < mutation_rate) {
                int32_t genome_position = genomes[island].size() * rng_0_1(generator);
                genome = genomes[island][genome_position]->copy();
                mutate(genome);

                genome->set_normalize_bounds(normalize_mins, normalize_maxs);
                genome->set_island(island);
            } else if (r < crossover_rate || number_islands == 1) {
                //intra-island crossover

                //select two distinct parent genomes in the same island
                int32_t p1 = genomes[island].size() * rng_0_1(generator);
                int32_t p2 = (genomes[island].size() - 1) * rng_0_1(generator);
// TODO: IS greater than here a bug? p2 will be even further than p1 if p2 > p1
                if (p2 >= p1) p2++;

                //swap so the first parent is the more fit parent
                if (p1 > p2) {
                    int32_t tmp = p1;
                    p1 = p2;
                    p2 = tmp;
                }

                genome = crossover(genomes[island][p1], genomes[island][p2]);
                genome->set_normalize_bounds(normalize_mins, normalize_maxs);
                genome->set_island(island);
            } else {
                //inter-island crossover

                //select two distinct parent genomes in the same island
                int32_t p1 = genomes[island].size() * rng_0_1(generator);

                //select a different island randomly
                //int32_t other_island = rng_0_1(generator) * (number_islands - 1);
                //if (other_island >= island) other_island++;

                int other_island = -1;
                double best_other_fitness = EXAMM_MAX_DOUBLE;
                for (int32_t i = 0; i < genomes.size(); i++) {
                    if (i == island) continue;
                    if (island_states[i] == ISLAND_REPOPULATING) continue;
                    if (genomes[i][0]->get_fitness() < best_other_fitness) {
                        other_island = i;
                        best_other_fitness = genomes[i][0]->get_fitness();
                    }
                }

                RNN_Genome *g1 = genomes[island][p1];
                RNN_Genome *g2 = genomes[other_island][0];

                //swap so the first parent is the more fit parent
                if (g1->get_fitness() > g2->get_fitness()) {
                    RNN_Genome *tmp = g1;
                    g1 = g2;
                    g2 = tmp;
                }

                genome = crossover(g1, g2);
                genome->set_normalize_bounds(normalize_mins, normalize_maxs);
                genome->set_island(island);
                //genome->set_bp_iterations(2 * bp_iterations);
            }

            if (genome->outputs_unreachable()) {
                //no path from at least one input to the outputs
                delete genome;
                genome = NULL;
            }
        }

    } else if (island_states[island] == ISLAND_REPOPULATING) {
        //here's where you put your repopulation code
        //select two other islands (non-overlapping) at random, and select genomes
        //from within those islands and generate a child via crossover

        if (genomes[island].size() == 0) {
            //this is the first genome to be generated
            //generate minimal genome, insert it into the population
            genome = create_ff(number_inputs, 0, 0, number_outputs, 0);
            genome->set_island(island);
            genome->set_parameter_names(input_parameter_names, output_parameter_names);
            genome->set_normalize_bounds(normalize_mins, normalize_maxs);

            edge_innovation_count = genome->edges.size() + genome->recurrent_edges.size();
            node_innovation_count = genome->nodes.size();

            genome->set_generated_by("initial");
            initialize_genome_parameters(genome);

            //insert a copy of it into the population so
            //additional requests can mutate it
            genome->initialize_randomly();
            double _mu, _sigma;
            cout << "getting mu/sigma after random initialization!" << endl;
            genome->get_mu_sigma(genome->best_parameters, _mu, _sigma);

            genome->best_validation_mse = EXAMM_MAX_DOUBLE;
            genome->best_validation_mae = EXAMM_MAX_DOUBLE;
            genome->best_parameters.clear();
            //genome->clear_generated_by();

            insert_genome(genome->copy());
        } else {
            
            while (genome == NULL) {
                int32_t parent_island1;
                do {
                    parent_island1 = genomes.size() * rng_0_1(generator);
                } while (island_states[parent_island1] != ISLAND_FILLED);
                
                int32_t parent_island2;
                do {
                    parent_island2 = (genomes.size() - 1) * rng_0_1(generator);
                } while (   parent_island1 == parent_island2 
                         || island_states[parent_island2] != ISLAND_FILLED);

                RNN_Genome* genome1 = genomes[parent_island1][genomes[parent_island1].size() * rng_0_1(generator)];
                RNN_Genome* genome2 = genomes[parent_island2][(genomes[parent_island1].size() - 1) * rng_0_1(generator)];

                genome = crossover(genome1, genome2);
                genome->set_normalize_bounds(normalize_mins, normalize_maxs);
                genome->set_island(island);

                if (genome->outputs_unreachable()) {
                    //no path from at least one input to the outputs
                    delete genome;
                    genome = NULL;
                }
            }
        }
    } else {
        cerr << "ERROR: unknown island state (" << island_states[island] << ")" << endl;
        cerr << "This should never happen!" << endl;
        exit(1);
    
    }
    //genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(generated_genomes) + ".gv");
    //genome->write_to_file(output_directory + "/rnn_genome_" + to_string(generated_genomes) + ".bin");

    if (!epigenetic_weights) genome->initialize_randomly();

    genome->set_generation_id(generated_genomes);
    return genome;
}

int EXAMM::get_random_node_type() {
    return possible_node_types[rng_0_1(generator) * possible_node_types.size()];
}

void EXAMM::mutate(RNN_Genome *g) {
    double total = clone_rate + add_edge_rate + add_recurrent_edge_rate + enable_edge_rate + disable_edge_rate + split_edge_rate + add_node_rate + enable_node_rate + disable_node_rate + split_node_rate + merge_node_rate;

    bool modified = false;

    double mu, sigma;

    //g->write_graphviz("rnn_genome_premutate_" + to_string(generated_genomes) + ".gv");

    cout << "generating new genome by mutation" << endl;
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

    while (!modified) {
        g->assign_reachability();
        double rng = rng_0_1(generator) * total;
        int new_node_type = get_random_node_type();
        string node_type_str = NODE_TYPES[new_node_type];
        cout << "rng: " << rng << ", total: " << total << ", new node type: " << new_node_type << " (" << node_type_str << ")" << endl;

        if (rng < clone_rate) {
            cout << "\tcloned" << endl;
            g->set_generated_by("clone");
            modified = true;
            continue;
        }
        rng -= clone_rate;

        if (rng < add_edge_rate) {
            modified = g->add_edge(mu, sigma, edge_innovation_count);
            cout << "\tadding edge, modified: " << modified << endl;
            if (modified) g->set_generated_by("add_edge");
            continue;
        }
        rng -= add_edge_rate;

        if (rng < add_recurrent_edge_rate) {
            modified = g->add_recurrent_edge(mu, sigma, max_recurrent_depth, edge_innovation_count);
            cout << "\tadding recurrent edge, modified: " << modified << endl;
            if (modified) g->set_generated_by("add_recurrent_edge");
            continue;
        }
        rng -= add_recurrent_edge_rate;

        if (rng < enable_edge_rate) {
            modified = g->enable_edge();
            cout << "\tenabling edge, modified: " << modified << endl;
            if (modified) g->set_generated_by("enable_edge");
            continue;
        }
        rng -= enable_edge_rate;

        if (rng < disable_edge_rate) {
            modified = g->disable_edge();
            cout << "\tdisabling edge, modified: " << modified << endl;
            if (modified) g->set_generated_by("disable_edge");
            continue;
        }
        rng -= disable_edge_rate;

        if (rng < split_edge_rate) {
            modified = g->split_edge(mu, sigma, new_node_type, max_recurrent_depth, edge_innovation_count, node_innovation_count);
            cout << "\tsplitting edge, modified: " << modified << endl;
            if (modified) g->set_generated_by("split_edge(" + node_type_str + ")");
            continue;
        }
        rng -= split_edge_rate;

        if (rng < add_node_rate) {
            modified = g->add_node(mu, sigma, new_node_type, max_recurrent_depth, edge_innovation_count, node_innovation_count);
            cout << "\tadding node, modified: " << modified << endl;
            if (modified) g->set_generated_by("add_node(" + node_type_str + ")");
            continue;
        }
        rng -= add_node_rate;

        if (rng < enable_node_rate) {
            modified = g->enable_node();
            cout << "\tenabling node, modified: " << modified << endl;
            if (modified) g->set_generated_by("enable_node");
            continue;
        }
        rng -= enable_node_rate;

        if (rng < disable_node_rate) {
            modified = g->disable_node();
            cout << "\tdisabling node, modified: " << modified << endl;
            if (modified) g->set_generated_by("disable_node");
            continue;
        }
        rng -= disable_node_rate;

        if (rng < split_node_rate) {
            modified = g->split_node(mu, sigma, new_node_type, max_recurrent_depth, edge_innovation_count, node_innovation_count);
            cout << "\tsplitting node, modified: " << modified << endl;
            if (modified) g->set_generated_by("split_node(" + node_type_str + ")");
            continue;
        }
        rng -= split_node_rate;

        if (rng < merge_node_rate) {
            modified = g->merge_node(mu, sigma, new_node_type, max_recurrent_depth, edge_innovation_count, node_innovation_count);
            cout << "\tmerging node, modified: " << modified << endl;
            if (modified) g->set_generated_by("merge_node(" + node_type_str + ")");
            continue;
        }
        rng -= merge_node_rate;
    }

    vector<double> new_parameters;
    g->get_weights(new_parameters);
    //cout << "getting mu/sigma before assign reachability" << endl;
    //g->get_mu_sigma(new_parameters, mu, sigma);

    g->assign_reachability();

    //reset the genomes statistics (as these carry over on copy)
    g->best_validation_mse = EXAMM_MAX_DOUBLE;
    g->best_validation_mae = EXAMM_MAX_DOUBLE;

    //get the new set of parameters (as new paramters may have been
    //added duriung mutatino) and set them to the initial parameters
    //for epigenetic_initialization
    g->get_weights(new_parameters);
    g->initial_parameters = new_parameters;

    //cout << "checking parameters after mutation" << endl;
    //g->get_mu_sigma(g->initial_parameters, mu, sigma);

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
            cerr << "ERROR in crossover! trying to push an edge with innovation_number: " << edge->get_innovation_number() << " and it already exists in the vector!" << endl;
            /*
            cerr << "p1_position: " << p1_position << ", p1_size: " << p1_child_edges.size() << endl;
            cerr << "p2_position: " << p2_position << ", p2_size: " << p2_child_edges.size() << endl;
            cerr << "vector innovation numbers: " << endl;
            */
            for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                cerr << "\t" << child_edges[i]->get_innovation_number() << endl;
            }

            cerr << "This should never happen!" << endl;
            exit(1);

            return;
        } else if (child_edges[i]->get_input_innovation_number() == edge->get_input_innovation_number() &&
                child_edges[i]->get_output_innovation_number() == edge->get_output_innovation_number()) {

            cerr << "Not inserting edge in crossover operation as there was already an edge with the same input and output innovation numbers!" << endl;
            return;
        }
    }

    vector<double> new_input_weights, new_output_weights;
    double new_weight = 0.0;
    if (second_edge != NULL) {
        double crossover_value = rng_crossover_weight(generator);
        new_weight = crossover_value * (second_edge->weight - edge->weight) + edge->weight;

        //cout << "EDGE WEIGHT CROSSOVER :: " << "better: " << edge->weight << ", worse: " << second_edge->weight << ", crossover_value: " << crossover_value << ", new_weight: " << new_weight << endl;

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
            //cout << "\tnew input weights[" << i <<  "]: " << new_input_weights[i] << endl;
        }

        for (int32_t i = 0; i < (int32_t)new_output_weights.size(); i++) {
            new_output_weights[i] = crossover_value * (output_weights2[i] - output_weights1[i]) + output_weights1[i];
            //cout << "\tnew output weights[" << i <<  "]: " << new_output_weights[i] << endl;
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
            cerr << "ERROR in crossover! trying to push an recurrent_edge with innovation_number: " << recurrent_edge->get_innovation_number() << " and it already exists in the vector!" << endl;
            /*
            cerr << "p1_position: " << p1_position << ", p1_size: " << p1_child_recurrent_edges.size() << endl;
            cerr << "p2_position: " << p2_position << ", p2_size: " << p2_child_recurrent_edges.size() << endl;
            cerr << "vector innovation numbers: " << endl;
            */
            for (int32_t i = 0; i < (int32_t)child_recurrent_edges.size(); i++) {
                cerr << "\t" << child_recurrent_edges[i]->get_innovation_number() << endl;
            }

            cerr << "This should never happen!" << endl;
            exit(1);

            return;
        } else if (child_recurrent_edges[i]->get_input_innovation_number() == recurrent_edge->get_input_innovation_number() &&
                child_recurrent_edges[i]->get_output_innovation_number() == recurrent_edge->get_output_innovation_number()) {

            cerr << "Not inserting recurrent_edge in crossover operation as there was already an recurrent_edge with the same input and output innovation numbers!" << endl;
            return;
        }
    }


    vector<double> new_input_weights, new_output_weights;
    double new_weight = 0.0;
    if (second_edge != NULL) {
        double crossover_value = rng_crossover_weight(generator);
        new_weight = crossover_value * (second_edge->weight - recurrent_edge->weight) + recurrent_edge->weight;

        //cout << "RECURRENT EDGE WEIGHT CROSSOVER :: " << "better: " << recurrent_edge->weight << ", worse: " << second_edge->weight << ", crossover_value: " << crossover_value << ", new_weight: " << new_weight << endl;

        vector<double> input_weights1, input_weights2, output_weights1, output_weights2;
        recurrent_edge->get_input_node()->get_weights(input_weights1);
        recurrent_edge->get_output_node()->get_weights(output_weights1);

        second_edge->get_input_node()->get_weights(input_weights2);
        second_edge->get_output_node()->get_weights(output_weights2);

        new_input_weights.resize(input_weights1.size());
        new_output_weights.resize(output_weights1.size());

        for (int32_t i = 0; i < (int32_t)new_input_weights.size(); i++) {
            new_input_weights[i] = crossover_value * (input_weights2[i] - input_weights1[i]) + input_weights1[i];
            //cout << "\tnew input weights[" << i <<  "]: " << new_input_weights[i] << endl;
        }

        for (int32_t i = 0; i < (int32_t)new_output_weights.size(); i++) {
            new_output_weights[i] = crossover_value * (output_weights2[i] - output_weights1[i]) + output_weights1[i];
            //cout << "\tnew output weights[" << i <<  "]: " << new_output_weights[i] << endl;
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
    cerr << "generating new genome by crossover!" << endl;
    cout << "p1->island: " << p1->get_island() << ", p2->island: " << p2->get_island() << endl;

    double _mu, _sigma;
    cout << "getting p1 mu/sigma!" << endl;
    if (p1->best_parameters.size() == 0) {
        p1->set_weights(p1->initial_parameters);
        p1->get_mu_sigma(p1->initial_parameters, _mu, _sigma);
    } else {
        p1->set_weights(p1->best_parameters);
        p1->get_mu_sigma(p1->best_parameters, _mu, _sigma);
    }

    cout << "getting p2 mu/sigma!" << endl;
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

    /*
    cerr << "\tp1 innovation numbers AFTER SORT: " << endl;
    for (int32_t i = 0; i < (int32_t)p1_edges.size(); i++) {
        cerr << "\t\t" << p1_edges[i]->innovation_number << endl;
    }
    cerr << "\tp2 innovation numbers AFTER SORT: " << endl;
    for (int32_t i = 0; i < (int32_t)p2_edges.size(); i++) {
        cerr << "\t\t" << p2_edges[i]->innovation_number << endl;
    }
    */

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


    if (p1->get_island() == p2->get_island()) {
        child->set_generated_by("crossover");
    } else {
        child->set_generated_by("island_crossover");
    }
    initialize_genome_parameters(child);

    double mu, sigma;

    vector<double> new_parameters;
    child->get_weights(new_parameters);
    cout << "getting mu/sigma before assign reachability" << endl;
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

    cout << "checking parameters after crossover" << endl;
    child->get_mu_sigma(child->initial_parameters, mu, sigma);

    child->best_parameters.clear();

    return child;
}
