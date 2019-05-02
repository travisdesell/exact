/*
-- This class is for ants to choose paths along the neural network structure.
-- The class will generate a fixed structure for the neural network and then make copies for
   the ants to march over and select their paths.
-- As the class generates the neural network, it will also generate a colony which will hold the
   pheromones of the ants.
-- Ants will also pick a the nodes types for the neural network as they march through the neural network.
-- Class can reward the fit genomes (colonies) and increase their phermones. It can also penalize them.
-- The class will also decrease the phermones periodically as throught the iterations.
**--AbdElRahman--**
*/


#include <algorithm>
using std::sort;

#include <chrono>

#include <stdlib.h>

#include <fstream>
using std::ofstream;

#include <iomanip>
using std::setw;
using std::setprecision;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <ios>
using std::hex;
using std::ios;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include<map>
using std::map;

#include <string>
using std::string;
using std::to_string;

//for mkdir
#include <sys/stat.h>

#include <cmath>
#include <algorithm>

#include "acnnto.hxx"
#include "edge_pheromone.hxx"
#include "node_pheromone.hxx"
#include "rnn_genome.hxx"
#include "generate_nn.hxx"

ACNNTO::~ACNNTO() {
    RNN_Genome *genome;
    while (population.size()>0){
      genome = population.back();
      delete genome;
      population.pop_back();
    }

    for (auto const& c : colony){
        int p_size = c.second->pheromone_lines->size();
        for ( int j=0; j<p_size; j++){
            delete c.second->pheromone_lines->at(j);
        }
        delete c.second->pheromone_lines;
        delete c.second->type_pheromones;

    }
    for (auto const& c : colony){
        colony.erase(c.first);
    }
    cout<<"INFO: CLEANED CLONY!"<<endl;
}



ACNNTO::ACNNTO(int32_t _population_size, int32_t _max_genomes, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs, int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold, double _low_threshold, string _output_directory, int32_t _ants, int32_t _hidden_layers_depth, int32_t _hidden_layer_nodes, double _pheromone_decay_parameter, double _pheromone_update_strength, double _pheromone_heuristic, int32_t _max_recurrent_depth, double _weight_reg_parameter) : population_size(_population_size), max_genomes(_max_genomes), number_inputs(_input_parameter_names.size()), number_outputs(_output_parameter_names.size()), bp_iterations(_bp_iterations), learning_rate(_learning_rate), use_high_threshold(_use_high_threshold), high_threshold(_high_threshold), use_low_threshold(_use_low_threshold), low_threshold(_low_threshold), output_directory(_output_directory), ants(_ants), hidden_layers_depth(_hidden_layers_depth), hidden_layer_nodes(_hidden_layer_nodes), pheromone_decay_parameter(_pheromone_decay_parameter), pheromone_update_strength(_pheromone_update_strength), pheromone_heuristic(_pheromone_heuristic), max_recurrent_depth(_max_recurrent_depth), weight_reg_parameter(_weight_reg_parameter) {

    input_parameter_names   = _input_parameter_names;
    output_parameter_names  = _output_parameter_names;
    normalize_mins          = _normalize_mins;
    normalize_maxs          = _normalize_maxs;

    inserted_genomes  = 0;
    generated_genomes = 0;
    total_bp_epochs   = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    // population = vector<RNN_Genome*>;

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);

    epigenetic_weights = true;


    possible_node_types.clear();
    possible_node_types.push_back(FEED_FORWARD_NODE);
    possible_node_types.push_back(JORDAN_NODE);
    possible_node_types.push_back(ELMAN_NODE);
    possible_node_types.push_back(UGRNN_NODE);
    possible_node_types.push_back(MGU_NODE);
    possible_node_types.push_back(GRU_NODE);
    possible_node_types.push_back(LSTM_NODE);
    possible_node_types.push_back(DELTA_NODE);

    node_types[FEED_FORWARD_NODE]   = 0;
    node_types[UGRNN_NODE]          = 1;
    node_types[MGU_NODE]            = 2;
    node_types[GRU_NODE]            = 3;
    node_types[DELTA_NODE]          = 4;
    node_types[LSTM_NODE]           = 5;


    if (output_directory != "") {
        mkdir(output_directory.c_str(), 0777);
        log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
        (*log_file) << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges, ants, initial_hidden_layers, initial_hidden_layer_nodes" << endl;
        memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges, ants, initial_hidden_layers, initial_hidden_layer_nodes" << endl;
    } else {
        log_file = NULL;
    }

    startClock = std::chrono::system_clock::now();


    create_colony_pheromones(number_inputs, hidden_layers_depth, hidden_layer_nodes, number_outputs, max_recurrent_depth, colony, INITIAL_PHEROMONE, colony_edges) ;
    if ( bias_forward_paths ) bias_for_forward_paths() ;
    if ( bias_edges ) bias_for_edges() ;
}

ACNNTO::ACNNTO(int32_t _population_size, int32_t _max_genomes, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs, int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold, double _low_threshold, string _output_directory, int32_t _ants, int32_t _hidden_layers_depth, int32_t _hidden_layer_nodes, double _pheromone_decay_parameter, double _pheromone_update_strength, double _pheromone_heuristic, int32_t _max_recurrent_depth, double _weight_reg_parameter, string _reward_type, string _norm, bool _bias_forward_paths, bool _bias_edges, bool _use_pheromone_weight_update, bool _use_edge_inherited_weights, bool _use_fitness_weight_update) : population_size(_population_size), max_genomes(_max_genomes), number_inputs(_input_parameter_names.size()), number_outputs(_output_parameter_names.size()), bp_iterations(_bp_iterations), learning_rate(_learning_rate), use_high_threshold(_use_high_threshold), high_threshold(_high_threshold), use_low_threshold(_use_low_threshold), low_threshold(_low_threshold), output_directory(_output_directory), ants(_ants), hidden_layers_depth(_hidden_layers_depth), hidden_layer_nodes(_hidden_layer_nodes), pheromone_decay_parameter(_pheromone_decay_parameter), pheromone_update_strength(_pheromone_update_strength), pheromone_heuristic(_pheromone_heuristic), max_recurrent_depth(_max_recurrent_depth), weight_reg_parameter(_weight_reg_parameter), reward_type(_reward_type), norm(_norm), bias_forward_paths(_bias_forward_paths), bias_edges(_bias_edges), use_pheromone_weight_update(_use_pheromone_weight_update), use_edge_inherited_weights(_use_edge_inherited_weights), use_fitness_weight_update(_use_fitness_weight_update) {

    input_parameter_names   = _input_parameter_names;
    output_parameter_names  = _output_parameter_names;
    normalize_mins          = _normalize_mins;
    normalize_maxs          = _normalize_maxs;

    inserted_genomes  = 0;
    generated_genomes = 0;
    total_bp_epochs   = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    // population = vector<RNN_Genome*>;

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);

    epigenetic_weights = true;


    possible_node_types.clear();
    possible_node_types.push_back(FEED_FORWARD_NODE);
    possible_node_types.push_back(JORDAN_NODE);
    possible_node_types.push_back(ELMAN_NODE);
    possible_node_types.push_back(UGRNN_NODE);
    possible_node_types.push_back(MGU_NODE);
    possible_node_types.push_back(GRU_NODE);
    possible_node_types.push_back(LSTM_NODE);
    possible_node_types.push_back(DELTA_NODE);

    node_types[FEED_FORWARD_NODE]   = 0;
    node_types[UGRNN_NODE]          = 1;
    node_types[MGU_NODE]            = 2;
    node_types[GRU_NODE]            = 3;
    node_types[DELTA_NODE]          = 4;
    node_types[LSTM_NODE]           = 5;


    if (output_directory != "") {
        mkdir(output_directory.c_str(), 0777);
        log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
        (*log_file) << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges, ants, initial_hidden_layers, initial_hidden_layer_nodes" << endl;
        memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges, ants, initial_hidden_layers, initial_hidden_layer_nodes" << endl;
    } else {
        log_file = NULL;
    }

    startClock = std::chrono::system_clock::now();

    create_colony_pheromones(number_inputs, hidden_layers_depth, hidden_layer_nodes, number_outputs, max_recurrent_depth, colony, INITIAL_PHEROMONE, colony_edges) ;
    if ( bias_forward_paths ) bias_for_forward_paths() ;
    if ( bias_edges ) bias_for_edges() ;
}


void ACNNTO::print_last_population(){
    (*log_file) << "POPULATIONS: " << endl;
    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        (*log_file) << "\tPOPULATION " << i << ":" << endl;

        (*log_file) << "\t" << RNN_Genome::print_statistics_header() << endl;

        (*log_file) << "\t" << population[i]->print_statistics() << endl;
    }
}

void ACNNTO::print_population() {
    cout << "POPULATIONS: " << endl;
    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        cout << "\tPOPULATION " << i << ":" << endl;

        cout << "\t" << RNN_Genome::print_statistics_header() << endl;

        cout << "\t" << population[i]->print_statistics() << endl;
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
                cerr << "ERROR, could not open ACNNTO output log: '" << output_file << "'" << endl;
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
            << "," << best_genome->get_enabled_recurrent_edge_count()
            << "," << ants
            << "," << hidden_layers_depth
            << "," << hidden_layer_nodes << endl;


        memory_log << inserted_genomes
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count()
            << "," << ants
            << "," << hidden_layers_depth
            << "," << hidden_layer_nodes << endl;
    }
}

void ACNNTO::write_memory_log(string filename) {
    ofstream log_file(filename);
    log_file << memory_log.str();
    log_file.close();
}

void ACNNTO::set_possible_node_types(vector<string> possible_node_type_strings) {
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

int32_t ACNNTO::population_contains(RNN_Genome* genome) {
    for (int32_t j = 0; j < (int32_t)population.size(); j++) {
        if (population[j]->equals(genome)) {
            return j;
        }
    }

    return -1;
}

bool ACNNTO::populations_full() const {
    if (population.size() < population_size) return false;
    return true;
}

string ACNNTO::get_output_directory() const {
    return output_directory;
}

RNN_Genome* ACNNTO::get_best_genome() {
  if (population.size() <= 0) {
      return NULL;
  } else {
      return population[0];
  }
}

RNN_Genome* ACNNTO::get_worst_genome() {
  if (population.size() <= 0) {
      return NULL;
  } else {
      return population.back();
  }
}



/* Will increase the phermones of the edges when
   the foreword edges are fewer than the backward edges */
void ACNNTO::bias_for_forward_paths() {

    for (auto const& c : colony){
        int forward_nodes = 0 ;
        int backward_nodes = 0 ;
        double total_forward_pheromones = 0.0 ;
        double total_backward_pheromones = 0.0 ;



        for ( int e=0; e<c.second->pheromone_lines->size(); e++){
            if ( c.second->get_current_layer() < colony[c.second->pheromone_lines->at(e)->get_output_innovation_number()]->get_current_layer() ) {
                // cout << "\t Forward_nodes: " << c.second->get_current_layer() << " -> " << colony[c.second->pheromone_lines->at(e)->get_output_innovation_number()]->get_current_layer() << endl ;
                forward_nodes++ ;
                total_forward_pheromones+= c.second->pheromone_lines->at(e)->get_edge_phermone() ;
            }
            else{
                // cout << "\t Backward_nodes: " << c.second->get_current_layer() << " <- " << colony[c.second->pheromone_lines->at(e)->get_output_innovation_number()]->get_current_layer() << endl ;
                backward_nodes++ ;
                total_backward_pheromones+= c.second->pheromone_lines->at(e)->get_edge_phermone() ;
            }
        }


        if ( forward_nodes < (int)( backward_nodes * 0.75 ) && total_forward_pheromones < 1.00 * total_backward_pheromones ) {
            for ( int e=0; e<c.second->pheromone_lines->size(); e++){
                if ( c.second->get_current_layer() < colony[c.second->pheromone_lines->at(e)->get_output_innovation_number()]->get_current_layer() ) {
                    c.second->pheromone_lines->at(e)->set_edge_phermone( ( c.second->pheromone_lines->at(e)->get_edge_phermone() / total_forward_pheromones ) * ( total_backward_pheromones ) ) ;

                }
            }
        }

    }
    bias_for_edges() ;
}
void ACNNTO::bias_for_edges() {

    for (auto const& c : colony){
        double total_edges_pheromones = 0.0 ;
        double total_recurrent_edges_pheromones = 0.0 ;
        int no_edges = 0 ;
        int no_recurrent_edges = 0 ;
        double max_pheromone_value = -1.0 ;

        for ( int e=0; e<c.second->pheromone_lines->size(); e++){
            if ( c.second->pheromone_lines->at(e)->get_depth() == 0 ) {
                no_edges+=1 ;
                total_edges_pheromones+= c.second->pheromone_lines->at(e)->get_edge_phermone() ;
            }
            else{
                no_recurrent_edges+=1 ;
                total_recurrent_edges_pheromones+= c.second->pheromone_lines->at(e)->get_edge_phermone() ;
            }
        }


        if ( no_edges < no_recurrent_edges && total_edges_pheromones < total_recurrent_edges_pheromones ) {
            for ( int e=0; e<c.second->pheromone_lines->size(); e++){
                if ( c.second->pheromone_lines->at(e)->get_depth() == 0 ) {
                    c.second->pheromone_lines->at(e)->set_edge_phermone( ( c.second->pheromone_lines->at(e)->get_edge_phermone() / total_edges_pheromones ) * ( total_recurrent_edges_pheromones ) ) ;
                    if ( c.second->pheromone_lines->at(e)->get_edge_phermone() > max_pheromone_value ) {
                        max_pheromone_value = c.second->pheromone_lines->at(e)->get_edge_phermone() ;
                    }
                }
            }
        }

        //Return the maximun Pheromone Value
        if ( reward_type == "fixed" && max_pheromone_value > PHEROMONES_THERESHOLD ) {
            for ( int e=0; e<c.second->pheromone_lines->size(); e++){
                c.second->pheromone_lines->at(e)->set_edge_phermone( ( c.second->pheromone_lines->at(e)->get_edge_phermone() / max_pheromone_value ) * ( PHEROMONES_THERESHOLD ) ) ;
            }
        }
        else if ( reward_type == "constant" ) {

        }
        else if ( reward_type == "regularized" && max_pheromone_value > max_pheromone ) {
            for ( int e=0; e<c.second->pheromone_lines->size(); e++){
                c.second->pheromone_lines->at(e)->set_edge_phermone( ( c.second->pheromone_lines->at(e)->get_edge_phermone() / max_pheromone_value ) * ( max_pheromone ) ) ;
            }
        }

    }
}

double ACNNTO::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXALT_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double ACNNTO::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXALT_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

//this will insert a COPY, original needs to be deleted
bool ACNNTO::insert_genome(RNN_Genome* genome) {
    if (!genome->sanity_check()) {
        cerr << "ERROR, genome failed sanity check on insert!" << endl;
        exit(1);
    }

    double new_fitness = genome->get_fitness();
    bool was_inserted = true;

    inserted_genomes++;
    total_bp_epochs += genome->get_bp_iterations();

    genome->update_generation_map(generated_from_map);

    cout << "genomes evaluated: " << setw(10) << inserted_genomes << ", inserting: " << parse_fitness(genome->get_fitness()) << endl;


    if ( use_fitness_weight_update && population.size() > 1 && population.front()->get_fitness() != population.back()->get_fitness() ) {
        phi = std::min (
                    std::max (
                            1  - (
                                ( new_fitness - population.back()->get_fitness() ) /
                                    ( population.front()->get_fitness() - population.back()->get_fitness() )
                            ), 0.0
                        )
                , 0.9 ) ;
        set_colony_best_weights ( genome ) ;
    }

    if (population.size() >= population_size  && new_fitness > population.back()->get_fitness()) {
        cout << "ignoring genome, fitness: " << new_fitness << " > worst population" << " fitness: " << population.back()->get_fitness() << endl;
        print_population();
        if ( reward_type == "fixed" ) reward_colony_fixed_rate(genome, 0.85);
        else if ( reward_type == "constant" ) reward_colony_constant_rate(genome, -1) ;
        else if ( reward_type == "regularized" ) reward_colony_regularization(genome, false) ;
        else {
            cerr << "ERROR: Invalidm Pheromone Rewarding Method" << endl;
            exit(1);
        }
        return false;
    }

    int32_t duplicate_genome = population_contains(genome);
    if (duplicate_genome >= 0) {
        //if fitness is better, replace this genome with new one
        cout << "found duplicate at position: " << duplicate_genome << endl;

        RNN_Genome *duplicate = population[duplicate_genome];
        if (duplicate->get_fitness() > new_fitness) {
            //erase the genome with loewr fitness from the vector;
            cout << "REPLACING DUPLICATE GENOME, fitness of genome in search: " << parse_fitness(duplicate->get_fitness()) << ", new fitness: " << parse_fitness(genome->get_fitness()) << endl;
            population.erase(population.begin() + duplicate_genome);
            delete duplicate;
            if ( reward_type == "fixed" ) reward_colony_fixed_rate(genome, 1.15);
            else if ( reward_type == "constant" ) reward_colony_constant_rate(genome, 1) ;
            else if ( reward_type == "regularized" ) {
                reward_colony_regularization(genome, true) ;
                max_pheromone = 1 / ( pheromone_decay_parameter * duplicate->get_fitness() ) ;
            }
            else {
                std::cerr << "ERROR: Invalid Rewarding Method" << '\n';
                exit(1);
            }

            /* Favoring forward paths by increasing their pheromones */
            if ( bias_forward_paths )
                bias_for_forward_paths() ;
            if ( bias_edges )
                bias_for_edges() ;

            /* Recording best weights in the colony */
            if ( use_edge_inherited_weights )
                set_colony_best_weights( genome );

        } else {
            cerr << "\tpopulation already contains genome! not inserting." << endl;
            print_population();
            return false;
        }
    }

    if (population.size() < population_size || population.back()->get_fitness() > new_fitness) {
        //this genome will be inserted
        was_inserted = true;
        if ( reward_type == "fixed" ) reward_colony_fixed_rate(genome, 1.15);
        else if ( reward_type == "constant" ) reward_colony_constant_rate(genome, 1) ;
        else if ( reward_type == "regularized" ) {
            reward_colony_regularization(genome, true) ;
            max_pheromone = 1 / ( pheromone_decay_parameter * new_fitness ) ;
        }
        else {
            std::cerr << "ERROR: Invalid Rewarding Method" << '\n';
            exit(1);
        }

        if ( bias_forward_paths )
            bias_for_forward_paths() ;
        if ( bias_edges )
            bias_for_edges() ;

        /* Recording best weights in the colony */
        if ( use_edge_inherited_weights )
            set_colony_best_weights( genome );


        if (population.size() == 0 || genome->get_fitness() < get_best_genome()->get_fitness()) {

            if (genome->get_fitness() != EXALT_MAX_DOUBLE) {
                //need to set the weights for non-initial genomes so we
                //can generate a proper graphviz file
                vector<double> best_parameters = genome->get_best_parameters();
                genome->set_weights(best_parameters);
            }

            genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".gv");
            genome->write_to_file(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".bin", false);

        }

        cout << "inserting new genome to population " << endl;
        //inorder insert the new individual
        RNN_Genome *copy = genome->copy();

        population.insert( upper_bound(population.begin(), population.end(), copy, sort_genomes_by_fitness()), copy);
        cout << "finished insert" << endl;

        //delete the worst individual if we've reached the population size
        if ((int32_t)population.size() > population_size) {
            cout << "deleting worst genome" << endl;
            RNN_Genome *worst = population.back();
            population.pop_back();

            delete worst;
        }
    } else {
        was_inserted = false;
        cout << "not inserting genome due to poor fitness" << endl;
    }

    print_population();

    cout << "printed population!" << endl;

    if ( use_pheromone_weight_update )
        pheromones_update_weights( genome );

    return was_inserted;
}

void ACNNTO::set_colony_best_weights( RNN_Genome* g ) {
    vector<double> best_parameters = g->get_best_parameters() ;
    int count = 0 ;
    for ( int i=0; i<g->nodes.size(); i++){
        count+= g->nodes[i]->get_number_weights() ;
    }
    // int c = 0 ;
    while ( count < best_parameters.size() ) {
        for ( int i=0; i<g->edges.size(); i++ ) {
            if (g->edges[i]->enabled){
                int32_t edge_inno = g->edges[i]->get_innovation_number() ;
                // cout << "BEFORE UPDATE:: Best Genome Weights: " << colony_edges[edge_inno]->edge_weight << endl ;
                if ( use_fitness_weight_update )
                    colony_edges[edge_inno]->edge_weight = ( phi * best_parameters[count++] ) + ( ( 1 - phi ) *  colony_edges[edge_inno]->edge_weight ) ;
                else
                    colony_edges[edge_inno]->edge_weight = best_parameters[count++] ;
                // cout << "AFTER UPDATE:: Best Genome Weights: " << colony_edges[edge_inno]->edge_weight << endl ;
                // cout << "________________________\n" ;

                /*
                for ( int j=0; j<colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++) {
                    if ( edge_inno == colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ) {
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_weight = best_parameters[k] ;
                    }
                }
                */
            }
        }
        for ( int i=0; i<g->recurrent_edges.size(); i++ ) {
            if (g->recurrent_edges[i]->enabled){
                int32_t edge_inno = g->recurrent_edges[i]->get_innovation_number() ;
                double before = colony_edges[edge_inno]->edge_weight;
                // cout << "BEFORE UPDATE:: Best Genome Weights: " << colony_edges[edge_inno]->edge_weight << endl;
                if ( use_fitness_weight_update )
                    colony_edges[edge_inno]->edge_weight = ( phi * best_parameters[count++] ) + ( ( 1 - phi ) *  colony_edges[edge_inno]->edge_weight ) ;
                else
                    colony_edges[edge_inno]->edge_weight = best_parameters[count++] ;
                // cout << "AFTER UPDATE:: Best Genome Weights: " << colony_edges[edge_inno]->edge_weight << endl;
                // cout << "________________________\n" ;
                // if ( before - colony_edges[edge_inno]->edge_weight == 0 ) c++;
            }

        }
    }
    // cout << "Number of Changed Weights: " << c << endl;
    cout << "FINISHED UPDATING WEIGHTS\n" ;
}


EDGE_Pheromone* ACNNTO::pick_line(vector<EDGE_Pheromone*>* edges_pheromones){
  double sum_pheromones = 0;
  // cout << "________________________\n";
  // cout << "EDGE PHEROMONES SIZE: " << edges_pheromones->size() << endl;
  for ( int i=0; i<edges_pheromones->size(); i++){
      sum_pheromones+=edges_pheromones->at(i)->get_edge_phermone();
  }
  uniform_real_distribution<double> rng(0.0, 1.0);
  double rand_gen = rng(generator) * sum_pheromones;
  // cout << "rand_gen: " << rand_gen << endl ;
  // for ( int i = 0; i < edges_pheromones->size(); i++) {
  //     cout<<"Edge to : "<<edges_pheromones->at(i)->get_output_innovation_number()<<" Pheromone Value: " << edges_pheromones->at(i)->get_edge_phermone() << " EDGE DEPTH: " << edges_pheromones->at(i)->get_depth()<<endl;
  // }
  int a;

  for ( a = 0; a < edges_pheromones->size(); a++) {

      if ( rand_gen<=edges_pheromones->at(a)->get_edge_phermone() )
        break;
      else
        rand_gen-= edges_pheromones->at(a)->get_edge_phermone();
  }
  // cout << "A: " << a << endl ;
  // cout << "Edge Picked to: " << edges_pheromones->at(a)->get_output_innovation_number() << " Pheromone Value: " <<  edges_pheromones->at(a)->get_edge_phermone() << " EDGE DEPTH: " << edges_pheromones->at(a)->get_depth()<<endl;
  // cout << "________________________\n";
  return edges_pheromones->at(a);
}

int ACNNTO::pick_node_type(double* type_pheromones){
    uniform_real_distribution<double> rng(0.0, 1.0);
    double rand_gen = rng(generator);
    double highest_value = type_pheromones[0]; int a = 0;
    if ( rand_gen < EXPLORATION_FACTOR ){
        for ( int i=0; i< node_types.size(); i++){
            if (highest_value <  type_pheromones[i])
                highest_value = type_pheromones[i];
                a = i;
        }
    }
    else{
        double sum_pheromones = 0;
        for (int i=0; i<node_types.size(); i++){
          sum_pheromones+=type_pheromones[i];
        }
        highest_value =  type_pheromones[0] / sum_pheromones;; a = 0;
        for ( int i=1; i< node_types.size(); i++){
            int dum = type_pheromones[i] / sum_pheromones;
            if (highest_value < dum )
                highest_value = dum;
                a = i;
        }
    }
    cout<<"AA: "<<a<<endl;
    switch (a) {
    case 0: return FEED_FORWARD_NODE;
        break;
    case 1: return UGRNN_NODE;
        break;
    case 2: return MGU_NODE;
        break;
    case 3: return GRU_NODE;
        break;
    case 4: return DELTA_NODE;
        break;
    case 5: return LSTM_NODE;
        break;
    }
    cerr << "ERROR choosing node type! None existing type!!"<<endl;
    return -1;
}


int ACNNTO::old_pick_node_type(double* type_pheromones){
  double sum_pheromones = 0;
  for (int i=0; i<node_types.size(); i++){
    sum_pheromones+=type_pheromones[i];
  }
  uniform_real_distribution<double> rng(0.0, 1.0);
  double rand_gen = rng(generator) * sum_pheromones;
  for ( int i=0; i<node_types.size(); i++){
  }
  int a;
  for ( a = 0; a < node_types.size(); a++) {
      if ( rand_gen<=type_pheromones[a] )
        break;
      else
        rand_gen-= type_pheromones[a];
  }
  // cout<<"Ant Choice: "<<a<<endl;
  // cout<<"IN PICK NODE TYPE\n";
  // getchar();
  switch (a) {
  case 0: return FEED_FORWARD_NODE;
      break;
  case 1: return UGRNN_NODE;
      break;
  case 2: return MGU_NODE;
      break;
  case 3: return GRU_NODE;
      break;
  case 4: return DELTA_NODE;
      break;
  case 5: return LSTM_NODE;
      break;
  }
  cerr << "ERROR choosing node type! None existing type!!"<<endl;
  return -1;
}

int32_t ACNNTO::get_colony_number_edges(){
    int32_t count = 0 ;
    for (auto const& c : colony){
        count+= c.second->pheromone_lines->size();
    }
    return count;
}

/*
  If genome fitness is good compared to the population=>
      reward it and increase the pheromones on its active nodes and edges
  If genome fitness is bad=>
      penalize it and reduce the pheromones on its active nodes and edges
*/


double ACNNTO::get_norm(RNN_Genome* g) {
    double weights_sum = 0.0;
    if ( norm == "l1" ) {
        cout << "Regularizing Pheromones Using L1 Norm" << endl;
        for (int i = 0; i < g->edges.size(); i++) {
            weights_sum += g->edges[i]->weight ;
        }
        for (int i = 0; i < g->recurrent_edges.size(); i++) {
            weights_sum += g->recurrent_edges[i]->weight ;
        }
        return ( weight_reg_parameter * weights_sum / ( g->recurrent_edges.size() + g->edges.size() ) );
    }
    else if ( norm == "l2" ) {
        cout << "Regularizing Pheromones Using L2 Norm" << endl;
        for (int i = 0; i < g->edges.size(); i++) {
            weights_sum += g->edges[i]->weight * g->edges[i]->weight ;
        }
        for (int i = 0; i < g->recurrent_edges.size(); i++) {
            weights_sum += g->recurrent_edges[i]->weight * g->recurrent_edges[i]->weight ;
        }
        return ( weight_reg_parameter * weights_sum / ( 2 * ( g->recurrent_edges.size() + g->edges.size() ) ) );
    }
    else {
        cerr << "ERROR: Wrong Norm Option: " <<  norm << endl ;
        exit(0) ;
    }
}

void ACNNTO::reward_colony_regularization(RNN_Genome* g, bool reward){
    double pheromone_decay_parameter_ = 0;
    // double pheromone_decay_parameter_ = pheromone_decay_parameter;
    double fitness = g->get_fitness();
    if ( fitness == 0 )
        fitness = 0.00000001 ;
    if (reward) cout << "REWARDING COLONY!" << endl ;
    else        cout << "PENALIZING COLONY!" << endl ;
    // double max_pheromone = 1 / ( pheromone_decay_parameter * fitness ) ;

    double min_pheromone =  max_pheromone / ( 2 * node_types.size() );

    g->set_weights(g->get_best_parameters());

    double W_reg = 0.0;
    if (reward && weight_reg_parameter!=0.0)
        W_reg =  get_norm( g ) ;

    cout << "ACNNTO-REWARD_COLONY_REGULARIZATION:: W_reg: " << W_reg << endl ;
    cout << "ACNNTO-REWARD_COLONY_REGULARIZATION:: 1 / ( fitness + W_reg ): " << 1 / ( fitness + W_reg ) << endl ;

    for ( int i=0; i<g->nodes.size(); i++){
        if ( g->nodes[i]->enabled==true ){
            int32_t node_inno = g->nodes[i]->get_innovation_number();
            int32_t node_type = g->nodes[i]->node_type;
            // cout << "NODE ID: " << node_inno << endl;
            // cout << "\tNODE LAYER: " << g->nodes[i]->layer_type << endl;
            if (i==g->nodes.size()-1)
                node_inno*=-1;

            double pheromone_update;

            if ( reward )
                pheromone_update = ( 1 - pheromone_decay_parameter_ ) * colony[node_inno]->type_pheromones[node_types[node_type]] + pheromone_decay_parameter * ( 1 / fitness ) ;
            else
                pheromone_update = ( 1 - pheromone_update_strength ) * colony[node_inno]->type_pheromones[node_types[node_type]] + pheromone_update_strength * INITIAL_PHEROMONE  ;

            if ( pheromone_update > max_pheromone )
                colony[node_inno]->type_pheromones[node_types[node_type]] = max_pheromone;
            else if ( pheromone_update < min_pheromone )
                colony[node_inno]->type_pheromones[node_types[node_type]] = min_pheromone;
            else
                colony[node_inno]->type_pheromones[node_types[node_type]] = pheromone_update;


            min_pheromone = max_pheromone / ( 2 * get_colony_number_edges() ) ;

            if (g->nodes[i]->layer_type==0){
                for ( int m=0; m < colony[-1]->pheromone_lines->size(); m++ ){
                    if ( colony[-1]->pheromone_lines->at(m)->get_output_innovation_number() == g->nodes[i]->innovation_number ){
                        double pheromone_update;
                        // cout << "\t\t>>BEFORE: EDGE PHEOMONE VALUE: " << colony[-1]->pheromone_lines->at(m)->edge_pheromone << endl ;
                        if ( reward )
                            pheromone_update = ( 1 - pheromone_decay_parameter_ ) * colony[-1]->pheromone_lines->at(m)->edge_pheromone + pheromone_decay_parameter * ( 1 / ( fitness + W_reg) ) ;
                        else
                            pheromone_update = ( 1 - pheromone_update_strength ) * colony[-1]->pheromone_lines->at(m)->edge_pheromone + pheromone_update_strength * INITIAL_PHEROMONE  ;

                        if ( pheromone_update > max_pheromone )
                            colony[-1]->pheromone_lines->at(m)->edge_pheromone = max_pheromone ;
                        else if ( pheromone_update < min_pheromone  )
                            colony[-1]->pheromone_lines->at(m)->edge_pheromone = min_pheromone ;
                        else
                            colony[-1]->pheromone_lines->at(m)->edge_pheromone = pheromone_update ;
                        // cout << "\t\t<<AFTER: EDGE PHEOMONE VALUE: " << colony[-1]->pheromone_lines->at(m)->edge_pheromone << endl ;
                    }
                }
            }
        }
    }

    for ( int i=0; i<g->edges.size(); i++){
        if (g->edges[i]->enabled){
            int32_t edge_inno = g->edges[i]->get_innovation_number() ;


            double pheromone_update ;
            if (reward)
                pheromone_update = ( 1 - pheromone_decay_parameter_ ) * colony_edges[edge_inno]->edge_pheromone + pheromone_decay_parameter * ( 1 / ( fitness + W_reg) ) ;
            else
                pheromone_update = ( 1 - pheromone_update_strength)   * colony_edges[edge_inno]->edge_pheromone + pheromone_update_strength  * INITIAL_PHEROMONE  ;
            if ( pheromone_update > max_pheromone )
                colony_edges[edge_inno]->edge_pheromone = max_pheromone ;
            else if ( pheromone_update < min_pheromone  )
                colony_edges[edge_inno]->edge_pheromone = min_pheromone ;
            else
                colony_edges[edge_inno]->edge_pheromone = pheromone_update ;

            /*
            for ( int j=0; j<colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    double pheromone_update;
                    // cout << "\t\tNODE INNO FROM GENOME: " << g->edges[i]->get_input_innovation_number() << endl ;
                    // cout << "\t\t>>BEFORE: EDGE PHEOMONE VALUE: " << colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone<< endl ;
                    if (reward)
                        pheromone_update = ( 1 - pheromone_decay_parameter_ ) * colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + pheromone_decay_parameter * ( 1 / ( fitness + W_reg) ) ;
                    else
                        pheromone_update = (1 - pheromone_update_strength) * colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + pheromone_update_strength  * INITIAL_PHEROMONE  ;
                    if ( pheromone_update > max_pheromone )
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = max_pheromone ;
                    else if ( pheromone_update < min_pheromone  )
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = min_pheromone ;
                    else
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = pheromone_update ;
                    // cout << "\t\t>>AFTER: EDGE PHEOMONE VALUE: " << colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone<< endl ;
                }
            }
            */
        }
    }

    for ( int i=0; i<g->recurrent_edges.size(); i++){
        if (g->recurrent_edges[i]->enabled){
            int32_t edge_inno = g->recurrent_edges[i]->get_innovation_number();

            double pheromone_update ;
            if (reward)
                pheromone_update = ( 1 - pheromone_decay_parameter_ ) * colony_edges[edge_inno]->edge_pheromone + pheromone_decay_parameter * ( 1 / ( fitness + W_reg) ) ;
            else
                pheromone_update = ( 1 - pheromone_update_strength)   * colony_edges[edge_inno]->edge_pheromone + pheromone_update_strength  * INITIAL_PHEROMONE  ;
            if ( pheromone_update > max_pheromone )
                colony_edges[edge_inno]->edge_pheromone = max_pheromone ;
            else if ( pheromone_update < min_pheromone  )
                colony_edges[edge_inno]->edge_pheromone = min_pheromone ;
            else
                colony_edges[edge_inno]->edge_pheromone = pheromone_update ;

            /*
            for ( int j=0; j<colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    double pheromone_update ;
                    if (reward)
                        pheromone_update = ( 1 - pheromone_decay_parameter_ ) * colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + pheromone_decay_parameter * ( 1 / ( fitness + W_reg) ) ;
                    else
                        pheromone_update = (1 - pheromone_update_strength) * colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + pheromone_update_strength  * INITIAL_PHEROMONE  ;
                    if ( pheromone_update > max_pheromone )
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = max_pheromone ;
                    else if ( pheromone_update < min_pheromone  )
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = min_pheromone ;
                    else
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = pheromone_update ;
                }
            }
            */
        }
    }
}
void ACNNTO::reward_colony(RNN_Genome* g, bool reward){
    double fitness = g->get_fitness();
    if ( fitness == 0 ) fitness = 0.00000001 ;
    if (reward) cout << "REWARDING COLONY!" << endl ;
    else        cout << "PENALIZING COLONY!" << endl ;
    double max_pheromone = 1 / ( pheromone_decay_parameter * fitness ) ;

    double min_pheromone = max_pheromone / ( 2 * g->nodes.size() * node_types.size() );


    for ( int i=0; i<g->nodes.size(); i++){
      if ( g->nodes[i]->enabled==true ){
            int32_t node_inno = g->nodes[i]->get_innovation_number();
            int32_t node_type = g->nodes[i]->node_type;
            if (i==g->nodes.size()-1)
                node_inno*=-1;

            double pheromone_update;

            if ( reward )
                pheromone_update = ( 1 - pheromone_decay_parameter ) * colony[node_inno]->type_pheromones[node_types[node_type]] + pheromone_decay_parameter * ( 1 / fitness ) ;
            else
                pheromone_update = ( 1 - pheromone_update_strength ) * colony[node_inno]->type_pheromones[node_types[node_type]] + pheromone_update_strength * INITIAL_PHEROMONE  ;

            if ( pheromone_update > max_pheromone )
                colony[node_inno]->type_pheromones[node_types[node_type]] = max_pheromone;
            else if ( pheromone_update < min_pheromone )
                colony[node_inno]->type_pheromones[node_types[node_type]] = min_pheromone;
            else
                colony[node_inno]->type_pheromones[node_types[node_type]] = pheromone_update;

            if (g->nodes[i]->layer_type==0){
                for ( int m=0; m < colony[-1]->pheromone_lines->size(); m++ ){
                    if ( colony[-1]->pheromone_lines->at(m)->get_output_innovation_number() == g->nodes[i]->innovation_number ){
                        if ( reward )
                            colony[-1]->pheromone_lines->at(m)->edge_pheromone = ( 1 - pheromone_decay_parameter ) * colony[-1]->pheromone_lines->at(m)->edge_pheromone + pheromone_decay_parameter * ( 1 / fitness ) ;
                        else
                            colony[-1]->pheromone_lines->at(m)->edge_pheromone = ( 1 - pheromone_update_strength ) * colony[-1]->pheromone_lines->at(m)->edge_pheromone + pheromone_update_strength * INITIAL_PHEROMONE  ;
                    }
                }
            }
      }
      if ( g->nodes[i]->get_depth() == 0.0 ) {
          for ( int j=0; j<colony[-1]->pheromone_lines->size(); j++){
              if ( g->nodes[i]->get_innovation_number() == colony[-1]->pheromone_lines->at(j)->get_output_innovation_number() ){
                  double treat_pheromone = 0.0;
                  if ( reward ) treat_pheromone = 1.15 ;
                  else treat_pheromone = 0.85 ;
                  colony[-1]->pheromone_lines->at(j)->edge_pheromone*=treat_pheromone;
                  if ( colony[-1]->pheromone_lines->at(j)->edge_pheromone > PHEROMONES_THERESHOLD )
                      colony[-1]->pheromone_lines->at(j)->edge_pheromone = PHEROMONES_THERESHOLD ;
              }
          }
      }
  }


  min_pheromone = max_pheromone / ( 2 * g->edges.size() );
  for ( int i=0; i<g->edges.size(); i++){
        if (g->edges[i]->enabled){
            int32_t edge_inno = g->edges[i]->get_innovation_number();
            for ( int j=0; j<colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    double pheromone_update;
                    if (reward)
                        pheromone_update = (1 - pheromone_decay_parameter) * colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + pheromone_decay_parameter * ( 1 / fitness );
                    else
                        pheromone_update = (1 - pheromone_update_strength) * colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + pheromone_update_strength  * INITIAL_PHEROMONE  ;
                    if ( pheromone_update > max_pheromone )
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = max_pheromone ;
                    else if ( pheromone_update < min_pheromone  )
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = min_pheromone ;
                    else
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = pheromone_update ;
                }
            }
        }
    }

  for ( int i=0; i<g->recurrent_edges.size(); i++){
        if (g->recurrent_edges[i]->enabled){
            int32_t edge_inno = g->recurrent_edges[i]->get_innovation_number();
            for ( int j=0; j<colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    double pheromone_update;
                    if (reward)
                        pheromone_update = (1 - pheromone_decay_parameter) * colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + pheromone_decay_parameter * ( 1 / fitness );
                    else
                        pheromone_update = (1 - pheromone_update_strength) * colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone + pheromone_update_strength  * INITIAL_PHEROMONE  ;
                    if ( pheromone_update > max_pheromone )
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = max_pheromone ;
                    else if ( pheromone_update < min_pheromone  )
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = min_pheromone ;
                    else
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = pheromone_update ;
                }
            }
        }
    }
}


void ACNNTO::reward_colony_fixed_rate(RNN_Genome* g, double treat_pheromone){

    if (treat_pheromone>1) cout << "REWARDING COLONY!" << endl ;
    else        cout << "PENALIZING COLONY!" << endl ;

    for ( int i=0; i<g->nodes.size(); i++){
        if (g->nodes[i]->enabled==true){
            int32_t node_inno = g->nodes[i]->get_innovation_number();
            int32_t node_type = g->nodes[i]->node_type;
            if (i==g->nodes.size()-1)
                node_inno*=-1;


            colony[node_inno]->type_pheromones[node_types[node_type]]*=treat_pheromone;
            if (colony[node_inno]->type_pheromones[node_types[node_type]] > PHEROMONES_THERESHOLD )
                colony[node_inno]->type_pheromones[node_types[node_type]] = PHEROMONES_THERESHOLD ;
        }


        if ( g->nodes[i]->get_depth() == 0.0 ) {
            for ( int j=0; j<colony[-1]->pheromone_lines->size(); j++){
                if ( g->nodes[i]->get_innovation_number() == colony[-1]->pheromone_lines->at(j)->get_output_innovation_number() ){
                    colony[-1]->pheromone_lines->at(j)->edge_pheromone*=treat_pheromone;
                    if ( colony[-1]->pheromone_lines->at(j)->edge_pheromone > PHEROMONES_THERESHOLD )
                        colony[-1]->pheromone_lines->at(j)->edge_pheromone = PHEROMONES_THERESHOLD ;
                }
            }
        }
    }


    for ( int i=0; i<g->edges.size(); i++){
        if (g->edges[i]->enabled){
            int32_t edge_inno = g->edges[i]->get_innovation_number();
            for ( int j=0; j<colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    if (colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*treat_pheromone < PHEROMONES_THERESHOLD){
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*=treat_pheromone;
                        break;
                    }
                    colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = PHEROMONES_THERESHOLD;
                    break;
                }
            }
        }
    }

    for ( int i=0; i<g->recurrent_edges.size(); i++){
        if (g->recurrent_edges[i]->enabled){
            int32_t edge_inno = g->recurrent_edges[i]->get_innovation_number();
            for ( int j=0; j<colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    if (colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*treat_pheromone < PHEROMONES_THERESHOLD){
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*=treat_pheromone;
                        break;
                    }
                    colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = PHEROMONES_THERESHOLD;
                    break;
                }
            }
        }
    }
}

void ACNNTO::reward_colony_constant_rate(RNN_Genome* g, int treat_pheromone){

    if ( treat_pheromone > 0 ) cout << "REWARDING COLONY!" << endl ;
    else        cout << "PENALIZING COLONY!" << endl ;

    for ( int i=0; i<g->nodes.size(); i++){
        if (g->nodes[i]->enabled==true){
            int32_t node_inno = g->nodes[i]->get_innovation_number();
            int32_t node_type = g->nodes[i]->node_type;
            if (i==g->nodes.size()-1)
                node_inno*=-1;


            colony[node_inno]->type_pheromones[node_types[node_type]]+=treat_pheromone;
            if (colony[node_inno]->type_pheromones[node_types[node_type]] > PHEROMONES_THERESHOLD )
                colony[node_inno]->type_pheromones[node_types[node_type]] = PHEROMONES_THERESHOLD ;
            if (colony[node_inno]->type_pheromones[node_types[node_type]] < 0 )
                colony[node_inno]->type_pheromones[node_types[node_type]] = MINIMUM_PHEROMONE_VALUE ;
        }


        if ( g->nodes[i]->get_depth() == 0.0 ) {
            for ( int j=0; j<colony[-1]->pheromone_lines->size(); j++) {
                if ( g->nodes[i]->get_innovation_number() == colony[-1]->pheromone_lines->at(j)->get_output_innovation_number() ) {
                    colony[-1]->pheromone_lines->at(j)->edge_pheromone+=treat_pheromone;
                    if ( colony[-1]->pheromone_lines->at(j)->edge_pheromone > PHEROMONES_THERESHOLD )
                        colony[-1]->pheromone_lines->at(j)->edge_pheromone = PHEROMONES_THERESHOLD ;
                    if ( colony[-1]->pheromone_lines->at(j)->edge_pheromone < 0 )
                        colony[-1]->pheromone_lines->at(j)->edge_pheromone = MINIMUM_PHEROMONE_VALUE ;
                }
            }
        }
    }


    for ( int i=0; i<g->edges.size(); i++){
        if (g->edges[i]->enabled){
            int32_t edge_inno = g->edges[i]->get_innovation_number();


            if ( colony_edges[edge_inno]->edge_pheromone * treat_pheromone < PHEROMONES_THERESHOLD && colony_edges[edge_inno]->edge_pheromone * treat_pheromone > 0 )
                colony_edges[edge_inno]->edge_pheromone+= treat_pheromone;
            if ( colony_edges[edge_inno]->edge_pheromone > PHEROMONES_THERESHOLD )
                colony_edges[edge_inno]->edge_pheromone = PHEROMONES_THERESHOLD;
            if ( colony_edges[edge_inno]->edge_pheromone < 0 )
                colony_edges[edge_inno]->edge_pheromone = MINIMUM_PHEROMONE_VALUE ;


            /*
            for ( int j=0; j<colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    if ( colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*treat_pheromone < PHEROMONES_THERESHOLD && colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*treat_pheromone > 0){
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone+=treat_pheromone;
                        break;
                    }
                    if ( colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone > PHEROMONES_THERESHOLD )
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = PHEROMONES_THERESHOLD;
                    if ( colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone < 0 )
                        colony[g->edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = MINIMUM_PHEROMONE_VALUE ;
                    break;
                }
            }
            */
        }
    }

    for ( int i=0; i<g->recurrent_edges.size(); i++){
        if (g->recurrent_edges[i]->enabled){
            int32_t edge_inno = g->recurrent_edges[i]->get_innovation_number();


            if ( colony_edges[edge_inno]->edge_pheromone * treat_pheromone < PHEROMONES_THERESHOLD && colony_edges[edge_inno]->edge_pheromone * treat_pheromone > 0 )
                colony_edges[edge_inno]->edge_pheromone+= treat_pheromone;
            if ( colony_edges[edge_inno]->edge_pheromone > PHEROMONES_THERESHOLD )
                colony_edges[edge_inno]->edge_pheromone = PHEROMONES_THERESHOLD;
            if ( colony_edges[edge_inno]->edge_pheromone < 0 )
                colony_edges[edge_inno]->edge_pheromone = MINIMUM_PHEROMONE_VALUE ;

            /*
            for ( int j=0; j<colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->size(); j++){
                if ( edge_inno == colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_innovation_number ){
                    if ( colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*treat_pheromone < PHEROMONES_THERESHOLD && colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone*treat_pheromone > 0){
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone+=treat_pheromone;
                        break;
                    }
                    if ( colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone > PHEROMONES_THERESHOLD )
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = PHEROMONES_THERESHOLD;
                    if ( colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone < 0 )
                        colony[g->recurrent_edges[i]->get_input_innovation_number()]->pheromone_lines->at(j)->edge_pheromone = MINIMUM_PHEROMONE_VALUE ;
                    break;
                }
            }
            */
        }
    }
}

/*Will reduce phermones periodically*/
void ACNNTO::evaporate_pheromones(){
    for (auto const& c : colony){
        if (c.first>-1) {
            c.second->type_pheromones[0]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
            c.second->type_pheromones[1]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
            c.second->type_pheromones[2]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
            c.second->type_pheromones[3]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
            c.second->type_pheromones[4]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
        }
        for ( int j=0; j<c.second->pheromone_lines->size(); j++){
            c.second->pheromone_lines->at(j)->edge_pheromone*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
        }
    }
}


RNN_Node* ACNNTO::check_node_existance(vector<RNN_Node_Interface*> &rnn_nodes,   EDGE_Pheromone* pheromone_line){
    for ( int i=0; i<rnn_nodes.size(); i++){
        if (rnn_nodes[i]->get_innovation_number() == abs(pheromone_line->get_output_innovation_number())){
            return (RNN_Node*)rnn_nodes[i];
        }
    }

    int innovation_number = abs(pheromone_line->get_output_innovation_number());
    int layer_type        = colony[pheromone_line->get_output_innovation_number()]->get_layer_type();
    int current_layer     = colony[pheromone_line->get_output_innovation_number()]->get_current_layer();
    int node_type         = old_pick_node_type(colony[pheromone_line->get_output_innovation_number()]->type_pheromones);

    if (node_type == LSTM_NODE) {
        rnn_nodes.push_back( new LSTM_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == DELTA_NODE) {
        rnn_nodes.push_back( new Delta_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == GRU_NODE) {
        rnn_nodes.push_back( new GRU_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == MGU_NODE) {
        rnn_nodes.push_back( new MGU_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == UGRNN_NODE) {
        rnn_nodes.push_back( new UGRNN_Node(innovation_number, layer_type, current_layer) );
    } else if (node_type == FEED_FORWARD_NODE ) {
        rnn_nodes.push_back( new RNN_Node(innovation_number, layer_type, current_layer, node_type) );
    } else {
        cerr << "ACNNTO:: Error reading node from stream, unknown node_type: " << node_type << endl;
        exit(1);
    }

    return (RNN_Node*)rnn_nodes.back();
}

void ACNNTO::check_edge_existance(vector<RNN_Edge*> &rnn_edges, int32_t innovation_number, RNN_Node* in_node, RNN_Node* out_node, double edge_weight){
    for ( int i=0; i<rnn_edges.size(); i++){
        if (rnn_edges[i]->get_innovation_number() == innovation_number){
            return;
        }
    }
    rnn_edges.push_back(new RNN_Edge(innovation_number, in_node, out_node));
    edges_inherited_weights.push_back( edge_weight ) ;
    return;
}

void ACNNTO::check_recurrent_edge_existance(vector<RNN_Recurrent_Edge*> &recurrent_edges, int32_t innovation_number, int32_t depth, RNN_Node* in_node, RNN_Node* out_node, double edge_weight){
    for ( int i=0; i<recurrent_edges.size(); i++){
        if (recurrent_edges[i]->get_innovation_number() == innovation_number){
            return;
        }
    }
    recurrent_edges.push_back(new RNN_Recurrent_Edge(innovation_number, depth, in_node, out_node));
    recedges_inherited_weights.push_back( edge_weight ) ;
    return;
}

void ACNNTO::initialize_genome_parameters(RNN_Genome* genome ) {
    int island  = 1;
    genome->set_island(island);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    genome->set_normalize_bounds(normalize_mins, normalize_maxs);

    edge_innovation_count = genome->edges.size() + genome->recurrent_edges.size();
    node_innovation_count = genome->nodes.size();

    genome->set_generated_by("initial");

    genome->set_bp_iterations(bp_iterations);
    genome->set_learning_rate(learning_rate);

    if (use_high_threshold) genome->enable_high_threshold(high_threshold);
    if (use_low_threshold) genome->enable_low_threshold(low_threshold);


    if ( use_edge_inherited_weights || use_pheromone_weight_update ) {
        int nodes_total_number_weights = 0 ;                        //<<<<<<<<================================================
        for ( int i=0; i<genome->nodes.size(); i++){
            nodes_total_number_weights+= genome->nodes[i]->get_number_weights() ;
        }

        vector <double> parameters ;
        parameters.assign( nodes_total_number_weights + edges_inherited_weights.size() + recedges_inherited_weights.size(), 0.0 );

        uniform_real_distribution<double> rng(-0.5, 0.5);
        int count = 0 ;
        while ( count<parameters.size() ) {
            for (uint32_t i = 0; i < nodes_total_number_weights; i++) {
                parameters[count++] = rng(generator);
            }
            for (uint32_t i = 0; i < edges_inherited_weights.size(); i++) {
                // cout << "INITIALIZING:: Edges Weight " << edges_inherited_weights[i] << endl;
                parameters[count++] = edges_inherited_weights[i];
            }
            for (uint32_t i = 0; i < recedges_inherited_weights.size(); i++) {
                // cout << "INITIALIZING:: RecEdges Weight " << recedges_inherited_weights[i] << endl;
                parameters[count++] = recedges_inherited_weights[i];
            }
        }
        genome->set_initial_parameter( parameters ) ;
    }
    else {
        genome->initialize_randomly() ;
    }                                                                //<<<<<<<<================================================

    // double _mu, _sigma;
    // cout << "getting mu/sigma after random initialization!" << endl;
    // genome->get_mu_sigma(genome->best_parameters, _mu, _sigma);

    genome->best_validation_mse = EXALT_MAX_DOUBLE;
    genome->best_validation_mae = EXALT_MAX_DOUBLE;
    genome->best_parameters.clear();
    edges_inherited_weights.clear() ;
    recedges_inherited_weights.clear() ;
    cout << "FISNHSIED WEIGHTS INTIALIZATION" << endl ;
}

void ACNNTO::pheromones_update_weights( RNN_Genome* genome ) {
    double min_pheromone = 99999999.0 ;
    double max_pheromone = -1.0 ;
    for ( auto const& c_e : colony_edges ) {
        if ( c_e.second->edge_pheromone > max_pheromone ) {
            max_pheromone = c_e.second->edge_pheromone ;
        }
        if ( c_e.second->edge_pheromone < min_pheromone ) {
            min_pheromone = c_e.second->edge_pheromone ;
        }
    }
    cout << "PHEROMONE_UPDATE WEIGHTS:: Max Pheromone: " << max_pheromone << endl ;
    double max_minus_min = max_pheromone - min_pheromone ;

    for ( int e=0; e<genome->edges.size(); e++ ) {
        int edge_inno = genome->edges[e]->get_innovation_number() ;
        // cout << "PHEROMONE_UPDATE WEIGHTS:: Edge Pheromone: " << colony_edges[edge_inno]->edge_pheromone << endl ;
        // cout << "PHEROMONE_UPDATE WEIGHTS:: Edge Weight BEFORE: " << colony_edges[edge_inno]->edge_weight << endl ;
        colony_edges[edge_inno]->edge_weight*= ( colony_edges[edge_inno]->edge_pheromone / max_pheromone ) ;
        // cout << "PHEROMONE_UPDATE WEIGHTS:: Edge Weight AFTER: " << colony_edges[edge_inno]->edge_weight << endl ;
        // cout << "_________________________________\n" ;
    }

    for ( int e=0; e<genome->recurrent_edges.size(); e++ ) {
        int edge_inno = genome->recurrent_edges[e]->get_innovation_number() ;
        // cout << "PHEROMONE_UPDATE WEIGHTS:: Edge Pheromone: " << colony_edges[edge_inno]->edge_pheromone << endl ;
        // cout << "PHEROMONE_UPDATE WEIGHTS:: Edge Weight BEFORE: " << colony_edges[edge_inno]->edge_weight << endl ;
        colony_edges[edge_inno]->edge_weight*= ( colony_edges[edge_inno]->edge_pheromone / max_pheromone ) ;
        // cout << "PHEROMONE_UPDATE WEIGHTS:: Edge Weight AFTER: " << colony_edges[edge_inno]->edge_weight << endl ;
        // cout << "_________________________________\n" ;
    }

    // for ( auto const& e : colony_edges ) {
    //     // e.second->edge_weight*= ( ( e.second->edge_pheromone - min_pheromone ) / ( max_minus_min ) ) ;
    //     e.second->edge_weight*= ( e.second->edge_pheromone / max_pheromone ) ;
    // }
    cout << "FINISHED PHEROMONE UPDATING WEIGHTS\n" ;
}

/* Each ant will marsh and choose a path and will turn each edge/node active in its path */
RNN_Genome* ACNNTO::ants_march(){
    if (generated_genomes == max_genomes){
        return NULL;
    }

    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;
    EDGE_Pheromone* pheromone_line = NULL;
    while (1){
        rnn_nodes.clear();
        rnn_edges.clear();
        recurrent_edges.clear();

        if (generated_genomes!=0 && generated_genomes%1==0){
            evaporate_pheromones();
        }
        for (int ant=0; ant<ants; ant++){
            cout<<"ANT:: "<<ant<<endl;
            pheromone_line = pick_line(colony[-1]->pheromone_lines); //begine with imaginary node
            RNN_Node* current_node = check_node_existance(rnn_nodes,  pheromone_line);
            RNN_Node* previous_node = current_node;

            while (1){
                // cout << "\tNode ID From Marching: " << pheromone_line->get_output_innovation_number() << endl;
                pheromone_line = pick_line(colony[pheromone_line->get_output_innovation_number()]->pheromone_lines);
                // cout << "\tNode Edge ID From Marching: " << pheromone_line->get_edge_innovation_number() << " Edge Depth: " << pheromone_line->get_depth() << endl;
                current_node = check_node_existance(rnn_nodes, pheromone_line);
                if (pheromone_line->get_depth()==0 && previous_node->get_innovation_number()<current_node->get_innovation_number()){
                    check_edge_existance(rnn_edges, pheromone_line->get_edge_innovation_number(), previous_node, current_node, pheromone_line->get_weight() );
                }
                else{
                    check_recurrent_edge_existance(recurrent_edges, pheromone_line->get_edge_innovation_number(), pheromone_line->get_depth(), previous_node, current_node, pheromone_line->get_weight() );
                }
                previous_node = current_node;
                if (pheromone_line->get_output_innovation_number()<0)
                    break;  //break while() when its iterator reaches an output node
            }
        }


        for (int i=0; i<this->number_inputs; i++){
            bool add_node = true;
            for (int j=0; j<rnn_nodes.size(); j++){
                if (rnn_nodes[j]->layer_type==INPUT_LAYER && rnn_nodes[j]->innovation_number==i){
                    add_node = false;
                    break;
                }
            }
            if (add_node){
                RNN_Node* node = new RNN_Node(-(i+100), INPUT_LAYER, 0, FEED_FORWARD_NODE);
                node->enabled = false;
                rnn_nodes.push_back(node);
            }
        }

        /** Print nodes and their memory addresses
        // for (int y=0; y<rnn_nodes.size(); y++)
        // cout<<"\t\tNODES["<<y<<"]: "<<rnn_nodes[y]->get_innovation_number()<<" Layer Type: "<<rnn_nodes[y]->layer_type<< " Node Address: "<< rnn_nodes[y] << " Node Type: " << rnn_nodes[y]->node_type <<endl;

        /** Printing Edges and their memory addresses
        // for (int y=0; y<rnn_edges.size(); y++)
        // cout<<"\t\tEDGE["<<y<<"] (innov: "<<rnn_edges[y]->innovation_number<<" Add: "<< rnn_edges[y] << ")  From NODE: "<<rnn_edges[y]->get_input_innovation_number()<< " (" << rnn_edges[y]->input_node << ") " <<" -- To NODE: "<<rnn_edges[y]->get_output_innovation_number()<< " (" << rnn_edges[y]->output_node << ") "<<endl;

        /** Printing Edges and their memory addresses
        // for (int y=0; y<recurrent_edges.size(); y++)
        // cout<<"\t\tRecEDGE["<<y<<"] (inov: "<<recurrent_edges[y]->innovation_number<<" Add: " << recurrent_edges[y] <<  ")  From NODE: "<<recurrent_edges[y]->get_input_innovation_number() << " (" << recurrent_edges[y]->output_node << ") "<< " -- To NODE: "<<recurrent_edges[y]->get_output_innovation_number() << " (" << recurrent_edges[y]->output_node << ") "<< " Depth: "<<recurrent_edges[y]->recurrent_depth<<endl;

        // cout<<"NUMBER of Nodes: "<<rnn_nodes.size()<<endl;
        **/

        RNN_Genome* g = NULL;
        g = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges);
        if (!g->outputs_unreachable()){
            g->set_generation_id(generated_genomes++);
            initialize_genome_parameters( g );

            cout << "ANTS SUCCEEDED IN FINDING A COMPLETE NN STRUCTURE.... WILL BEGIN GENOME EVALUATION..."<<endl;
            cout << "GENOME No.: " << generated_genomes << " STARTING!" << endl;
            write_to_file (output_directory + "/colony_" + to_string(generated_genomes) + ".bin", false);
            g->write_to_file(output_directory + "/gene_" + to_string(generated_genomes) + ".bin", false);
            return g;
            exit(0);
        }
        cout<<"ANTS FAILED TO FIND A COMPLETE NN STRUCTURE.... BEGINING ANOTHER ITERATION..."<<endl;
    }
}


void ACNNTO::write_to_file(string bin_filename, bool verbose) {
    ofstream bin_outfile(bin_filename, ios::out | ios::binary);
    write_to_stream(bin_outfile, verbose);
    bin_outfile.close();
}


void ACNNTO::write_to_stream(ostream &bin_ostream, bool verbose) {
    if (verbose) cout << "WRITING COLONY TO STREAM" << endl;

    bin_ostream.write((char*)&generated_genomes, sizeof(int32_t));
    bin_ostream.write((char*)&max_recurrent_depth, sizeof(int32_t));

    int number_of_nodes = colony.size();
    bin_ostream.write((char*)&number_of_nodes, sizeof(int));
    int32_t m = 0;
    double k = 0.0;
    for (auto const& c : colony){
        m = c.first;
        bin_ostream.write((char*)&m, sizeof(int32_t));


        //saving number of node types that will be printed to the file
        // from each node in the colony
        int number_of_node_types = node_types.size();
        bin_ostream.write((char*)&number_of_node_types, sizeof(int));

        if (c.first!=-1){
            for (int n=0; n<node_types.size(); n++){
                k = c.second->type_pheromones[n];
                // cout << "Node Pheromone: " << k << endl;
                bin_ostream.write((char*)&k, sizeof(double));
            }
        }

        //print number of pheromone lines in coming out from each
        // node in the colony

        int number_of_pheromone_lines = c.second->pheromone_lines->size();
        bin_ostream.write((char*)&number_of_pheromone_lines, sizeof(int));

        for ( int e=0; e<c.second->pheromone_lines->size(); e++){
            int32_t innovation_number        = c.second->pheromone_lines->at(e)->get_edge_innovation_number();
            double  edge_phermone            = c.second->pheromone_lines->at(e)->get_edge_phermone();
            int depth                        = c.second->pheromone_lines->at(e)->get_depth();
            int32_t input_innovation_number  = c.second->pheromone_lines->at(e)->get_input_innovation_number();
            int32_t output_innovation_number = c.second->pheromone_lines->at(e)->get_output_innovation_number();

            // cout << "Edge Pheromone: " << edge_phermone << endl;

            bin_ostream.write((char*)&innovation_number         , sizeof(int32_t));
            bin_ostream.write((char*)&edge_phermone             , sizeof(double));
            bin_ostream.write((char*)&depth                     , sizeof(int));
            bin_ostream.write((char*)&input_innovation_number   , sizeof(int32_t));
            bin_ostream.write((char*)&output_innovation_number  , sizeof(int32_t));
        }
    }
}
