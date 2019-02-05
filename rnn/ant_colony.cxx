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

#include map;
using std::map

#include <string>
using std::string;
using std::to_string;

//for mkdir
#include <sys/stat.h>

#include "ant_colony.hxx"
#include "node_pheromone.hxx"
#include "rnn_genome.hxx"
#include "rnn_colony.hxx"
#include "generate_nn.hxx"

ANT_COLONY::~ANT_COLONY() {
    RNN_Genome *genome;
    for (int32_t i = 0; i < population.size(); i++) {
        while (population[i].size() > 0) {
            genome = population[i].back();
            population[i].pop_back();
            delete genome;
        }
    }
}



ANT_COLONY::ANT_COLONY(int32_t _population_size, int32_t _max_genomes, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs, int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold, double _low_threshold, string _output_directory) : population_size(_population_size), max_genomes(_max_genomes), number_inputs(_input_parameter_names.size()), number_outputs(_output_parameter_names.size()), bp_iterations(_bp_iterations), learning_rate(_learning_rate), use_high_threshold(_use_high_threshold), high_threshold(_high_threshold), use_low_threshold(_use_low_threshold), low_threshold(_low_threshold), output_directory(_output_directory) {

    input_parameter_names   = _input_parameter_names;
    output_parameter_names  = _output_parameter_names;
    normalize_mins          = _normalize_mins;
    normalize_maxs          = _normalize_maxs;

    inserted_genomes  = 0;
    generated_genomes = 0;
    total_bp_epochs   = 0;

    edge_innovation_count = 0;
    node_innovation_count = 0;

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);

    max_recurrent_depth = 10;

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


    if (output_directory != "") {
        mkdir(output_directory.c_str(), 0777);
        log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
        (*log_file) << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges" << endl;
        memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges" << endl;
    } else {
        log_file = NULL;
    }

    startClock = std::chrono::system_clock::now();
}

void ANT_COLONY::print_population() {
    cout << "POPULATIONS: " << endl;
    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        cout << "\tPOPULATION " << i << ":" << endl;

        cout << "\t" << RNN_Genome::print_statistics_header() << endl;

        for (int32_t j = 0; j < (int32_t)population[i].size(); j++) {
            cout << "\t" << population[i][j]->print_statistics() << endl;
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
                cerr << "ERROR, could not open ANT_COLONY output log: '" << output_file << "'" << endl;
                exit(1);
            }
        }

        RNN_Genome *best_genome = get_best_genome();

        std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

        (*log_file) << inserted_population
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count() << endl;

        memory_log << inserted_genomes
            << "," << total_bp_epochs
            << "," << milliseconds
            << "," << best_genome->best_validation_mae
            << "," << best_genome->best_validation_mse
            << "," << best_genome->get_enabled_node_count()
            << "," << best_genome->get_enabled_edge_count()
            << "," << best_genome->get_enabled_recurrent_edge_count() << endl;
    }
}

void ANT_COLONY::write_memory_log(string filename) {
    ofstream log_file(filename);
    log_file << memory_log.str();
    log_file.close();
}

void ANT_COLONY::set_possible_node_types(vector<string> possible_node_type_strings) {
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


bool ANT_COLONY::populations_full() const {
    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        if (population[i].size() < population_size) return false;
    }
    return true;
}

string ANT_COLONY::get_output_directory() const {
    return output_directory;
}

RNN_Genome* ANT_COLONY::get_best_genome() {
    int32_t best_genome_population = -1;
    double best_fitness = EXALT_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        if (population[i].size() > 0) {
            if (population[i][0]->get_fitness() <= best_fitness) {
                best_fitness = population[i][0]->get_fitness();
                best_genome_population = i;
            }
        }
    }

    if (best_genome_population < 0) {
        return NULL;
    } else {
        return population[best_genome_population][0];
    }
}

RNN_Genome* ANT_COLONY::get_worst_genome() {
    int32_t worst_genome_population = -1;
    double worst_fitness = -EXALT_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)population.size(); i++) {
        if (population[i].size() > 0) {
            if (population[i].back()->get_fitness() > worst_fitness) {
                worst_fitness = population[i].back()->get_fitness();
                worst_genome_population = i;
            }
        }
    }

    if (worst_genome_population < 0) {
        return NULL;
    } else {
        return population[worst_genome_population].back();
    }
}


double ANT_COLONY::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXALT_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double ANT_COLONY::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXALT_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

//this will insert a COPY, original needs to be deleted
bool ANT_COLONY::insert_genome(RNN_Genome* genome) {
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

    if (new_fitness > population.back()->get_fitness()) {
        cout << "ignoring genome, fitness: " << new_fitness << " > worst population" << " fitness: " << population.back()->get_fitness() << endl;
        print_population();
        reward_colony(genome, 0.85);
        return false;
    }
    reward_colony(genome, 1.15);
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

        } else {
            cerr << "\tpopulation already contains genome! not inserting." << endl;
            print_population();
            return false;
        }
    }

    if (population.back()->get_fitness() > new_fitness) {
        //this genome will be inserted
        was_inserted = true;

        if (genome->get_fitness() < get_best_genome()->get_fitness()) {
            cout << "new best fitness!" << endl;

            if (genome->get_fitness() != EXALT_MAX_DOUBLE) {
                //need to set the weights for non-initial genomes so we
                //can generate a proper graphviz file
                vector<double> best_parameters = genome->get_best_parameters();
                genome->set_weights(best_parameters);
            }

            genome->write_graphviz(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".gv");
            genome->write_to_file(output_directory + "/rnn_genome_" + to_string(inserted_genomes) + ".bin", true);

        }

        genome->update_generation_map(inserted_from_map);

        cout << "inserting new genome to population " << endl;
        //inorder insert the new individual
        RNN_Genome *copy = genome->copy();
        // cout << "created copy with island: " << copy->get_island() << endl;

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

    return was_inserted;
}

void ANT_COLONY::initialize_genome_parameters(RNN_Genome* genome) {
    genome->set_bp_iterations(bp_iterations);
    genome->set_learning_rate(learning_rate);

    if (use_high_threshold) genome->enable_high_threshold(high_threshold);
    if (use_low_threshold) genome->enable_low_threshold(low_threshold);
}

ANT_COLONY::generate_genome(int number_hidden_layers, int number_hidden_nodes, int max_recurrent_depth) {

    create_ff_w_pheromones(number_inputs, 0, 0, number_outputs, max_recurrent_depth, genome, &colony);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    genome->set_normalize_bounds(normalize_mins, normalize_maxs);

    edge_innovation_count = genome->edges.size() + genome->recurrent_edges.size();
    node_innovation_count = genome->nodes.size();

    genome->set_generated_by("initial");
    initialize_genome_parameters(genome);

    //insert a copy of it into the population so
    //additional requests can mutate it
    genome->initialize_randomly();
    bool initialize_phermones_to_ones = true;
    colony->initialize_randomly(initialize_phermones_to_ones);

    genome->best_validation_mse = EXALT_MAX_DOUBLE;
    genome->best_validation_mae = EXALT_MAX_DOUBLE;
    genome->best_parameters.clear();
    //genome->clear_generated_by();

    // genome->set_generation_id(generated_genomes);
}

EDGE_Pheromone ANT_COLONY::pick_line(vector<EDGE_Pheromone> edges_pheromones){
  uniform_real_distribution<double> rng(0.0, 1.0);
  double rand_gen = rng(generator) * sum_pheromones;
  int a;
  for ( a = 0; a < edges_pheromones.size(); a++) {
      if ( rand_gen<=edges_pheromones[a].get_edge_phermone() )
        break;
      else:
        rand_gen-= pheromones_basket[a]
  }
  return edges_pheromones[a];
}

int ANT_COLONY::pick_node_type(double* type_pheromones){
  uniform_real_distribution<double> rng(0.0, 1.0);
  double rand_gen = rng(generator) * sum_pheromones;
  int a;
  for ( a = 0; a < (sizeof(type_pheromones)/sizeof(type_pheromones[0])); a++) {
      if ( rand_gen<=type_pheromones[a].get_edge_phermone() )
        break;
      else:
        rand_gen-= type_pheromones[a]
  }
  switch (a) {
    case 0: return FEED_FORWARD_NODE;
      break;
    case 1: return DELTA_NODE;
      break;
    case 2: return GRU_NODE;
      break;
    case 3: return MGU_NODE;
      break;
    case 4: return UGRNN_NODE;
      break;
  }
}

/*If genome fitness is good compared to the population=>
      reward it and increase the pheromones on its active nodes and edges
  If genome fitness is bad=>
      penalize it and reduce the pheromones on its active nodes and edges */
void ANT_COLONY::reward_colony(Genome* g, double treat){
  for ( int i=0; i<g->nodes->size(); i++){
    if (treat_pheromone==1.15){
      if (g->nodes[i]->node_type==FEED_FORWARD_NODE) {
        if (colony[g->nodes[i].get_innovation_number()].type_pheromones[0]*treat_pheromone < PHEROMONES_THERESHOLD) //Check if the increase will be more the max threshold
          colony[g->nodes[i].get_innovation_number()].type_pheromones[0]*=treat_pheromone;    //if No=> increase it
        else
          colony[g->nodes[i].get_innovation_number()].type_pheromones[0]=PHEROMONES_THERESHOLD; //if Yes=> equate it with the threshold
      }
      if (g->nodes[i]->node_type==DELTA_NODE) {
        if (colony[g->nodes[i].get_innovation_number()].type_pheromones[1]*treat_pheromone < PHEROMONES_THERESHOLD)
          colony[g->nodes[i].get_innovation_number()].type_pheromones[1]*=treat_pheromone;
          else
            colony[g->nodes[i].get_innovation_number()].type_pheromones[1]=PHEROMONES_THERESHOLD;
      }
      if (g->nodes[i]->node_type==GRU_NODE) {
        if (colony[g->nodes[i].get_innovation_number()].type_pheromones[2]*treat_pheromone < PHEROMONES_THERESHOLD)
          colony[g->nodes[i].get_innovation_number()].type_pheromones[2]*=treat_pheromone;
        else
          colony[g->nodes[i].get_innovation_number()].type_pheromones[2]=PHEROMONES_THERESHOLD;
      }
      if (g->nodes[i]->node_type==MGU_NODE) {
        if (colony[g->nodes[i].get_innovation_number()].type_pheromones[3]*treat_pheromone < PHEROMONES_THERESHOLD)
          colony[g->nodes[i].get_innovation_number()].type_pheromones[3]*=treat_pheromone;
        else
          colony[g->nodes[i].get_innovation_number()].type_pheromones[3]=PHEROMONES_THERESHOLD;
      }
      if (g->nodes[i]->node_type==UGRNN_NODE) {
        if (colony[g->nodes[i].get_innovation_number()].type_pheromones[4]*treat_pheromone < PHEROMONES_THERESHOLD)
          colony[g->nodes[i].get_innovation_number()].type_pheromones[4]*=treat_pheromone;
        else
          colony[g->nodes[i].get_innovation_number()].type_pheromones[4]=PHEROMONES_THERESHOLD;
      }
    }
  }

  for ( int i=0; i<g->edges->size(); i++){
    if (g->edges[i]->enabled){
      if (colony[g->edges[i].get_input_innovation_number()].pheromone_lines[g->edges[i].get_innovation_number()].edge_pheromone*treat_pheromone < PHEROMONES_THERESHOLD)
        colony[g->edges[i].get_input_innovation_number()].pheromone_lines[g->edges[i].get_innovation_number()].edge_pheromone*=treat_pheromone;
      else
        colony[g->edges[i].get_input_innovation_number()].pheromone_lines[g->edges[i].get_innovation_number()].edge_pheromone = PHEROMONES_THERESHOLD;
    }
  }

}

/*Will reduce phermones periodically*/
void ANT_COLONY::reward_colony(){
  for ( int i=0; i<colony.size(); i++){
    colony[i].type_pheromones[0]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
    colony[i].type_pheromones[1]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
    colony[i].type_pheromones[2]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
    colony[i].type_pheromones[3]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
    colony[i].type_pheromones[4]*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
    for ( int j=0; j<colony[i][1].size(); j++){
          colony[i].pheromone_lines[j].edge_pheromone*=PERIODIC_PHEROMONE_REDUCTION_RATIO;
    }
  }
}


/* Each ant will marsh and choose a path and will turn each edge/node active in its path */
ANT_COLONY::ants_march(){
    for (int ant=0; ant<ANT; ant++){
      RNN_Genome g = genome.copy();
      g->set_generation_id(generated_genomes++);
      if (genome%100=0){
        reduce_pheromones();
      }
      EDGE_Pheromone phermone_line = pick_line(colony[-1].pheromone_lines); //begine with imaginary node
      for (int i=0; i<g.get_edge_count(); i++){
        if ( g->edges[i].get_innovation_number()==phermone_line.get_edge_innovation_number() ){
          g->edges[i]->enabled = true;
        }
      }
      for ( int i=0; i<g.get_node_count(); i++){                 //fix the edges of the selected input node
        if ( g->nodes[i].get_innovation_number()==phermone_line.get_output_innovation_number() ){
          g->nodes[i]->enabled = true;
          g->nodes[i]->node_type = pick_node_type(colony[phermone_line.get_out_innovation_number()].type_pheromones );
        }
      }
      while (1){                                               // work on the following node and their edges
        phermone_line = pick_line(colony[phermone_line.get_output_innovation_number()].pheromone_lines);
        for (int i=0; i<g.get_edge_count(); i++){
          if ( g->edges[i].get_innovation_number()==phermone_line.get_edge_innovation_number() ){
            g->edges[i]->enabled = true;
          }
        }
        for ( int i=0; i<g.get_node_count(); i++){
          if ( g->nodes[i].get_innovation_number()==phermone_line.get_output_innovation_number() ){
            g->nodes[i]->enabled = true;
            g->nodes[i]->node_type = pick_node_type(colony[phermone_line.get_out_innovation_number()].type_pheromones );
          }
        }

        if (phermone_line.get_out_innovation_number()==-1)
          break;  //break when reaches an output node
      }
    }
}
