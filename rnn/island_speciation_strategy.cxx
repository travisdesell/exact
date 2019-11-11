#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <string>
using std::string;

#include "examm.hxx"
#include "rnn_genome.hxx"
#include "island_speciation_strategy.hxx"


RNN_Genome* IslandSpeciationStrategy::get_best_genome() {
    int32_t best_genome_island = -1;
    double best_fitness = EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i].size() > 0) {
            if (islands[i][0]->get_fitness() <= best_fitness) {
                best_fitness = islands[i][0]->get_fitness();
                best_genome_island = i;
            }
        }
    }

    if (best_genome_island < 0) {
        return NULL;
    } else {
        return islands[best_genome_island][0];
    }
}

RNN_Genome* IslandSpeciationStrategy::get_worst_genome() {
    int32_t worst_genome_island = -1;
    double worst_fitness = -EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i].size() > 0) {
            if (islands[i].back()->get_fitness() > worst_fitness) {
                worst_fitness = islands[i].back()->get_fitness();
                worst_genome_island = i;
            }
        }
    }

    if (worst_genome_island < 0) {
        return NULL;
    } else {
        return islands[worst_genome_island].back();
    }
}


double IslandSpeciationStrategy::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double IslandSpeciationStrategy::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

bool IslandSpeciationStrategy::populations_full() const {
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (genomes[i].size() < population_size) return false;
    }

    return true;
}

int32_t IslandSpeciationStrategy::check_on_island() {
    cout << "check_on_island with method '" << speciation_method << "'" << endl;

    if (speciation_method.compare("") == 0) {
        return -1;
    }

    cout << "num_genomes_check_on_island: " << num_genomes_check_on_island << ", inserted genomes: " << inserted_genomes << endl;
    // only run this function every n inserted genomes
    if (num_genomes_check_on_island == 0 || inserted_genomes % num_genomes_check_on_island != 0) {
        cout << "not trying to kill island" << endl;
        return -1;
    }

    if (speciation_method.compare("clear_island_with_worst_best_genome") == 0) {
        cout << "runnning clear island with worst best genome!" << endl;

        return clear_island_with_worst_best_genome();
    }
    
    return -1;
}

int32_t IslandSpeciationStrategy::clear_island_with_worst_best_genome() {
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


//this will insert a COPY, original needs to be deleted
int32_t IslandSpeciationStrategy::insert_genome(RNN_Genome* genome) {
    int32_t island = genome->get_group_id();

    bool was_inserted = islands[island]->insert_genome();

    return was_inserted;
}


RNN_Genome* IslandSpeciationStrategy::generate_genome() {
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

                //TODO: make max_mutations an IslandSpeciationStrategy option
                mutate(1 /*max_mutations*/, genome);

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

                //TODO: make max_mutations an IslandSpeciationStrategy option
                mutate(1 /*max_mutations*/, genome);

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

        while (genome == NULL) {
            int32_t parent_island1;
            do {
                parent_island1 = genomes.size() * rng_0_1(generator);
            } while (island_states[parent_island1] != ISLAND_FILLED);

            int32_t parent_island2;
            do {
                parent_island2 = genomes.size() * rng_0_1(generator);
            } while (   parent_island1 == parent_island2 
                     || island_states[parent_island2] != ISLAND_FILLED);

            RNN_Genome* genome1 = genomes[parent_island1][genomes[parent_island1].size() * rng_0_1(generator)];
            RNN_Genome* genome2 = genomes[parent_island2][genomes[parent_island2].size() * rng_0_1(generator)];

            genome = crossover(genome1, genome2);
            genome->set_normalize_bounds(normalize_mins, normalize_maxs);
            genome->set_island(island);

            if (genome->outputs_unreachable()) {
                //no path from at least one input to the outputs
                delete genome;
                genome = NULL;
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
 
