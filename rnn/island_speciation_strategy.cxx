#include <functional>
using std::function;

#include <chrono>

//#include <iostream>

#include <random>
#include <string>
using std::minstd_rand0;
using std::uniform_real_distribution;
using std::string;

#include "examm.hxx"
#include "rnn_genome.hxx"
#include "island_speciation_strategy.hxx"

#include "common/log.hxx"

IslandSpeciationStrategy::IslandSpeciationStrategy(int32_t _number_of_islands, int32_t _max_island_size, double _mutation_rate, double _intra_island_crossover_rate, double _inter_island_crossover_rate, RNN_Genome *seed_genome, int32_t _number_stir_mutations) : generation_island(0), number_of_islands(_number_of_islands), max_island_size(_max_island_size), mutation_rate(_mutation_rate), intra_island_crossover_rate(_intra_island_crossover_rate), inter_island_crossover_rate(_inter_island_crossover_rate), generated_genomes(0), inserted_genomes(0), minimal_genome(seed_genome), number_stir_mutations(_number_stir_mutations) {

    double rate_sum = mutation_rate + intra_island_crossover_rate + inter_island_crossover_rate;
    if (rate_sum != 1.0) {
        mutation_rate = mutation_rate / rate_sum;
        intra_island_crossover_rate = intra_island_crossover_rate / rate_sum;
        inter_island_crossover_rate = inter_island_crossover_rate / rate_sum;
    }

    intra_island_crossover_rate += mutation_rate;
    inter_island_crossover_rate += intra_island_crossover_rate;


    for (uint32_t i = 0; i < number_of_islands; i++) {
        islands.push_back(new Island(i, max_island_size));
    }

    //set the generation id for the initial minimal genome
    minimal_genome->set_generation_id(generated_genomes);
    generated_genomes++;
}

int32_t IslandSpeciationStrategy::get_generated_genomes() const {
    return generated_genomes;
}

int32_t IslandSpeciationStrategy::get_inserted_genomes() const {
    return inserted_genomes;
}

RNN_Genome* IslandSpeciationStrategy::get_best_genome() {
    int32_t best_genome_island = -1;
    double best_fitness = EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i]->size() > 0) {
            double island_best_fitness = islands[i]->get_best_fitness();
            if (island_best_fitness <= best_fitness) {
                best_fitness = island_best_fitness;
                best_genome_island = i;
            }
        }
    }

    if (best_genome_island < 0) {
        return NULL;
    } else {
        return islands[best_genome_island]->get_best_genome();
    }
}

RNN_Genome* IslandSpeciationStrategy::get_worst_genome() {
    int32_t worst_genome_island = -1;
    double worst_fitness = -EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i]->size() > 0) {
            double island_worst_fitness = islands[i]->get_worst_fitness();
            if (island_worst_fitness > worst_fitness) {
                worst_fitness = island_worst_fitness;
                worst_genome_island = i;
            }
        }
    }

    if (worst_genome_island < 0) {
        return NULL;
    } else {
        return islands[worst_genome_island]->get_worst_genome();
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

bool IslandSpeciationStrategy::islands_full() const {
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (!islands[i]->is_full()) return false;
    }

    return true;
}


//this will insert a COPY, original needs to be deleted
//returns 0 if a new global best, < 0 if not inserted, > 0 otherwise
int32_t IslandSpeciationStrategy::insert_genome(RNN_Genome* genome) {
    Log::debug("inserting genome!\n");
    inserted_genomes++;
    int32_t island = genome->get_group_id();

    Log::info("inserting genome to island: %d\n", island);

    int32_t insert_position = islands[island]->insert_genome(genome);

    if (insert_position == 0) {
        //check and see if the inserted genome has the same fitness as the best fitness
        //of all islands
        double best_fitness = get_best_fitness();
        if (genome->get_fitness() == best_fitness) return 0;
        else return 1; //was the best for the island but not the global best
    } else {
        return insert_position; //will be -1 if not inserted, or > 0 if not the global best
    }
}

RNN_Genome* IslandSpeciationStrategy::generate_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover) {
    //generate the genome from the next island in a round
    //robin fashion.
    RNN_Genome *genome = NULL;

    Log::debug("getting island: %d\n", generation_island);
    Island *island = islands[generation_island];

    Log::info("generating new genome for island[%d], island_size: %d, max_island_size: %d, mutation_rate: %lf, intra_island_crossover_rate: %lf, inter_island_crossover_rate: %lf\n", generation_island, island->size(), max_island_size, mutation_rate, intra_island_crossover_rate, inter_island_crossover_rate);

    Log::debug("islands.size(): %d, selected island is null? %d\n", islands.size(), (island == NULL));

    if (island->is_initializing()) {
        Log::debug("island is initializing!\n");

        if (island->size() == 0) {
            Log::debug("starting with minimal genome\n");
            RNN_Genome *genome_copy = minimal_genome->copy();
            
            // Stir the seed genome if need be
            if (this->number_stir_mutations) {
                Log::debug("Stirring seed genome for island %d by applying %d mutations!\n", 
                            generation_island, this->number_stir_mutations);
                mutate(this->number_stir_mutations, genome_copy);
            }

            //set the generation id for the copy and increment generated genomes 
            genome_copy->set_generation_id(generated_genomes);
            generated_genomes++;
            genome_copy->set_group_id(generation_island);

            Log::debug("inserting genome copy!\n");
            insert_genome(genome_copy);
            Log::debug("inserted genome copy!\n");

            //return a copy of the minimal genome to be trained for each island
            genome = minimal_genome->copy();
        } else {
            Log::debug("island is not empty, mutating a random genome\n");

            while (genome == NULL) {
                island->copy_random_genome(rng_0_1, generator, &genome);

                //TODO: make max_mutations an IslandSpeciationStrategy option
                mutate(1 /*max_mutations*/, genome);

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
            copy->best_validation_mse = EXAMM_MAX_DOUBLE;
            copy->best_validation_mae = EXAMM_MAX_DOUBLE;
            copy->best_parameters.clear();
            
            //set the generation id for the copy and increment generated genomes 
            copy->set_generation_id(generated_genomes);
            generated_genomes++;
            copy->set_group_id(generation_island);

            insert_genome(copy);

            //also randomly initialize this genome as
            //what it was generated from was also randomly
            //initialized as the population hasn't been
            //filled
            genome->initialize_randomly();
        }

    } else if (island->is_full()) {
        //generate a genome via crossover or mutation
        Log::info("island is full\n");

        while (genome == NULL) {
            //if we haven't filled ALL of the island populations yet, only use mutation
            //otherwise do mutation at %, crossover at %, and island crossover at %

            double r = rng_0_1(generator);
            if (!islands_full() || r < mutation_rate) {
                Log::info("performing mutation\n");

                island->copy_random_genome(rng_0_1, generator, &genome);

                //TODO: make max_mutations an IslandSpeciationStrategy option
                mutate(1 /*max_mutations*/, genome);

            } else if (r < intra_island_crossover_rate || number_of_islands == 1) {
                //intra-island crossover
                Log::info("performing intra-island crossover\n");

                //select two distinct parent genomes in the same island
                RNN_Genome *parent1 = NULL, *parent2 = NULL;
                island->copy_two_random_genomes(rng_0_1, generator, &parent1, &parent2);

                genome = crossover(parent1, parent2);
            } else {
                //inter-island crossover
                Log::info("performing inter-island crossover\n");

                //get a random genome from this island
                RNN_Genome *parent1 = NULL; 
                island->copy_random_genome(rng_0_1, generator, &parent1);

                //select a different island randomly
                int32_t other_island = rng_0_1(generator) * (number_of_islands - 1);
                if (other_island >= generation_island) other_island++;

                //get the best genome from the other island
                RNN_Genome *parent2 = islands[other_island]->get_best_genome()->copy();

                //swap so the first parent is the more fit parent
                if (parent1->get_fitness() > parent2->get_fitness()) {
                    RNN_Genome *tmp = parent1;
                    parent1 = parent2;
                    parent2 = tmp;
                }

                genome = crossover(parent1, parent2);
            }

            if (genome->outputs_unreachable()) {
                //no path from at least one input to the outputs
                delete genome;
                genome = NULL;
            }
        }

    } else if (island->is_repopulating()) {
        //here's where you put your repopulation code
        //select two other islands (non-overlapping) at random, and select genomes
        //from within those islands and generate a child via crossover

        Log::info("island is repopulating");

        while (genome == NULL) {
            //get two other islands that are not the island we'r generating for
            //note the -1 and -2 on the number_of_islands. We subtract one from both
            //so that if we randomly select this island id we increment the target
            //island. Likewise, we increment parent island 2 
            int32_t parent_island1 = (number_of_islands - 1) * rng_0_1(generator);
            int32_t parent_island2 = (number_of_islands - 2) * rng_0_1(generator);
            if (parent_island1 >= generation_island) parent_island1++;
            if (parent_island2 >= generation_island) parent_island2++;

            if (parent_island2 >= parent_island1) parent_island2++;

            RNN_Genome *parent1 = NULL;
            islands[parent_island1]->copy_random_genome(rng_0_1, generator, &parent1);
            RNN_Genome *parent2 = NULL;
            islands[parent_island2]->copy_random_genome(rng_0_1, generator, &parent2);

            //swap so the first parent is the more fit parent
            if (parent1->get_fitness() > parent2->get_fitness()) {
                RNN_Genome *tmp = parent1;
                parent1 = parent2;
                parent2 = tmp;
            }

            genome = crossover(parent1, parent2);

            if (genome->outputs_unreachable()) {
                //no path from at least one input to the outputs
                delete genome;
                genome = NULL;
            }
        }

    } else {
        Log::fatal("ERROR: island was neither initializing, repopulating or full.\n");
        Log::fatal("This should never happen!\n");
        exit(1);
    
    }

    if (genome != NULL) { 
        //set the island for the genome and increment to the next island
        genome->set_group_id(generation_island);
        generation_island++;
        if (generation_island >= islands.size()) generation_island = 0; 

        //set th generation id and increment generated genomes
        genome->set_generation_id(generated_genomes);
        generated_genomes++;
    } else {
        Log::fatal("ERROR: genome was NULL at the end of generate genome!\n");
        Log::fatal( "This should never happen.\n");
        exit(1);
    }

    return genome;
}


void IslandSpeciationStrategy::print(string indent) const {
    Log::info("%sIslands: \n", indent.c_str());
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        Log::info("%sIsland %d:\n", indent.c_str(), i);
        islands[i]->print(indent + "\t");
    }
}

/**
 * Gets speciation strategy information headers for logs
 */
string IslandSpeciationStrategy::get_strategy_information_headers() const {
    string info_header="";
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        info_header.append(",");
        info_header.append("Island_");
        info_header.append(to_string(i));
        info_header.append("_best_fitness");
        info_header.append(",");
        info_header.append("Island_");
        info_header.append(to_string(i));
        info_header.append("_worst_fitness");
    }
    return info_header;
}

/**
 * Gets speciation strategy information values for logs
 */
string IslandSpeciationStrategy::get_strategy_information_values() const {
    string info_value="";
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        double best_fitness = islands[i]->get_best_fitness();
        double worst_fitness = islands[i]->get_worst_fitness();
        info_value.append(",");
        info_value.append(to_string(best_fitness));
        info_value.append(",");
        info_value.append(to_string(worst_fitness));
    }
    return info_value;
}
 
