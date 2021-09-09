#include <functional>
using std::function;

#include <chrono>

//#include <iostream>

#include <random>

using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <stdlib.h>

#include "examm.hxx"
#include "rnn_genome.hxx"
#include "neat_speciation_strategy.hxx"

#include "common/log.hxx"

/**
 *
 */
NeatSpeciationStrategy::NeatSpeciationStrategy(
                double _mutation_rate, double _intra_island_crossover_rate,
                double _inter_island_crossover_rate, RNN_Genome *_seed_genome,
                double _species_threshold, double _fitness_threshold,
                double _neat_c1, double _neat_c2, double _neat_c3, minstd_rand0 &_generator) :
                        generation_species(0),
                        species_count(0),
                        population_not_improving_count(0),
                        species_threshold(_species_threshold),
                        fitness_threshold(_fitness_threshold),
                        neat_c1(_neat_c1),
                        neat_c2(_neat_c2),
                        neat_c3(_neat_c3),
                        mutation_rate(_mutation_rate),
                        intra_island_crossover_rate(_intra_island_crossover_rate),
                        inter_island_crossover_rate(_inter_island_crossover_rate),
                        generated_genomes(0),
                        inserted_genomes(0),
                        minimal_genome(_seed_genome),
                        generator(_generator) {

    double rate_sum = mutation_rate + intra_island_crossover_rate + inter_island_crossover_rate;
    if (rate_sum != 1.0) {
        mutation_rate = mutation_rate / rate_sum;
        intra_island_crossover_rate = intra_island_crossover_rate / rate_sum;
        inter_island_crossover_rate = inter_island_crossover_rate / rate_sum;
    }

    intra_island_crossover_rate += mutation_rate;
    inter_island_crossover_rate += intra_island_crossover_rate;

    Log::info("Neat speciation strategy, the species threshold is %f. \n", species_threshold);

    //set the generation id for the initial minimal genome
    generated_genomes++;
    minimal_genome->set_generation_id(generated_genomes);
    // set the fitst species with minimal genome
    Neat_Species.push_back(new Species(species_count));
    Log::info("initialized the first species, current neat species size: %d \n", Neat_Species.size() );
    species_count++;
    insert_genome(minimal_genome);

    global_best_genome = NULL;


}

int32_t NeatSpeciationStrategy::get_generated_genomes() const {
    return generated_genomes;
}

int32_t NeatSpeciationStrategy::get_inserted_genomes() const {
    return inserted_genomes;
}

RNN_Genome* NeatSpeciationStrategy::get_best_genome() {
    int32_t best_genome_species = -1;
    double best_fitness = EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)Neat_Species.size(); i++) {
        if (Neat_Species[i]->size() > 0) {
            double island_best_fitness = Neat_Species[i]->get_best_fitness();
            if (island_best_fitness <= best_fitness) {
                best_fitness = island_best_fitness;
                best_genome_species = i;
            }
        }
    }

    if (best_genome_species < 0)  return NULL;
     else {
        return Neat_Species[best_genome_species]->get_best_genome();
    }
}

RNN_Genome* NeatSpeciationStrategy::get_worst_genome() {
    int32_t worst_genome_species = -1;
    double worst_fitness = -EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)Neat_Species.size(); i++) {
        if (Neat_Species[i]->size() > 0) {
            double island_worst_fitness = Neat_Species[i]->get_worst_fitness();
            if (island_worst_fitness > worst_fitness) {
                worst_fitness = island_worst_fitness;
                worst_genome_species = i;
            }
        }
    }

    if (worst_genome_species < 0) {
        return NULL;
    } else {
        return Neat_Species[worst_genome_species]->get_worst_genome();
    }
}


double NeatSpeciationStrategy::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double NeatSpeciationStrategy::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}


//this will insert a COPY, original needs to be deleted
//returns 0 if a new global best, < 0 if not inserted, > 0 otherwise
int32_t NeatSpeciationStrategy::insert_genome(RNN_Genome* genome) {
    bool inserted = false;
    bool erased_population = check_population();
    if (!erased_population) {
            check_species();
    }
    else {
        population_not_improving_count = 0;
    }
    vector<double> best = genome->get_best_parameters();

    if(best.size() != 0){
        genome->set_weights(best);
    }

    Log::info("inserting genome id %d!\n", genome->get_generation_id());
    inserted_genomes++;

    int32_t insert_position;
    if (Neat_Species.size() == 1 && Neat_Species[0]->size() == 0) {
        // insert the first genome in the evolution
        insert_position = Neat_Species[0]->insert_genome(genome);
        Log::info("first genome of this species inserted \n");
        inserted = true;
    }

    if (!inserted) {
        vector<int32_t> species_list = get_random_species_list();
        for (int i = 0; i < species_list.size(); i++){
            Species* random_species = Neat_Species[species_list[i]];
            if (random_species == NULL || random_species->size() == 0) {
                Log::error("random_species is empty\n");
                continue;
            }

            RNN_Genome* genome_representation = random_species->get_latested_genome();
            if (genome_representation == NULL) {
                Log::error("genome representation is null!\n");
                break;
            }

            if (genome_representation == NULL){
                Log::fatal("the latest genome is null, this should never happen!\n");
            }
            double distance = get_distance(genome_representation, genome);

            // Log::error("distance is %f \n", distance);

            if (distance < species_threshold) {
                Log::info("inserting genome to species: %d\n", species_list[i]);
                insert_position = random_species->insert_genome(genome);
                inserted = true;
                break;
            }
        }
    }

    if (!inserted) {
        Species* new_species = new Species(species_count);
        species_count++;
        Neat_Species.push_back(new_species);
        if (species_count != Neat_Species.size()){
            Log::error("this should never happen, the species count is not the same as the number of species we have! \n");
            Log::error("num of species: %d, and species count is %d \n", Neat_Species.size(), species_count);
        }
        insert_position = new_species->insert_genome(genome);
        inserted = true;
    }

    if (insert_position == 0) {
        //check and see if the inserted genome has the same fitness as the best fitness
        //of all islands
        double best_fitness = get_best_fitness();
        if (genome->get_fitness() == best_fitness) {
            population_not_improving_count = 0;
            return 0;
        }
        else {
            population_not_improving_count ++;
            return 1;
        } //was the best for the island but not the global best
    } else {
        population_not_improving_count ++;
        return insert_position; //will be -1 if not inserted, or > 0 if not the global best
    }
}


RNN_Genome* NeatSpeciationStrategy::generate_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover, int32_t number_stir_mutations) {
    //generate the genome from the next island in a round
    //robin fashion.
    RNN_Genome *genome = NULL;
    if (generation_species >= (signed) Neat_Species.size()) generation_species = 0;
    Log::debug("getting species: %d\n", generation_species);

    Species *currentSpecies = Neat_Species[generation_species];

    function<double (RNN_Genome*, RNN_Genome*)> distance_function =
    [=](RNN_Genome *g1, RNN_Genome *g2) {
        return this->get_distance(g1, g2);
    };

    Log::info("generating new genome for species[%d], species_size: %d, mutation_rate: %lf, intra_island_crossover_rate: %lf, inter_island_crossover_rate: %lf\n", generation_species, currentSpecies->size(), mutation_rate, intra_island_crossover_rate, inter_island_crossover_rate);

    if (currentSpecies->size() <= 2) {
        Log::info("current species has less than 2 genomes, doing mutation!\n");
        Log::info("generating genome with id: %d \n", generated_genomes);
        while (genome == NULL) {
            currentSpecies->copy_random_genome(rng_0_1, generator, &genome);

            mutate(1 /*max_mutations*/, genome);

            if (genome->outputs_unreachable()) {
                //no path from at least one input to the outputs
                delete genome;
                genome = NULL;
            }

            // genome->initialize_randomly();
            generated_genomes++;
            genome->set_generation_id(generated_genomes);
            genome->set_group_id(generation_species);

            generation_species++;
            if (generation_species >= (signed) Neat_Species.size()) generation_species = 0;
            return genome;
        }
    } else {
        //first eliminate genomes who have low fitness sharing in this species
            if (currentSpecies->size() > 10){
                currentSpecies->fitness_sharing_remove(fitness_threshold, distance_function);
            }
        //generate a genome via crossover or mutation
        Log::info("current species size %d, doing mutaion or crossover\n", currentSpecies->size());
        Log::info("generating genome with id: %d \n", generated_genomes);
        while (genome == NULL) {
            genome = generate_for_species(rng_0_1, generator, mutate, crossover);
        }
    }
    if (genome != NULL) {
        //set th generation id and increment generated genomes
        generated_genomes++;
        genome->set_generation_id(generated_genomes);
        genome->set_group_id(generation_species);

        //set the island for the genome and increment to the next island
        generation_species++;

        // Neat_Species[generation_species]->set_latest_generation_id(generated_genomes);

    } else {
        Log::fatal("ERROR: genome was NULL at the end of generate genome!\n");
        Log::fatal( "This should never happen.\n");
        exit(1);
    }

    return genome;
}

RNN_Genome* NeatSpeciationStrategy::generate_for_species(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover) {
    //if we haven't filled ALL of the island populations yet, only use mutation
    //otherwise do mutation at %, crossover at %, and island crossover at %
    Species *currentSpecies = Neat_Species[generation_species];
    RNN_Genome* genome;
    double r = rng_0_1(generator);
    if ( r < mutation_rate) {
        Log::info("performing mutation\n");

        currentSpecies->copy_random_genome(rng_0_1, generator, &genome);

        //TODO: make max_mutations an IslandSpeciationStrategy option
        mutate(1 /*max_mutations*/, genome);

    } else if (r < intra_island_crossover_rate || Neat_Species.size() == 1) {
        //intra-island crossover
        Log::info("performing intra-island crossover\n");

        //select two distinct parent genomes in the same island
        RNN_Genome *parent1 = NULL, *parent2 = NULL;
        currentSpecies->copy_two_random_genomes(rng_0_1, generator, &parent1, &parent2);

        genome = crossover(parent1, parent2);
        delete parent1;
        delete parent2;
    } else {
        //inter-island crossover
        // Log::info("performing inter-island crossover\n");

        //get a random genome from this island
        RNN_Genome *parent1 = NULL;
        currentSpecies->copy_random_genome(rng_0_1, generator, &parent1);

        //select a different island randomly
        int32_t other_island = rng_0_1(generator) * (Neat_Species.size() - 1);
        if (other_island >= generation_species) other_island++;

        //get the best genome from the other island
        RNN_Genome *parent2 = Neat_Species[other_island]->get_best_genome()->copy(); // new RNN GENOME

        //swap so the first parent is the more fit parent
        if (parent1->get_fitness() > parent2->get_fitness()) {
            RNN_Genome *tmp = parent1;
            parent1 = parent2;
            parent2 = tmp;
        }

        genome = crossover(parent1, parent2); // new RNN GENOME
        delete parent1;
        delete parent2;
    }

    if (genome->outputs_unreachable()) {
        //no path from at least one input to the outputs
        delete genome;
        genome = NULL;
    }
    return genome;
}

void NeatSpeciationStrategy::print(string indent) const {
    Log::info("%NEAT Species: \n", indent.c_str());
    for (int32_t i = 0; i < (int32_t)Neat_Species.size(); i++) {
        Log::info("%sSpecies %d:\n", indent.c_str(), i);
        Neat_Species[i]->print(indent + "\t");
    }
}

/**
 * Gets speciation strategy information headers for logs
 */
string NeatSpeciationStrategy::get_strategy_information_headers() const {
    string info_header="";
    for (int32_t i = 0; i < (int32_t)Neat_Species.size(); i++) {
        info_header.append(",");
        info_header.append("Species_");
        info_header.append(to_string(i));
        info_header.append("_best_fitness");
        // info_header.append(",");
        // info_header.append("Species_");
        // info_header.append(to_string(i));
        // info_header.append("_worst_fitness");
    }
    return info_header;
}

/**
 * Gets speciation strategy information values for logs
 */
string NeatSpeciationStrategy::get_strategy_information_values() const {
    string info_value="";
    for (int32_t i = 0; i < (int32_t)Neat_Species.size(); i++) {
        double best_fitness = Neat_Species[i]->get_best_fitness();
        // double worst_fitness = Neat_Species[i]->get_worst_fitness();
        info_value.append(",");
        info_value.append(to_string(best_fitness));
        // info_value.append(",");
        // info_value.append(to_string(worst_fitness));
    }
    return info_value;
}

RNN_Genome* NeatSpeciationStrategy::get_global_best_genome() {
    return global_best_genome;
}

vector<int32_t> NeatSpeciationStrategy::get_random_species_list() {
    vector<int32_t> species_list;
    for (int i = 0; i < Neat_Species.size(); i++) {
        species_list.push_back(i);
    }

    shuffle(species_list.begin(), species_list.end(), generator);
    // Log::error("species shuffle list: \n");
    // for (std::vector<int32_t>::iterator it=species_list.begin(); it!=species_list.end(); ++it)
    //     std::cout << ' ' << *it;
    // std::cout << '\n';
    return species_list;
}

double NeatSpeciationStrategy::get_distance(RNN_Genome* g1, RNN_Genome* g2) {

    double distance;
    int E;
    int D;
    int32_t N;
    // d = c1*E/N + c2*D/N + c3*w
    vector<int32_t> innovation1 = g1->get_innovation_list();
    vector<int32_t> innovation2 = g2->get_innovation_list();
    double weight1 = g1-> get_avg_edge_weight();
    double weight2 = g2-> get_avg_edge_weight();
    double w = abs(weight1 - weight2);
    Log::debug("weight difference: %f \n", w);
    if (innovation1.size() >= innovation2.size()){
        N = innovation1.size();

    } else {
        N = innovation2.size();
    }
    if (innovation1.back() == innovation2.back()) {
        E = 0;
    } else if (innovation1.back() > innovation2.back()) {
        E = get_exceed_number(innovation1, innovation2);
    } else {
        // innovation1.back() < innovation2.back()
        E = get_exceed_number(innovation2, innovation1);
    }

    std::vector<int32_t> setunion;
    std::vector<int32_t> intersec;
    std::set_union(innovation1.begin(), innovation1.end(), innovation2.begin(), innovation2.end(), std::inserter(setunion, setunion.begin()));
    std::set_intersection(innovation1.begin(), innovation1.end(), innovation2.begin(), innovation2.end(), std::inserter(intersec, intersec.begin()));

    D = setunion.size() - intersec.size() - E;
    distance = neat_c1 * E / N + neat_c2 * D / N + neat_c3 * w ;
    Log::debug("distance is %f \n", distance);
    return distance;

}
//v1.max > v2.max
int NeatSpeciationStrategy::get_exceed_number(vector<int32_t> v1, vector<int32_t> v2) {
    int exceed = 0;

    for (auto it = v1.rbegin(); it != v1.rend(); ++it)  {
        if(*it > v2.back()){
            exceed++;
        } else {
            break;
        }
    }
    return exceed;
}

void NeatSpeciationStrategy::rank_species() {

    Species* temp;
    double fitness_j1, fitness_j2;

    for (int32_t i = 0; i < Neat_Species.size() - 1; i++)   {
        for (int32_t j = 0; j < Neat_Species.size() - i - 1; j++)  {
            fitness_j1 = Neat_Species[j]->get_best_fitness();
            fitness_j2 = Neat_Species[j+1]->get_best_fitness();
            if (fitness_j1 < fitness_j2) {
                temp = Neat_Species[j];
                Neat_Species[j] = Neat_Species[j+1];
                Neat_Species[j+1]= temp;
            }
        }
    }
    for (int32_t i = 0; i < Neat_Species.size() -1; i++) {
        Log::error("Neat specis rank: %f \n", Neat_Species[i]->get_best_fitness());
    }
}

bool NeatSpeciationStrategy::check_population() {
    bool erased = false;
    // check if the population fitness is not improving for 3000 genomes,
    // if so only save the top 2 species and erase the rest

    if (population_not_improving_count >= 3000) {
        Log::error("the population fitness has not been improved for 3000 genomes, start to erasing \n");
        rank_species();

	    Neat_Species.erase(Neat_Species.begin(), Neat_Species.end()-2);
        if (Neat_Species.size() != 2) {
            Log::error("It should never happen, the population has %d number of species instead of 2! \n", Neat_Species.size());
        }
        for (int i = 0; i < 2; i++){
            Log::error("species %d size %d\n",i, Neat_Species[i]->size() );
            Log::error("species %d fitness %f\n",i, Neat_Species[i]->get_best_fitness() );
            Neat_Species[i]->set_species_not_improving_count(0);
        }
        Log::error("erase finished!\n");
        Log::error ("current number of species: %d \n", Neat_Species.size());
        erased = true;
    }
    return erased;
}

void NeatSpeciationStrategy::check_species() {

    Log::info("checking speies \n");
    auto it = Neat_Species.begin();

    while (it != Neat_Species.end()) {
        if (Neat_Species[it - Neat_Species.begin()]->get_species_not_improving_count() >= 2250) {
            Log::error("Species at position %d hasn't been improving for 2250 genomes, erasing it \n", it - Neat_Species.begin() );
            Log::error ("current number of species: %d \n", Neat_Species.size());
            // Neat_Species[Neat_Species.begin() - it]->erase_species();
            it = Neat_Species.erase(it);
        }
        else {
            ++it;
        }
    }
    Log::info("finished checking species, current number of species: %d \n", Neat_Species.size());
}

void NeatSpeciationStrategy::set_rates(double _mutation_rate, double _intra_island_crossover_rate, double _inter_island_crossover_rate){
    mutation_rate = _mutation_rate;
    intra_island_crossover_rate = _intra_island_crossover_rate;
    inter_island_crossover_rate = _inter_island_crossover_rate;
}
