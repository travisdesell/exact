#include <functional>
using std::function;

#include <chrono>

//#include <iostream>

#include <random>

using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include "examm.hxx"
#include "rnn_genome.hxx"
#include "onenet_island_speciation_strategy.hxx"

#include "common/log.hxx"

/**
 *
 */
OneNetIslandSpeciationStrategy::OneNetIslandSpeciationStrategy(
        int32_t _number_of_islands, int32_t _generated_genome_size, int32_t _elite_genome_size, 
        double _mutation_rate, double _intra_island_crossover_rate,
        double _inter_island_crossover_rate, RNN_Genome *_seed_genome,
        string _island_ranking_method, string _repopulation_method,
        int32_t _extinction_event_generation_number, int32_t _repopulation_mutations,
        int32_t _islands_to_exterminate, int32_t _max_genomes,
        bool _repeat_extinction,
        bool _seed_genome_was_minimal
        ) :
                        generation_island(0),
                        number_of_islands(_number_of_islands),
                        generated_genome_size(_generated_genome_size),
                        elite_genome_size(_elite_genome_size),
                        mutation_rate(_mutation_rate),
                        intra_island_crossover_rate(_intra_island_crossover_rate),
                        inter_island_crossover_rate(_inter_island_crossover_rate),
                        generated_genomes(0),
                        evaluated_genomes(0),
                        seed_genome(_seed_genome),
                        island_ranking_method(_island_ranking_method),
                        repopulation_method(_repopulation_method),
                        extinction_event_generation_number(_extinction_event_generation_number),
                        repopulation_mutations(_repopulation_mutations),
                        islands_to_exterminate(_islands_to_exterminate),
                        max_genomes(_max_genomes),
                        repeat_extinction(_repeat_extinction),
                        seed_genome_was_minimal(_seed_genome_was_minimal) {
    double rate_sum = mutation_rate + intra_island_crossover_rate + inter_island_crossover_rate;
    if (rate_sum != 1.0) {
        mutation_rate = mutation_rate / rate_sum;
        intra_island_crossover_rate = intra_island_crossover_rate / rate_sum;
        inter_island_crossover_rate = inter_island_crossover_rate / rate_sum;
    }

    intra_island_crossover_rate += mutation_rate;
    inter_island_crossover_rate += intra_island_crossover_rate;
    Log::error("generated genome size is %d, elite populaiton size is %d\n", generated_genome_size, elite_genome_size);
    Log::error("mutation rate %f, inter-island crossover rate %f, intra island crossover rate %f\n", mutation_rate, inter_island_crossover_rate, intra_island_crossover_rate);
    for (int32_t i = 0; i < number_of_islands; i++) {
        islands.push_back(new OneNetIsland(i, generated_genome_size, elite_genome_size));
    }

    //set the generation id for the initial minimal genome
    seed_genome->set_generation_id(generated_genomes);
    generated_genomes++;
    global_best_genome = NULL;
}

/**
 *
 */
// OneNetIslandSpeciationStrategy::OneNetIslandSpeciationStrategy(
//         int32_t _number_of_islands, int32_t _max_island_size,
//         double _mutation_rate, double _intra_island_crossover_rate,
//         double _inter_island_crossover_rate, RNN_Genome *_seed_genome,
//         string _island_ranking_method, string _repopulation_method,
//         int32_t _extinction_event_generation_number, int32_t _repopulation_mutations,
//         int32_t _islands_to_exterminate, bool _seed_genome_was_minimal, function<void (RNN_Genome*)> &modify) :
//                         generation_island(0),
//                         number_of_islands(_number_of_islands),
//                         max_island_size(_max_island_size),
//                         mutation_rate(_mutation_rate),
//                         intra_island_crossover_rate(_intra_island_crossover_rate),
//                         inter_island_crossover_rate(_inter_island_crossover_rate),
//                         generated_genomes(0),
//                         evaluated_genomes(0),
//                         seed_genome(_seed_genome),
//                         island_ranking_method(_island_ranking_method),
//                         repopulation_method(_repopulation_method),
//                         extinction_event_generation_number(_extinction_event_generation_number),
//                         repopulation_mutations(_repopulation_mutations),
//                         islands_to_exterminate(_islands_to_exterminate),
//                         seed_genome_was_minimal(_seed_genome_was_minimal) {

//     double rate_sum = mutation_rate + intra_island_crossover_rate + inter_island_crossover_rate;
//     if (rate_sum != 1.0) {
//         mutation_rate = mutation_rate / rate_sum;
//         intra_island_crossover_rate = intra_island_crossover_rate / rate_sum;
//         inter_island_crossover_rate = inter_island_crossover_rate / rate_sum;
//     }

//     intra_island_crossover_rate += mutation_rate;
//     inter_island_crossover_rate += intra_island_crossover_rate;

//     auto make_filled_island = [](int32_t id, RNN_Genome *seed_genome, int32_t size, int32_t nmutations, function<void (RNN_Genome*)> &modify) {
//         vector<RNN_Genome*> genomes;
//         genomes.reserve(size);
//         for (int i = 0 ; i < size ; i += 1) {
//             RNN_Genome *clone = seed_genome->copy();
//             modify(clone);
//             clone->set_generation_id(0);
//             genomes.push_back(clone);
//         }

//         return new OneNetIsland(id, genomes);
//     };

//     for (int i = 0 ; i < number_of_islands; i += 1)
//         islands.push_back(make_filled_island(i, _seed_genome, max_island_size, repopulation_mutations, modify));

//     //set the generation id for the initial minimal genome

//     seed_genome->set_generation_id(0);

//     generated_genomes++;
//     global_best_genome = NULL;
// }

int32_t OneNetIslandSpeciationStrategy::get_generated_genomes() const {
    return generated_genomes;
}

int32_t OneNetIslandSpeciationStrategy::get_evaluated_genomes() const {
    return evaluated_genomes;
}

RNN_Genome* OneNetIslandSpeciationStrategy::get_best_genome() {
    //the global_best_genome is updated every time a genome is inserted
    return global_best_genome;
}

RNN_Genome* OneNetIslandSpeciationStrategy::get_worst_genome() {
    int32_t worst_genome_island = -1;
    double worst_fitness = -EXAMM_MAX_DOUBLE;

    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i]->elite_size() > 0) {
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


double OneNetIslandSpeciationStrategy::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double OneNetIslandSpeciationStrategy::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

bool OneNetIslandSpeciationStrategy::islands_full() const {
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (!islands[i]->elite_is_full()) return false;
    }

    return true;
}


//this will insert a COPY, original needs to be deleted
//returns 0 if a new global best, < 0 if not inserted, > 0 otherwise
int32_t OneNetIslandSpeciationStrategy::insert_genome(RNN_Genome* genome) {
    Log::debug("inserting genome!\n");
    if (extinction_event_generation_number != 0){
        if(evaluated_genomes > 1 && evaluated_genomes % extinction_event_generation_number == 0 && max_genomes - evaluated_genomes >= extinction_event_generation_number) {
            if (island_ranking_method.compare("EraseWorst") == 0 || island_ranking_method.compare("") == 0){
                global_best_genome = get_best_genome()->copy();
                vector<int32_t> rank = rank_islands();
                for (int32_t i = 0; i < islands_to_exterminate; i++){
                    if (rank[i] >= 0){
                        Log::info("found island: %d is the worst island \n",rank[0]);
                        islands[rank[i]]->erase_island();
                        // islands[rank[i]]->erase_structure_map();
                        islands[rank[i]]->set_status(OneNetIsland::REPOPULATING);
                    }
                    else Log::error("Didn't find the worst island!");
                    // set this so the island would not be re-killed in 5 rounds
                    if (!repeat_extinction) {
                        set_erased_islands_status();
                    }
                }
            }
        }
    }

    bool new_global_best = false;
    if (global_best_genome == NULL) {
        //this is the first insert of a genome so it's the global best by default
        global_best_genome = genome->copy();
        new_global_best = true;
    } else if (global_best_genome->get_fitness() > genome->get_fitness()) {
        //since we're re-setting this to a copy you need to delete it.
        delete global_best_genome;
        global_best_genome = genome->copy();
        new_global_best = true;
    }

    evaluated_genomes++;
    int32_t island = genome->get_group_id();

    Log::info("inserting genome to island: %d\n", island);

    int32_t insert_position = islands[island]->insert_genome(genome);

    if (insert_position == 0) {
        if (new_global_best) return 0;
        else return 1;
    } else {
        return insert_position; //will be -1 if not inserted, or > 0 if not the global best
    }
}

int32_t OneNetIslandSpeciationStrategy::get_worst_island_by_best_genome() {
    int32_t worst_island = -1;
    double worst_best_fitness = 0;
    for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
        if (islands[i]->elite_size() > 0) {
            if (islands[i]->get_erase_again_num() > 0) continue;
            double island_best_fitness = islands[i]->get_best_fitness();
            if (island_best_fitness > worst_best_fitness) {
                worst_best_fitness = island_best_fitness;
                worst_island = i;
            }
        }
    }
    return worst_island;
}

vector<int32_t> OneNetIslandSpeciationStrategy::rank_islands() {
    vector<int32_t> island_rank;
    int32_t temp;
    double fitness_j1, fitness_j2;
    Log::info("ranking islands \n");
    Log::info("repeat extinction: %s \n", repeat_extinction? "true":"false");
    for (int32_t i = 0; i< number_of_islands; i++){
        if (repeat_extinction) {
            island_rank.push_back(i);
        } else {
            if (islands[i] -> get_erase_again_num() == 0) {
                island_rank.push_back(i);
            }
        }
    }

    for (int32_t i = 0; i < island_rank.size() - 1; i++)   {
        for (int32_t j = 0; j < island_rank.size() - i - 1; j++)  {
            fitness_j1 = islands[island_rank[j]]->get_best_fitness();
            fitness_j2 = islands[island_rank[j+1]]->get_best_fitness();
            if (fitness_j1 < fitness_j2) {
                temp = island_rank[j];
                island_rank[j] = island_rank[j+1];
                island_rank[j+1]= temp;
            }
        }
    }
    Log::info("island rank: \n");
    for (int32_t i = 0; i< island_rank.size(); i++){
        Log::info("island: %d fitness %f \n", island_rank[i], islands[island_rank[i]]->get_best_fitness());
    }
    return island_rank;
}


RNN_Genome* OneNetIslandSpeciationStrategy::generate_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover, int32_t number_stir_mutations=0) {
    //generate the genome from the next island in a round
    //robin fashion.
    RNN_Genome *genome = NULL;

    Log::info("getting island: %d\n", generation_island);
    OneNetIsland *island = islands[generation_island];

    // Log::info("generating new genome for island[%d], island_size: %d, max_island_size: %d, mutation_rate: %lf, intra_island_crossover_rate: %lf, inter_island_crossover_rate: %lf\n", generation_island, island->size(), generated_genome_size, mutation_rate, intra_island_crossover_rate, inter_island_crossover_rate);

    Log::info("islands.size(): %d, selected island is null? %d\n", islands.size(), (island == NULL));

    if (island->is_initializing()) {
        Log::info("island is initializing!\n");

        if (island->generated_size() == 0 ) {
            Log::info("starting with minimal genome\n");
            RNN_Genome *genome_copy = seed_genome->copy();

            //the architectures may be the same but we can give each copy of the minimal genome different
            //starting weights for more variety

            if (seed_genome_was_minimal) genome_copy->initialize_randomly();
            if (!genome_copy->tl_with_epigenetic) genome_copy->initialize_randomly();

            // This is commented out because transfer learning islands should probably start out filled up.
            // it is probably going to be removed soon.

            // Stir the seed genome if need be
            if (number_stir_mutations > 0) {
                Log::info("Stirring seed genome for island %d by applying %d mutations!\n",
                            generation_island, number_stir_mutations);
                mutate(number_stir_mutations, genome_copy);
            }

            //set the generation id for the copy and increment generated genomes
            genome_copy->set_generation_id(generated_genomes);
            islands[generation_island]->set_latest_generation_id(generated_genomes);
            generated_genomes++;
            genome_copy->set_generation_id(generated_genomes);
            genome_copy->set_group_id(generation_island);
            genome_copy->set_genome_type(GENERATED);

            Log::info("inserting genome copy!\n");
            insert_genome(genome_copy);
            //return a copy of the minimal genome to be trained for each island
            genome = genome_copy->copy();
            genome->set_generation_id(generated_genomes);
            genome->set_genome_type(GENERATED);
            generated_genomes++;
            return genome;
        } else {
            Log::info("island is not empty, mutating a random genome\n");

            while (genome == NULL) {
                island->copy_random_genome(rng_0_1, generator, &genome);

                //TODO: make max_mutations an OneNetIslandSpeciationStrategy option
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
            islands[generation_island]->set_latest_generation_id(generated_genomes);
            generated_genomes++;
            copy->set_generation_id(generated_genomes);
            copy->set_group_id(generation_island);
            copy->set_genome_type(GENERATED);
            insert_genome(copy);

            //also randomly initialize this genome as
            //what it was generated from was also randomly
            //initialized as the population hasn't been
            //filled

            genome->initialize_randomly();
            genome->set_generation_id(generated_genomes);
            genome->set_genome_type(GENERATED);
            generated_genomes++;
            generation_island++;
            if (generation_island >= (signed) islands.size()) generation_island = 0;
            islands[generation_island] -> set_latest_generation_id(generated_genomes);
            return genome;
        }

    } else if (island->elite_is_full()) {
        //generate a genome via crossover or mutation
        Log::info("island is full\n");
        island->set_status(OneNetIsland::FILLED);
        while (genome == NULL) {
            genome = generate_for_filled_island(rng_0_1, generator, mutate, crossover);
        }

    } else if (island->is_repopulating()) {
        //select two other islands (non-overlapping) at random, and select genomes
        //from within those islands and generate a child via crossover

        Log::error("island is repopulating \n");

        while (genome == NULL) {
            if (repopulation_method.compare("randomParents") == 0 || repopulation_method.compare("randomparents") == 0){
                Log::info("island is repopulating through random parents method!\n");
                genome = parents_repopulation("randomParents", rng_0_1, generator, mutate, crossover);

            } else if (repopulation_method.compare("bestParents") == 0 || repopulation_method.compare("bestparents") == 0){
                Log::info("island is repopulating through best parents method!\n");
                genome = parents_repopulation("bestParents", rng_0_1, generator, mutate, crossover);

            } else if (repopulation_method.compare("bestGenome") == 0 || repopulation_method.compare("bestgenome") == 0){
                genome = get_global_best_genome()->copy();
                if (repopulation_mutations){
                    mutate(repopulation_mutations, genome);
                }

            } else if (repopulation_method.compare("bestIsland") == 0 || repopulation_method.compare("bestisland") == 0){
                //copy the best island to the worst at once
                //after the worst island is filled, set the island status to filled
                //then generate a genome for filled status, so this function still return a generated genome
                Log::info("island is repopulating through bestIsland method! Coping the best island to the population island\n");
                Log::info("island current size is: %d \n", islands[generation_island]->get_genomes().size());
                RNN_Genome *best_genome = get_best_genome()->copy();
                int32_t best_island_id = best_genome->get_group_id();
                fill_island(best_island_id, mutate);

                if (island->elite_is_full()) {
                    Log::info("island is full now, and generating a new one!\n");
                    island->set_status(OneNetIsland::FILLED);
                } else {
                    Log::error("Island is not full after coping the best island over!\n");
                }

                while (genome == NULL) {
                    genome = generate_for_filled_island(rng_0_1, generator, mutate, crossover);
                }

            } else {
                Log::fatal("Wrong repopulation_method argument");
                exit(1);
            }
        }

    } else {
        Log::fatal("ERROR: island was neither initializing, repopulating or full.\n");
        Log::fatal("This should never happen!\n");

    }

    if (genome != NULL) {
        //set th generation id and increment generated genomes
        generated_genomes++;
        genome->set_generation_id(generated_genomes);
        genome->set_group_id(generation_island);
        genome->set_genome_type(GENERATED);
        //set the island for the genome and increment to the next island
        generation_island++;
        if (generation_island >= (signed) islands.size()) generation_island = 0;
        islands[generation_island] -> set_latest_generation_id(generated_genomes);

    } else {
        Log::fatal("ERROR: genome was NULL at the end of generate genome!\n");
        Log::fatal( "This should never happen.\n");
        exit(1);
    }

    return genome;
}

RNN_Genome* OneNetIslandSpeciationStrategy::generate_for_filled_island(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover){
    //if we haven't filled ALL of the island populations yet, only use mutation
    //otherwise do mutation at %, crossover at %, and island crossover at %
    OneNetIsland *island = islands[generation_island];
    RNN_Genome* genome;
    double r = rng_0_1(generator);
    if (!islands_full() || r < mutation_rate) {
        Log::info("performing mutation\n");

        island->copy_random_genome(rng_0_1, generator, &genome);

        //TODO: make max_mutations an OneNetIslandSpeciationStrategy option
        mutate(1 /*max_mutations*/, genome);

    } else if (r < intra_island_crossover_rate || number_of_islands == 1) {
        //intra-island crossover
        Log::info("performing intra-island crossover\n");

        //select two distinct parent genomes in the same island
        RNN_Genome *parent1 = NULL, *parent2 = NULL;
        island->copy_two_random_genomes(rng_0_1, generator, &parent1, &parent2);

        genome = crossover(parent1, parent2);
        delete parent1;
        delete parent2;
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
        RNN_Genome *parent2 = islands[other_island]->get_best_genome()->copy(); // new RNN GENOME

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


void OneNetIslandSpeciationStrategy::print(string indent) const {
    // Log::info("%sIslands: \n", indent.c_str());
    // for (int32_t i = 0; i < (int32_t)islands.size(); i++) {
    //     Log::info("%sIsland %d:\n", indent.c_str(), i);
    //     islands[i]->print(indent + "\t");
    // }
}

/**
 * Gets speciation strategy information headers for logs
 */
string OneNetIslandSpeciationStrategy::get_strategy_information_headers() const {
    string info_header = "";
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
string OneNetIslandSpeciationStrategy::get_strategy_information_values() const {
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

RNN_Genome* OneNetIslandSpeciationStrategy::parents_repopulation(string method, uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover){
    RNN_Genome* genome = NULL;

    Log::info("generation island: %d \n", generation_island);
    int32_t parent_island1;
    do {
        parent_island1 = (number_of_islands - 1) * rng_0_1(generator);
    } while (parent_island1 == generation_island);

    Log::info("parent island 1: %d \n", parent_island1);
    int32_t parent_island2;
    do {
        parent_island2 = (number_of_islands - 1) * rng_0_1(generator);
    } while (parent_island2 == generation_island || parent_island2 == parent_island1);

    Log::info("parent island 2: %d \n", parent_island2);
    RNN_Genome *parent1 = NULL;
    RNN_Genome *parent2 = NULL;

    while (parent1 == NULL) {
        if (method.compare("randomParents") == 0) {
            islands[parent_island1]->copy_random_genome(rng_0_1, generator, &parent1);
        } else if (method.compare("bestParents") == 0) {
            parent1 = islands[parent_island1]->get_best_genome();
        }
    }

    while (parent2 == NULL) {
        if (method.compare("randomParents") == 0) {
            islands[parent_island2]->copy_random_genome(rng_0_1, generator, &parent2);
        } else if (method.compare("bestParents") == 0) {
            parent2 = islands[parent_island2]->get_best_genome();
        }   
    }

    Log::info("current island is %d, the parent1 island is %d, parent 2 island is %d\n", generation_island, parent_island1, parent_island2);

    //swap so the first parent is the more fit parent
    if (parent1->get_fitness() > parent2->get_fitness()) {
        RNN_Genome *tmp = parent1;
        parent1 = parent2;
        parent2 = tmp;
    }
    genome = crossover(parent1, parent2);

    if (repopulation_mutations > 0) {
        Log::info("Doing %d mutations to the child genome generated by %s\n", repopulation_mutations, method.c_str());
        mutate(repopulation_mutations, genome);
    }

    if (genome->outputs_unreachable()) {
        //no path from at least one input to the outputs
        delete genome; genome = NULL;
    }
    return genome;
}

void OneNetIslandSpeciationStrategy::fill_island(int32_t best_island_id, function<void (int32_t, RNN_Genome*)> &mutate){
    vector<RNN_Genome*>best_island = islands[best_island_id]->get_genomes();
    for (uint32_t i = 0; i < best_island.size(); i++){
        // copy the genome from the best island
        RNN_Genome *copy = best_island[i]->copy();
        generated_genomes++;
        copy->set_generation_id(generated_genomes);
        islands[generation_island] -> set_latest_generation_id(generated_genomes);
        copy->set_group_id(generation_island);
        if (repopulation_mutations > 0) {
            Log::info("Doing %d mutations to genome %d before inserted to the repopulating island\n", repopulation_mutations,copy->generation_id);
            mutate(repopulation_mutations, copy);
        }
        insert_genome(copy);
        delete copy;
    }
}

RNN_Genome* OneNetIslandSpeciationStrategy::get_global_best_genome(){
    return global_best_genome;
}

void OneNetIslandSpeciationStrategy::set_erased_islands_status() {
    for (int i = 0; i < islands.size(); i++) {
        if (islands[i] -> get_erase_again_num() > 0) {
            islands[i] -> set_erase_again_num();
            Log::info("Island %d can be removed in %d rounds.\n", i, islands[i] -> get_erase_again_num());
        }
    }
}

void OneNetIslandSpeciationStrategy::finalize_generation(string filename, const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output, TimeSeriesSets *time_series_sets, string result_dir) {
    // Log::error("finalizing the generation\n");
    // Log::error("Generated population size %d, trained population size %d\n", Generated_population->get_genomes().size(), Trained_population->get_genomes().size());
    evaluate_elite_population(validation_input, validation_output);
    select_elite_population();
    global_best_genome = get_best_genome();
    for (int i = 0; i < number_of_islands; i++) {
        islands[i]->write_prediction(filename, test_input, test_output);
    }
    // Elite_population->write_prediction(filename, test_input, test_output, time_series_sets);
    // make_online_predictions(test_input, test_output);
    // generation ++;
    // return global_best_genome;
}

void OneNetIslandSpeciationStrategy::evaluate_elite_population(const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output) {
    // vector<RNN_Genome*> elite_genomes = Elite_population->get_genomes();
    for (int i = 0; i < number_of_islands; i++) {
        islands[i] -> evaluate_elite_population(validation_input, validation_output);
    }
}

void OneNetIslandSpeciationStrategy::select_elite_population() {
    for (int i = 0; i < number_of_islands; i++) {
        islands[i] -> select_elite_population();
    }
    
}

void OneNetIslandSpeciationStrategy::make_online_predictions(const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    double prediction_mse;
    vector<double> best_parameters;

    // global_best_genome = Elite_population->get_best_genome();
    best_parameters = global_best_genome->get_best_parameters();
    prediction_mse = global_best_genome->get_mse(best_parameters, test_input, test_output);
    Log::info("finished online predictions, online prediction mse is %f\n", prediction_mse);
}