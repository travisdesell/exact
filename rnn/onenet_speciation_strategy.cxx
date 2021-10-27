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
#include "onenet_speciation_strategy.hxx"

#include "common/log.hxx"

/**
 *
 */
OneNetSpeciationStrategy::OneNetSpeciationStrategy(
        int32_t _elite_population_size, int32_t _generation_size,
        double _mutation_rate, double _crossover_rate,
        RNN_Genome *_seed_genome,
        bool _seed_genome_was_minimal
        ) :
                        elite_population_size(_elite_population_size),
                        generation_size(_generation_size),
                        mutation_rate(_mutation_rate),
                        crossover_rate(_crossover_rate),
                        generated_genomes(0),
                        evaluated_genomes(0),
                        seed_genome(_seed_genome),
                        seed_genome_was_minimal(_seed_genome_was_minimal) {
    double rate_sum = mutation_rate + crossover_rate;
    if (rate_sum != 1.0) {
        mutation_rate = mutation_rate / rate_sum;
        crossover_rate = crossover_rate / rate_sum;
    }

    crossover_rate += mutation_rate;

    Elite_population = new Population(ELITE, elite_population_size);
    Trained_population = new Population(TRAINED, generation_size);
    Generated_population = new Population(GENERATED ,generation_size);
    Log::error("ONENET: generation size is %d\n", generation_size);

    //set the generation id for the initial minimal genome
    seed_genome->set_generation_id(generated_genomes);
    generated_genomes++;
    global_best_genome = NULL;
    global_best_fitness = EXAMM_MAX_DOUBLE;
    generation = 1;
}

int32_t OneNetSpeciationStrategy::get_generated_genomes() const {
    return generated_genomes;
}

int32_t OneNetSpeciationStrategy::get_evaluated_genomes() const {
    return evaluated_genomes;
}

RNN_Genome* OneNetSpeciationStrategy::get_best_genome() {
    //the global_best_genome is updated every time a genome is inserted
    return global_best_genome;
}

RNN_Genome* OneNetSpeciationStrategy::get_worst_genome() {
    return NULL;
}


double OneNetSpeciationStrategy::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double OneNetSpeciationStrategy::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}


//this will insert a COPY, original needs to be deleted
//returns 0 if a new global best, < 0 if not inserted, > 0 otherwise
int32_t OneNetSpeciationStrategy::insert_genome(RNN_Genome* genome) {
    Log::debug("inserting genome!\n");
    int32_t population = genome->get_group_id();

    bool new_global_best = false;
    if (population == ELITE) {
        if (global_best_genome == NULL) {
            //this is the first insert of a genome so it's the global best by default
            global_best_genome = genome->copy();
            global_best_fitness = genome->get_fitness();
            new_global_best = true;
        } else if (global_best_fitness > genome->get_fitness()) {
            //since we're re-setting this to a copy you need to delete it.
            delete global_best_genome;
            global_best_genome = genome->copy();
            global_best_fitness = genome->get_fitness();
            new_global_best = true;
        }
    }

    evaluated_genomes++;


    Log::info("inserting genome to population: %d\n", population);
    
    int insert_position = -1;

    if (population == GENERATED) {
        insert_position = Generated_population->insert_genome(genome);
    } else if (population == TRAINED) {
        insert_position = Trained_population->insert_genome(genome);
    } else if (population == ELITE) {
        insert_position = Elite_population->insert_genome(genome);
    } else {
        Log::fatal("Genome noes not belong to elite population or trained population\n");
        Log::fatal("This should never happen\n");
        exit(1);
    }

    if (insert_position == 0) {
        if (new_global_best) return 0;
        else return 1;
    } else {
        return insert_position; //will be -1 if not inserted, or > 0 if not the global best
    }
}


RNN_Genome* OneNetSpeciationStrategy::generate_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover, int32_t number_stir_mutations=0) {
    //generate the genome from the next population in a round
    //robin fashion.
    RNN_Genome *genome = NULL;

    if (Elite_population->size() < 2) {
        Log::info("Generating seed genomes by mutation only!\n");

        if (Generated_population->is_empty()) {
            Log::debug("starting with minimal genome\n");
            RNN_Genome *genome_copy = seed_genome->copy();

            //the architectures may be the same but we can give each copy of the minimal genome different
            //starting weights for more variety

            if (seed_genome_was_minimal) genome_copy->initialize_randomly();
            if (!genome_copy->tl_with_epigenetic) genome_copy->initialize_randomly();


            //set the generation id for the copy and increment generated genomes
            genome_copy->set_generation_id(generated_genomes);
            // populations[generation_population]->set_latest_generation_id(generated_genomes);
            // genome_copy->set_generation_id(generated_genomes);
            genome_copy->set_group_id(GENERATED);

            Log::debug("inserting genome copy!\n");
            insert_genome(genome_copy);
            //return a copy of the minimal genome to be trained for each population
            genome = genome_copy->copy();
            genome->initialize_randomly();
            genome->set_generation_id(generated_genomes);
            genome->set_group_id(GENERATED);
            generated_genomes++;
            return genome;
        } else {
            Log::info("generated population is not empty, mutating a random genome\n");

            while (genome == NULL) {
                Generated_population->copy_random_genome(rng_0_1, generator, &genome);

                //TODO: make max_mutations an OneNetSpeciationStrategy option
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
            // copy->set_generation_id(generated_genomes);
            // populations[generation_population]->set_latest_generation_id(generated_genomes);
            // generated_genomes++;
            copy->set_generation_id(generated_genomes);
            copy->set_group_id(GENERATED);
            
            insert_genome(copy);

            //also randomly initialize this genome as
            //what it was generated from was also randomly
            //initialized as the population hasn't been
            //filled

            genome->initialize_randomly();
            genome->set_generation_id(generated_genomes);
            genome->set_group_id(GENERATED);
            generated_genomes++;
            return genome;
        }

    } else {
        //generate a genome via crossover or mutation
        Log::info("population is full\n");

        while (genome == NULL) {
            genome = generate_for_filled_population(rng_0_1, generator, mutate, crossover);
            genome->initialize_randomly();
        }

    // } else {
    //     Log::fatal("ERROR: Elite population size %d is neither initializing or full.\n", Elite_population->get_genomes().size());
    //     Log::fatal("This should never happen!\n");
    //     exit(1);
    }

    if (genome != NULL) {
        //set th generation id and increment generated genomes
        RNN_Genome* genome_copy = genome->copy();
        genome_copy->initialize_randomly();
        genome_copy->best_validation_mse = EXAMM_MAX_DOUBLE;
        genome_copy->best_validation_mae = EXAMM_MAX_DOUBLE;
        genome_copy->best_parameters.clear();
        genome_copy->set_generation_id(generated_genomes);
        genome_copy->set_group_id(GENERATED);
        insert_genome(genome_copy);

        genome->set_generation_id(generated_genomes);
        genome->set_group_id(GENERATED);
        generated_genomes++;

    } else {
        Log::fatal("ERROR: genome was NULL at the end of generate genome!\n");
        Log::fatal( "This should never happen.\n");
        exit(1);
    }

    return genome;
}

RNN_Genome* OneNetSpeciationStrategy::generate_for_filled_population(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover){
    //if we haven't filled ALL of the population populations yet, only use mutation
    //otherwise do mutation at %, crossover at %, and population crossover at %

    RNN_Genome* genome;
    double r = rng_0_1(generator);
    if ( r < mutation_rate) {
        Log::info("performing mutation\n");

        Elite_population->copy_random_genome(rng_0_1, generator, &genome);

        //TODO: make max_mutations an OneNetSpeciationStrategy option
        mutate(1 /*max_mutations*/, genome);

    } else {
        //intra-population crossover
        Log::info("performing crossover\n");

        //select two distinct parent genomes in the same population
        RNN_Genome *parent1 = NULL, *parent2 = NULL;
        Elite_population->copy_two_random_genomes(rng_0_1, generator, &parent1, &parent2);

        genome = crossover(parent1, parent2);
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


void OneNetSpeciationStrategy::print(string indent) const {

    Log::info("Elite population \n");
    Elite_population->print(indent + "\t");

    // Log::info("Trained population \n");
    // Trained_population->print(indent + "\t");
}

/**
 * Gets speciation strategy information headers for logs
 */
string OneNetSpeciationStrategy::get_strategy_information_headers() const {
    string info_header = "";
    info_header.append(",");
    info_header.append("Elite_population_best_fitness");
    info_header.append(",");
    info_header.append("Elite_population_worst_fitness");

    return info_header;
}

/**
 * Gets speciation strategy information values for logs
 */
string OneNetSpeciationStrategy::get_strategy_information_values() const {
    string info_value="";
    info_value.append(",");
    info_value.append(to_string(Elite_population->get_best_fitness()));
    info_value.append(",");
    info_value.append(to_string(Elite_population->get_worst_fitness()));
    // info_value.append(",");
    // info_value.append(to_string(Trained_population->get_best_fitness()));
    // info_value.append(",");
    // info_value.append(to_string(Trained_population->get_worst_fitness()));

    return info_value;
}

RNN_Genome* OneNetSpeciationStrategy::get_global_best_genome(){
    return global_best_genome;
}

void OneNetSpeciationStrategy::finalize_generation(string filename, const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output, TimeSeriesSets *time_series_sets, string result_dir) {
    // Log::error("Generated population size %d, trained population size %d\n", Generated_population->get_genomes().size(), Trained_population->get_genomes().size());
    evaluate_elite_population(validation_input, validation_output);
    select_elite_population();
    global_best_genome = Elite_population->get_best_genome();
    Elite_population->write_prediction(filename, test_input, test_output, time_series_sets);
    // make_online_predictions(test_input, test_output);
    generation ++;
    // return global_best_genome;
}

void OneNetSpeciationStrategy::evaluate_elite_population(const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output) {
    vector<RNN_Genome*> elite_genomes = Elite_population->get_genomes();
    if (elite_genomes.size() == 0) return;
    for (int i = 0; i < elite_genomes.size(); i++) {
        RNN_Genome* g = elite_genomes[i];
        g->evaluate_online(validation_input, validation_output);
    }
    Elite_population->sort_population("MSE");
}

void OneNetSpeciationStrategy::select_elite_population() {

    vector<RNN_Genome*> elite_genomes = Elite_population->get_genomes();
    vector<RNN_Genome*> trained_genomes = Trained_population->get_genomes();

    for (int i = 0; i < trained_genomes.size(); i++) {
        RNN_Genome* genome_copy = trained_genomes[i]->copy();
        genome_copy->set_group_id(ELITE);
        Elite_population->insert_genome(genome_copy);
    }

    Trained_population->erase_population();
    Generated_population->erase_population();
    
}

void OneNetSpeciationStrategy::make_online_predictions(const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    double prediction_mse;
    vector<double> best_parameters;

    // global_best_genome = Elite_population->get_best_genome();
    best_parameters = global_best_genome->get_best_parameters();
    prediction_mse = global_best_genome->get_mse(best_parameters, test_input, test_output);
    Log::info("finished online predictions, online prediction mse is %f\n", prediction_mse);
}