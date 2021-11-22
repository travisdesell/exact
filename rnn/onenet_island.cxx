#include <algorithm>
using std::sort;
using std::lower_bound;
using std::upper_bound;

#include <iomanip>
using std::setw;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;
using std::to_string;

#include <unordered_map>
using std::unordered_map;

#include "onenet_island.hxx"
// #include "rnn_ge/nome.hxx"

#include "common/log.hxx"

OneNetIsland::OneNetIsland(int32_t _id, int32_t _generated_size, int32_t _elite_size) {
    id = _id;
    generate_population_size = _generated_size;
    elite_population_size = _elite_size;
    status = INITIALIZING;
    erase_again = 0; 
    erased = false;

    elite_population = new Population(ELITE, elite_population_size, id);
    generated_population = new Population(GENERATED, generate_population_size, id);
}

OneNetIsland::OneNetIsland(int32_t _id, Population* _elite_population, int32_t _generated_size, int32_t _elite_size) : id(_id), generate_population_size(_generated_size), elite_population_size(_elite_size), elite_population(_elite_population), status(OneNetIsland::FILLED), erase_again(0), erased(false) {
    elite_population = _elite_population;
    generated_population = new Population(GENERATED, generate_population_size, id);
}

RNN_Genome* OneNetIsland::get_best_genome() {
    return elite_population->get_best_genome();
}

RNN_Genome* OneNetIsland::get_worst_genome() {
    return elite_population->get_worst_genome();
}

double OneNetIsland::get_best_fitness() {
    // RNN_Genome *best_genome = elite_population->get_best_genome();
    // if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    // else return best_genome->get_fitness();
    return elite_population->get_best_fitness();
}

double OneNetIsland::get_worst_fitness() {
    // RNN_Genome *worst_genome = elite_population->get_worst_genome();
    // if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    // else return worst_genome->get_fitness();
    return elite_population->get_worst_fitness();
}

// int32_t OneNetIsland::get_max_size() {
//     return max_size;
// }

int32_t OneNetIsland::generated_size() {
    return generated_population->size();
}

int32_t OneNetIsland::elite_size() {
    return elite_population->size();
}

bool OneNetIsland::elite_is_full() {
    return elite_population->is_full();
}

bool OneNetIsland::generated_is_full() {
    return generated_population->is_full();
}

bool OneNetIsland::is_initializing() {
    return status == OneNetIsland::INITIALIZING;
}

// bool OneNetIsland::is_filled() {
//     return status == OneNetIsland::FILLED;
// }

bool OneNetIsland::is_repopulating() {
    return status == OneNetIsland::REPOPULATING;
}



void OneNetIsland::copy_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome) {
    // int32_t genome_position = size() * rng_0_1(generator);
    // *genome = genomes[genome_position]->copy();
    if (elite_size() != 0) {
        elite_population->copy_random_genome(rng_0_1, generator, genome);
    } else {
        generated_population->copy_random_genome(rng_0_1, generator, genome);
    }

}

void OneNetIsland::copy_two_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome1, RNN_Genome **genome2) {
    if (elite_size() >= 2) {
        elite_population->copy_two_random_genomes(rng_0_1, generator, genome1, genome2);
    } else if (generated_size() >= 2) {
        generated_population->copy_two_random_genomes(rng_0_1, generator, genome1, genome2);
    } else {
        Log::fatal("CANNOT copy two random genomes, elite population size is %d, generated population size is %d\n", elite_size(), generated_size());
    }

}

// void OneNetIsland::do_population_check(int line, int initial_size) {
//     if (status == OneNetIsland::FILLED && genomes.size() < max_size) {
//         Log::error("ERROR: do_population_check had issue on island.cxx line %d, status was FILLED and genomes.size() was: %d, size at beginning of insert was: %d\n", line, genomes.size(), initial_size);
//         status = OneNetIsland::INITIALIZING;
//     }
// }


//returns -1 for not inserted, otherwise the index it was inserted at
//inserts a copy of the genome, caller of the function will need to delete their
//pointer
int32_t OneNetIsland::insert_genome(RNN_Genome *genome) {
    int32_t insert_position;
    if (genome->get_genome_type() == GENERATED) {
        Log::info("inserting genome to generated population\n");
        insert_position = generated_population->insert_genome(genome);
    } else if (genome->get_genome_type() == ELITE) {
        Log::info("inserting genome to elite population\n");

        insert_position = elite_population->insert_genome(genome);
    } else {
        Log::fatal("genome type is %d, not inserting \n", genome->get_genome_type());
        exit(1);
    }
    if (elite_size() >= elite_population_size) {
        //the island is filled
        status = OneNetIsland::FILLED;
    }
    return insert_position;
}

// void OneNetIsland::print(string indent) {
//     if (Log::at_level(Log::INFO)) {

//         Log::info("%s\t%s\n", indent.c_str(), RNN_Genome::print_statistics_header().c_str());

//         for (int32_t i = 0; i < genomes.size(); i++) {
//             Log::info("%s\t%s\n", indent.c_str(), genomes[i]->print_statistics().c_str());
//         }
//     }
// }

void OneNetIsland::erase_island() {
    erased_generation_id = latest_generation_id;
    elite_population->erase_population();
    generated_population->erase_population();
    erased = true;
    erase_again = 5;
}

// void OneNetIsland::erase_structure_map() {
//     Log::info("Erasing the structure map in the worst performing island\n");
//     structure_map.clear();
//     Log::debug("after erase structure map size is %d\n", structure_map.size());
// }

int32_t OneNetIsland::get_erased_generation_id() {
    return erased_generation_id;
}

int32_t OneNetIsland::get_status() {
    return status;
}

void OneNetIsland::set_status(int32_t status_to_set) {
    if (status_to_set == OneNetIsland::INITIALIZING || status_to_set == OneNetIsland::FILLED || status_to_set == OneNetIsland::REPOPULATING) {
        status = status_to_set;
    } else {
        Log::error("OneNetIsland::set_status: Wrong island status to set! %d\n", status_to_set);
        exit(1);
    }
}

bool OneNetIsland::been_erased() {
    return erased;
}

vector<RNN_Genome *> OneNetIsland::get_genomes() {
    return elite_population->get_genomes();
}

void OneNetIsland::set_latest_generation_id(int32_t _latest_generation_id){
    latest_generation_id = _latest_generation_id;
}

int32_t OneNetIsland::get_erase_again_num() {
    return erase_again;
}

void OneNetIsland::set_erase_again_num() {
    erase_again--;
}

void OneNetIsland::evaluate_elite_population(const vector< vector< vector<double> > > &validation_input, const vector< vector< vector<double> > > &validation_output) {
        // if (elite_genomes.size() == 0) return;
    vector<RNN_Genome *> elite_genomes = elite_population->get_genomes();
    for (int i = 0; i < elite_genomes.size(); i++) {
        RNN_Genome* g = elite_genomes[i];
        g->evaluate_online(validation_input, validation_output);
    }
    elite_population->sort_population("MSE");
}


void OneNetIsland::select_elite_population() {
    // vector<RNN_Genome*> elite_genomes = Elite_population->get_genomes();
    vector<RNN_Genome*> trained_genomes = generated_population->get_genomes();

    for (int i = 0; i < trained_genomes.size(); i++) {
        RNN_Genome* genome_copy = trained_genomes[i]->copy();
        genome_copy->set_genome_type(ELITE);
        elite_population->insert_genome(genome_copy);
    }

    generated_population->erase_population();

}

void OneNetIsland::write_prediction(string filename, const vector< vector< vector<double> > > &test_input, const vector< vector< vector<double> > > &test_output) {
    elite_population->write_prediction(filename, test_input, test_output);
}

