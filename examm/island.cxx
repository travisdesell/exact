#include <algorithm>
#include <memory>
using std::upper_bound;

#include <iomanip>
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include "common/log.hxx"
#include "island.hxx"
#include "rnn/rnn_genome.hxx"

Island::Island(int32_t _id, int32_t _max_size, AnnealingPolicy& annealing_policy)
    : id(_id),
      max_size(_max_size),
      annealing_policy(annealing_policy),
      status(Island::INITIALIZING),
      erase_again(0),
      erased(false) {
    using namespace std::chrono;
    long long t = time_point_cast<nanoseconds>(system_clock::now()).time_since_epoch().count();
    generator = minstd_rand0(t);
}

Island::Island(int32_t _id, vector<RNN_Genome*> _genomes, AnnealingPolicy& annealing_policy)
    : id(_id),
      max_size((int32_t) _genomes.size()),
      genomes(_genomes),
      annealing_policy(annealing_policy),
      status(Island::FILLED),
      erase_again(0),
      erased(false) {
    using namespace std::chrono;
    long long t = time_point_cast<nanoseconds>(system_clock::now()).time_since_epoch().count();
    generator = minstd_rand0(t);
}

RNN_Genome* Island::get_best_genome() {
    if (genomes.size() == 0) {
        return NULL;
    } else {
        return genomes[0];
    }
}

RNN_Genome* Island::get_worst_genome() {
    if (genomes.size() == 0) {
        return NULL;
    } else {
        return genomes.back();
    }
}

double Island::get_best_fitness() {
    RNN_Genome* best_genome = get_best_genome();
    if (best_genome == NULL) {
        return EXAMM_MAX_DOUBLE;
    } else {
        return best_genome->get_fitness();
    }
}

double Island::get_worst_fitness() {
    RNN_Genome* worst_genome = get_worst_genome();
    if (worst_genome == NULL) {
        return EXAMM_MAX_DOUBLE;
    } else {
        return worst_genome->get_fitness();
    }
}

int32_t Island::get_max_size() {
    return (int32_t) max_size;
}

int32_t Island::size() {
    return (int32_t) genomes.size();
}

bool Island::is_full() {
    bool filled = (int32_t) genomes.size() >= max_size;
    if (filled) {
        status = Island::FILLED;
    }
    return filled;
}

bool Island::is_initializing() {
    return status == Island::INITIALIZING;
}

bool Island::is_repopulating() {
    return status == Island::REPOPULATING;
}

void Island::copy_random_genome(
    uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator, RNN_Genome** genome
) {
    int32_t genome_position = size() * rng_0_1(generator);
    *genome = genomes[genome_position]->copy();
}

void Island::copy_two_random_genomes(
    uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator, RNN_Genome** genome1, RNN_Genome** genome2
) {
    int32_t p1 = size() * rng_0_1(generator);
    int32_t p2 = (size() - 1) * rng_0_1(generator);
    if (p2 >= p1) {
        p2++;
    }

    // swap the gnomes so that the first parent is the more fit parent
    if (p1 > p2) {
        int32_t tmp = p1;
        p1 = p2;
        p2 = tmp;
    }

    *genome1 = genomes[p1]->copy();
    *genome2 = genomes[p2]->copy();
}

void Island::do_population_check(int32_t line, int32_t initial_size) {
    if (status == Island::FILLED && (int32_t) genomes.size() < max_size) {
        Log::error(
            "ERROR: do_population_check had issue on island.cxx line %d, status was FILLED and genomes.size() was: %d, "
            "size at beginning of insert was: %d\n",
            line, genomes.size(), initial_size
        );
        status = Island::INITIALIZING;
    }
}

// returns -1 for not inserted, otherwise the index it was inserted at
// inserts a copy of the genome, caller of the function will need to delete their
// pointer
int32_t Island::insert_genome(RNN_Genome* genome) {
    int32_t initial_size = (int32_t) genomes.size();
    if (genome->get_generation_id() <= erased_generation_id) {
        Log::trace("genome already erased, not inserting");
        do_population_check(__LINE__, initial_size);
        return -1;
    }
    Log::debug("getting fitness of genome copy\n");
    double new_fitness = genome->get_fitness();
    Log::info("inserting genome with fitness: %s to island %d\n", parse_fitness(genome->get_fitness()).c_str(), id);

    // discard the genome if the island is full and it's fitness is worse than the worst in thte population
    if (is_full() && new_fitness > get_worst_fitness()) {
        Log::debug(
            "ignoring genome, fitness: %lf > worst for island[%d] fitness: %lf\n", new_fitness, id,
            genomes.back()->get_fitness()
        );
        do_population_check(__LINE__, initial_size);
        return false;
    }

    // check and see if the structural hash of the genome is in the
    // set of hashes for this population
    Log::info("getting structural hash\n");
    auto duplicate_it = structure_set.find(genome);

    bool duplicate_exists = duplicate_it != structure_set.end();
    if (duplicate_exists) {
        RNN_Genome* duplicate = *duplicate_it;
        // TODO: Add annealment here
        if (duplicate->get_fitness() > genome->get_fitness()) {
            genomes.erase(std::find(genomes.begin(), genomes.end(), duplicate));
        }
    }

    // inorder insert the new individual
    RNN_Genome* copy = genome->copy();
    copy->set_generation_id(genome->get_generation_id());

    vector<double> best = copy->get_best_parameters();
    if (best.size() != 0) {
        copy->set_weights(best);
    }

    // Only do simulated annealing if the island is full
    // This will with a probability prescribed by the annealing policy (a function of genome number) randomly accept
    // genomes by deleting a random member of the population./
    if (genomes.size() == max_size
        && uniform_real_distribution<>(0.0, 1.0)(generator) < annealing_policy(copy->get_generation_id())) {
        int32_t index = uniform_real_distribution<>(0., 1.)(generator) * genomes.size();

        RNN_Genome* victim = genomes[index];
        genomes.erase(genomes.begin() + index);
        structure_set.erase(victim);
    }

    auto index_iterator = upper_bound(genomes.begin(), genomes.end(), copy, sort_genomes_by_fitness());
    int32_t insert_index = index_iterator - genomes.begin();
    Log::debug("inserting genome at index: %d\n", insert_index);

    if (insert_index >= max_size) {
        // For simulated annealing: if this is true, then we should remove a random member of the population to insert.
        // if we're going to insert this at the back of the population
        // its just going to get removed anyways, so we can delete
        // it and report it was not inserted.
        Log::debug("not inserting genome because it is worse than the worst fitness\n");
        delete copy;
        do_population_check(__LINE__, initial_size);
        return -1;
    }

    genomes.insert(index_iterator, copy);
    structure_set.insert(copy);

    if (insert_index == 0) {
        // this was a new best genome for this island
        Log::info("Island %d: new best fitness found!\n", id);

        if (genome->get_fitness() != EXAMM_MAX_DOUBLE) {
            // need to set the weights for non-initial genomes so we
            // can generate a proper graphviz file
            vector<double> best_parameters = genome->get_best_parameters();
            genome->set_weights(best_parameters);
            Log::info("set genome parameters to best\n");
        }
    }

    if ((int32_t) genomes.size() >= max_size) {
        // the island is filled
        status = Island::FILLED;
    }

    Log::info("Island %d: genomes.size(): %d, max_size: %d, status: %d\n", id, genomes.size(), max_size, status);

    if ((int32_t) genomes.size() > max_size) {
        // island was full before insert so now we need to
        // delete the worst genome in the island.

        Log::debug("deleting worst genome\n");
        RNN_Genome* worst = genomes.back();
        genomes.pop_back();
        structure_set.erase(worst);

        delete worst;
    }

    if (insert_index >= max_size) {
        // technically we shouldn't get here but it might happen
        // if the genome's fitness == the worst fitness in the
        // island. So in this case it was not inserted to the
        // island and return -1
        do_population_check(__LINE__, initial_size);
        return -1;
    } else {
        do_population_check(__LINE__, initial_size);
        return insert_index;
    }
}

void Island::print(string indent) {
    if (Log::at_level(Log::TRACE)) {
        Log::trace("%s\t%s\n", indent.c_str(), RNN_Genome::print_statistics_header().c_str());

        for (int32_t i = 0; i < (int32_t) genomes.size(); i++) {
            Log::trace("%s\t%s\n", indent.c_str(), genomes[i]->print_statistics().c_str());
        }
    }
}

void Island::erase_island() {
    structure_set.clear();

    for (int32_t i = 0; i < (int32_t) genomes.size(); i++) {
        delete genomes[i];
    }

    genomes.clear();

    erased = true;
    erase_again = 5;
    erased_generation_id = latest_generation_id;

    Log::debug("Worst island size after erased: %d\n", genomes.size());
}

int32_t Island::get_erased_generation_id() {
    return erased_generation_id;
}

int32_t Island::get_status() {
    return status;
}

void Island::set_status(int32_t status_to_set) {
    if (status_to_set == Island::INITIALIZING || status_to_set == Island::FILLED
        || status_to_set == Island::REPOPULATING) {
        status = status_to_set;
    } else {
        Log::error("Island::set_status: Wrong island status to set! %d\n", status_to_set);
        exit(1);
    }
}

bool Island::been_erased() {
    return erased;
}

vector<RNN_Genome*> Island::get_genomes() {
    return genomes;
}

void Island::set_latest_generation_id(int32_t _latest_generation_id) {
    latest_generation_id = _latest_generation_id;
}

int32_t Island::get_erase_again_num() {
    return erase_again;
}

void Island::set_erase_again_num() {
    erase_again--;
}

void Island::fill_with_mutated_genomes(
    RNN_Genome* seed_genome, int32_t num_mutations, bool tl_epigenetic_weights,
    function<void(int32_t, RNN_Genome*)>& mutate
) {
    Log::info("Island %d: Filling island with mutated seed genomes\n", id);
    for (int32_t i = 0; i < max_size; i++) {
        RNN_Genome* new_genome = seed_genome->copy();
        mutate(num_mutations, seed_genome);
        new_genome->set_generation_id(0);
        if (tl_epigenetic_weights) {
            new_genome->initialize_randomly();
        }
        genomes.push_back(new_genome);
    }
    if (is_full()) {
        Log::info("island %d: is filled with mutated genome\n", id);
    } else {
        Log::fatal(
            "island (max capacity %d) is still not full after filled with mutated genomes, current island size is %d\n",
            max_size, genomes.size()
        );
        exit(1);
    }
}

void Island::save_population(string output_path) {
    for (int32_t i = 0; i < (int32_t) genomes.size(); i++) {
        RNN_Genome* genome = genomes[i];
        genome->write_graphviz(output_path + "/island_" + to_string(id) + "_genome_" + to_string(i) + ".gv");
        genome->write_to_file(output_path + "/island_" + to_string(id) + "_genome_" + to_string(i) + ".bin");
    }
}
