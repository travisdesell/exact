#include <algorithm>
using std::sort;
using std::upper_bound;

#include <iomanip>
using std::setw;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;
using std::to_string;

#include "island.hxx"
#include "rnn_genome.hxx"

#include "common/log.hxx"

Island::Island(int32_t _id, int32_t _max_size) : id(_id), max_size(_max_size), status(Island::INITIALIZING) {
}

RNN_Genome* Island::get_best_genome() {
    if (genomes.size() == 0)  return NULL;
    else return genomes[0];
}

RNN_Genome* Island::get_worst_genome() {
    if (genomes.size() == 0)  return NULL;
    else return genomes.back();
}

double Island::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double Island::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

int32_t Island::size() {
    return genomes.size();
}

bool Island::is_full() {
    return genomes.size() >= max_size;
}

bool Island::is_initializing() {
    return status == Island::INITIALIZING;
}

bool Island::is_repopulating() {
    return status == Island::REPOPULATING;
}


int32_t Island::contains(RNN_Genome* genome) {
    for (int32_t j = 0; j < (int32_t)genomes.size(); j++) {
        if (genomes[j]->equals(genome)) {
            return j;
        }
    }

    return -1;
}


void Island::copy_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome) {
    int32_t genome_position = size() * rng_0_1(generator);
    *genome = genomes[genome_position]->copy();
}

void Island::copy_two_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome1, RNN_Genome **genome2) {
    int32_t p1 = size() * rng_0_1(generator);
    int32_t p2 = (size() - 1) * rng_0_1(generator);
    if (p2 >= p1) p2++;

    //swap the gnomes so that the first parent is the more fit parent
    if (p1 > p2) {
        int32_t tmp = p1;
        p1 = p2;
        p2 = tmp;
    }

    *genome1 = genomes[p1]->copy();
    *genome2 = genomes[p2]->copy();
}


//returns -1 for not inserted, otherwise the index it was inserted at
//inserts a copy of the genome, caller of the function will need to delete their
//pointer
int32_t Island::insert_genome(RNN_Genome *genome) {

    if(genome->get_generation_id() <= erased_generation_id){
        Log::info("genome already erased, not inserting");
        return -1;
    }

    Log::debug("getting fitness of genome copy\n");

    double new_fitness = genome->get_fitness();

    Log::info("inserting genome with fitness: %s to island %d\n", parse_fitness(genome->get_fitness()).c_str(), id);

    //discard the genome if the island is full and it's fitness is worse than the worst in thte population
    if (is_full() && new_fitness > get_worst_fitness()) {
        Log::info("ignoring genome, fitness: %lf > worst for island[%d] fitness: %lf\n", new_fitness, id, genomes.back()->get_fitness());
        return false;
    }

    int32_t duplicate_genome_index = contains(genome);
    if (duplicate_genome_index >= 0) {
        //if fitness is better, replace this genome with new one
        Log::info("found duplicate genome at position: %d\n", duplicate_genome_index);

        RNN_Genome *duplicate = genomes[duplicate_genome_index];
        if (duplicate->get_fitness() > new_fitness) {
            //erase the genome with loewr fitness from the vector;
            Log::info("REPLACING DUPLICATE GENOME, fitness of genome in search: %s, new fitness: %s\n", parse_fitness(duplicate->get_fitness()).c_str(), parse_fitness(genome->get_fitness()).c_str());
            genomes.erase(genomes.begin() + duplicate_genome_index);
            delete duplicate;

        } else {
            Log::info("island already contains genome with a better fitness! not inserting.\n");
            return -1;
        }
    }

    //inorder insert the new individual
    RNN_Genome *copy = genome->copy();
    Log::debug("created copy to insert to island: %d\n", copy->get_group_id());

    auto index_iterator = genomes.insert( upper_bound(genomes.begin(), genomes.end(), copy, sort_genomes_by_fitness()), copy);
    //calculate the index the genome was inseretd at from the iterator
    int32_t insert_index = index_iterator - genomes.begin();
    Log::info("inserted genome at index: %d\n", insert_index);

    if (insert_index == 0) {
        //this was a new best genome for this island

        Log::info("new best fitness for island: %d!\n", id);

        if (genome->get_fitness() != EXAMM_MAX_DOUBLE) {
            //need to set the weights for non-initial genomes so we
            //can generate a proper graphviz file
            vector<double> best_parameters = genome->get_best_parameters();
            genome->set_weights(best_parameters);
        }
    }


    if (genomes.size() >= max_size) {
        //the island is filled
        status = Island::FILLED;
    }

    Log::info("genomes.size(): %d, max_size: %d, status: %d\n", genomes.size(), max_size, status);

    if (genomes.size() > max_size) {
        //island was full before insert so now we need to 
        //delete the worst genome in the island.

        Log::debug("deleting worst genome\n");
        RNN_Genome *worst = genomes.back();
        genomes.pop_back();

        delete worst;
    }

    if (insert_index >= max_size) {
        //technically we shouldn't get here but it might happen
        //if the genome's fitness == the worst fitness in the
        //island. So in this case it was not inserted to the
        //island and return -1
        return -1;
    } else {
        return insert_index;
    }
}

void Island::print(string indent) {
    Log::info("%s\t%s\n", indent.c_str(), RNN_Genome::print_statistics_header().c_str());

    for (int32_t i = 0; i < genomes.size(); i++) {
        Log::info("%s\t%s\n", indent.c_str(), genomes[i]->print_statistics().c_str());
    }
}

void Island::erase_island() {

    for (int32_t i = 0; i < genomes.size(); i++) {
        if(genomes[i]->get_generation_id() > erased_generation_id){
            erased_generation_id=genomes[i]->get_generation_id();
        }
    }
    genomes.clear();
    Log::info("Worst island size after erased: %d", genomes.size());
    if(genomes.size()!=0){
        Log::error("The worst island is not fully erased!");
    }
}

int32_t Island::get_erased_generation_id(){

    return erased_generation_id;
}

void Island::set_repopulating(){
    status = REPOPULATING;
}

