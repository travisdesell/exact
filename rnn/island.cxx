#include <iomanip>
using std::setw;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <string>
using std::string;
using std::to_string;

#include "island.hxx"
#include "rnn_genome.hxx"

Island::Island(int32_t _id, int32_t _max_size) : id(_id), max_size(_max_size), status(Island::FILLED) {
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

bool Island::is_full() {
    return genomes.size() >= max_size;
}

int32_t Island::contains(RNN_Genome* genome) {
    for (int32_t j = 0; j < (int32_t)genomes.size(); j++) {
        if (genomes[j]->equals(genome)) {
            return j;
        }
    }

    return -1;
}


//returns -1 for not inserted, otherwise the index it was inserted at
//inserts a copy of the genome, caller of the function will need to delete their
//pointer
int32_t Island::insert_genome(RNN_Genome *genome) {
    double new_fitness = genome->get_fitness();

    bool was_inserted = true;

    cout << "inserting genome with fitness : " << parse_fitness(genome->get_fitness()) << " to island " << id << endl;

    //discard the genome if the island is full and it's fitness is worse than the worst in thte population
    if (is_full() && new_fitness > get_worst_fitness()) {
        cout << "ignoring genome, fitness: " << new_fitness << " > worst for island[" << id << "] fitness: " << genomes.back()->get_fitness() << endl;
        return false;
    }

    int32_t duplicate_genome_index = contains(genome);
    if (duplicate_genome_index >= 0) {
        //if fitness is better, replace this genome with new one
        cout << "found duplicate genome at position: " << duplicate_genome_index << endl;

        RNN_Genome *duplicate = genomes[duplicate_genome_index];
        if (duplicate->get_fitness() > new_fitness) {
            //erase the genome with loewr fitness from the vector;
            cout << "REPLACING DUPLICATE GENOME, fitness of genome in search: " << parse_fitness(duplicate->get_fitness()) << ", new fitness: " << parse_fitness(genome->get_fitness()) << endl;
            genomes.erase(genomes.begin() + duplicate_genome_index);
            delete duplicate;

        } else {
            cout << "island already contains genome with a better fitness! not inserting." << endl;
            return -1;
        }
    }

    //inorder insert the new individual
    RNN_Genome *copy = genome->copy();
    cout << "created copy to insert to island: " << copy->get_island() << endl;

    auto index_iterator = genomes.insert( upper_bound(genomes.begin(), genomes.end(), copy, sort_genomes_by_fitness()), copy);
    //calculate the index the genome was inseretd at from the iterator
    int32_t insert_index = index_iterator - genomes.begin();
    cout << "inserted genome at index: " << insert_index << endl;

    if (insert_index == 0) {
        //this was a new best genome for this island

        cout << "new best fitness!" << endl;

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

    if (genomes.size() > max_size) {
        //island was full before insert so now we need to 
        //delete the worst genome in the island.

        cout << "deleting worst genome" << endl;
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

