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

#include "species.hxx"
#include "rnn_genome.hxx"

#include "common/log.hxx"
        // Species(int32_t id, double fitness_th);
Species::Species(int32_t _id) : id(_id), species_not_improving_count(0) {
}

RNN_Genome* Species::get_best_genome() {
    if (genomes.size() == 0)  return NULL;
    else return genomes[0];
}

RNN_Genome* Species::get_worst_genome() {
    if (genomes.size() == 0)  return NULL;
    else return genomes.back();
}

RNN_Genome* Species::get_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator) {
    if (genomes.size() == 0)  return NULL;
    else {
        int32_t genome_position = size() * rng_0_1(generator);
        return genomes[genome_position];
    }
}

double Species::get_best_fitness() {
    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return best_genome->get_fitness();
}

double Species::get_worst_fitness() {
    RNN_Genome *worst_genome = get_worst_genome();
    if (worst_genome == NULL) return EXAMM_MAX_DOUBLE;
    else return worst_genome->get_fitness();
}

int32_t Species::size() {
    return genomes.size();
}

int32_t Species::contains(RNN_Genome* genome) {
    for (int32_t j = 0; j < (int32_t)genomes.size(); j++) {
        if (genomes[j]->equals(genome)) {
            return j;
        }
    }

    return -1;
}


void Species::copy_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome) {
    int32_t genome_position = size() * rng_0_1(generator); 
    *genome = genomes[genome_position]->copy();
}

void Species::copy_two_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome1, RNN_Genome **genome2) {
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
int32_t Species::insert_genome(RNN_Genome *genome) {
    Log::info("inserting genome with fitness: %s to species %d\n", parse_fitness(genome->get_fitness()).c_str(), id);

    // inorder insert the new individual
    RNN_Genome *copy = genome->copy();
    copy->set_generation_id(genome->get_generation_id());
    copy->set_group_id(id);
    vector<double> best = copy->get_best_parameters();

    if(best.size() != 0){
        copy->set_weights(copy->get_best_parameters());
    }
    vector<double> copy_weights;
    copy->get_weights(copy_weights);
    auto index_iterator = genomes.insert( upper_bound(genomes.begin(), genomes.end(), copy, sort_genomes_by_fitness()), copy);
   
    // calculate the index the genome was inseretd at from the iterator
    int32_t insert_index = index_iterator - genomes.begin();

    if (insert_index == 0) {
        // this was a new best genome for this island
        Log::info("new best fitness for island: %d!\n", id);
        if (genome->get_fitness() != EXAMM_MAX_DOUBLE) {
            // need to set the weights for non-initial genomes so we
            // can generate a proper graphviz file
            vector<double> best_parameters = genome->get_best_parameters();
            genome->set_weights(best_parameters);
        }
        species_not_improving_count = 0;
    } else  {
        species_not_improving_count ++;
    }

    inserted_genome_id.push_back( copy->get_generation_id());

    Log::info("Inserted genome %d at index %d\n", genome->get_generation_id(), insert_index);
    return insert_index;
}

void Species::print(string indent) {
    Log::info("%s\t%s\n", indent.c_str(), RNN_Genome::print_statistics_header().c_str());
    for (int32_t i = 0; i < genomes.size(); i++) {
        Log::info("%s\t%s\n", indent.c_str(), genomes[i]->print_statistics().c_str());
    }
}

vector<RNN_Genome *> Species::get_genomes() {
    return genomes;
}

RNN_Genome* Species::get_latested_genome() {
    RNN_Genome* latest = NULL;
    for (auto it = inserted_genome_id.rbegin(); it != inserted_genome_id.rend(); ++it){
        int32_t latest_id = *it;
        for (int i = 0; i < genomes.size(); i++) {
            if (genomes[i]->get_generation_id() == latest_id) {
                latest = genomes[i];
                break;
            }
        } 
        if (latest) {
            break;
        }
    }
    return latest;
}

void Species::fitness_sharing_remove(double fitness_threshold, function<double (RNN_Genome*, RNN_Genome*)> &get_distance) {
    int32_t N = genomes.size();
    double distance_sum[N];
    double fitness_share[N];
    double fitness_share_total = 0;
    double sum_square = 0;
    double distance [N][N];
    for (int i = 0; i < N; i++) {
        distance_sum[i] = 0;
        for (int j = 0; j < N; j++) {
            if (i < j) {
                distance[i][j] = get_distance(genomes[i], genomes[j]);
            }
            else if (i > j) {
                distance[i][j] = distance[j][i]; 
            }
            else {
                distance[i][j] = 0;
            }
            if (distance[i][j] > fitness_threshold) {
                distance_sum[i] += 0;
            } else {
                distance_sum[i] += 1;
            }
        }
        fitness_share[i] = (genomes[i] -> get_fitness()) / distance_sum[i];
        fitness_share_total += fitness_share[i];
        sum_square += fitness_share[i] * fitness_share[i];
    }
    double fitness_share_mean = fitness_share_total / N;
    double fitness_share_std = sqrt( sum_square / (N-1) );
    double upper_cut_off = fitness_share_mean + fitness_share_std * 3;

    int32_t i = 0;
    auto it = genomes.begin();
	while (it != genomes.end()) {
		if (fitness_share[i] > upper_cut_off) {
			it = genomes.erase(it);
		}
		else {
			++it;
		}
        i++;
	}
    
}

void Species::erase_species() {
    genomes.clear();
    if(genomes.size() != 0){
        Log::error("The worst island is not fully erased!\n");
    }
}

int32_t Species::get_species_not_improving_count() {
    return species_not_improving_count;
}

void Species::set_species_not_improving_count(int32_t count) {
    species_not_improving_count = count;
}