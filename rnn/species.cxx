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

#include "common/log.hxx"
#include "rnn_genome.hxx"
#include "species.hxx"
// Species(int32_t id, double fitness_th);
Species::Species(int32_t _id) : id(_id), species_not_improving_count(0) {}

const shared_ptr<const RNN_Genome> &Species::get_best_genome() const { return genomes[0]; }

shared_ptr<const RNN_Genome> &Species::get_worst_genome() { return genomes.back(); }

shared_ptr<const RNN_Genome> Species::get_random_genome(uniform_real_distribution<double> &rng_0_1,
                                                        minstd_rand0 &generator) {
  int32_t genome_position = size() * rng_0_1(generator);
  return genomes[genome_position];
}

double Species::get_best_fitness() const {
  if (genomes.size())
    return get_best_genome()->get_fitness();
  else
    return EXAMM_MAX_DOUBLE;
}

double Species::get_worst_fitness() {
  if (genomes.size())
    return get_worst_genome()->get_fitness();
  else
    return EXAMM_MAX_DOUBLE;
}

int32_t Species::get_id() { return id; }

int32_t Species::size() { return genomes.size(); }

int32_t Species::contains(const RNN_Genome *genome) {
  for (int32_t j = 0; j < (int32_t) genomes.size(); j++) {
    if (genomes[j]->equals(genome)) { return j; }
  }

  return -1;
}

void Species::get_two_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator,
                                     shared_ptr<const RNN_Genome> &g1, shared_ptr<const RNN_Genome> &g2) {
  int32_t p1 = size() * rng_0_1(generator);
  int32_t p2 = (size() - 1) * rng_0_1(generator);
  if (p2 >= p1) p2++;

  // swap the gnomes so that the first parent is the more fit parent
  if (p1 > p2) { std::swap(p1, p2); }

  g1 = genomes[p1];
  g2 = genomes[p2];
}

void Species::get_n_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, int32_t n,
                                   vector<shared_ptr<const RNN_Genome>> &parents) {
  if (n > genomes.size()) {
    Log::fatal("Cannot give n parents with species size of %d\n", size());
    exit(1);
  }

  vector<int> indices(genomes.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Fischer-yates shuffle
  for (int i = indices.size() - 1; i > 0; i--) {
    int j = rng_0_1(generator) * indices.size();
    swap(indices[i], indices[j]);
  }

  indices.resize(n);
  for (int i = 0; i < indices.size(); i++) parents.push_back(genomes[i]);
}

// returns -1 for not inserted, otherwise the index it was inserted at
// inserts a copy of the genome, caller of the function will need to delete
// their pointer
pair<int32_t, const RNN_Genome *> Species::insert_genome(shared_ptr<const RNN_Genome> genome) {
  Log::info("inserting genome with fitness: %s to species %d\n", parse_fitness(genome->get_fitness()).c_str(), id);

  auto insert_it = upper_bound(genomes.begin(), genomes.end(), genome, sort_genomes_by_fitness());
  int32_t insert_index = insert_it - genomes.begin();

  if (insert_index == 0) {
    // this was a new best genome for this island
    Log::info("new best fitness for island: %d!\n", id);
    species_not_improving_count = 0;
  } else {
    species_not_improving_count++;
  }

  auto gid = genome->get_generation_id();
  inserted_genome_id.push_back(gid);

  genomes.emplace(insert_it, move(genome));
  Log::info("Inserted genome %d at index %d\n", gid, insert_index);

  return pair(insert_index, genomes[insert_index].get());
}

void Species::print(string indent) const {
  Log::info("%s\t%s\n", indent.c_str(), RNN_Genome::print_statistics_header().c_str());
  for (uint32_t i = 0; i < genomes.size(); i++) {
    Log::info("%s\t%s\n", indent.c_str(), genomes[i]->print_statistics().c_str());
  }
}

vector<shared_ptr<const RNN_Genome>> &Species::get_genomes() { return genomes; }

const RNN_Genome *Species::get_representative_genome() {
  const RNN_Genome *latest = nullptr;
  for (auto it = inserted_genome_id.rbegin(); it != inserted_genome_id.rend(); ++it) {
    int32_t latest_id = *it;
    for (uint32_t i = 0; i < genomes.size(); i++) {
      if (genomes[i]->get_generation_id() == latest_id) {
        latest = genomes[i].get();
        break;
      }
    }
    if (latest != nullptr) { break; }
  }
  return latest;
}

void Species::fitness_sharing_remove(double fitness_threshold,
                                     function<double(const RNN_Genome *, const RNN_Genome *)> &get_distance) {
  int32_t N = genomes.size();
  double distance_sum[N];
  double fitness_share[N];
  double fitness_share_total = 0;
  double sum_square = 0;
  double distance[N][N];
  for (int i = 0; i < N; i++) {
    distance_sum[i] = 0;
    for (int j = 0; j < N; j++) {
      if (i < j) {
        distance[i][j] = get_distance(genomes[i].get(), genomes[j].get());
      } else if (i > j) {
        distance[i][j] = distance[j][i];
      } else {
        distance[i][j] = 0;
      }
      if (distance[i][j] > fitness_threshold) {
        distance_sum[i] += 0;
      } else {
        distance_sum[i] += 1;
      }
    }
    fitness_share[i] = (genomes[i]->get_fitness()) / distance_sum[i];
    fitness_share_total += fitness_share[i];
    sum_square += fitness_share[i] * fitness_share[i];
  }
  double fitness_share_mean = fitness_share_total / N;
  double fitness_share_std = sqrt(sum_square / (N - 1));
  double upper_cut_off = fitness_share_mean + fitness_share_std * 3;

  int32_t i = 0;
  auto it = genomes.begin();
  while (it != genomes.end()) {
    if (fitness_share[i] > upper_cut_off) {
      it = genomes.erase(it);
    } else {
      ++it;
    }
    i++;
  }
}

void Species::erase_species() {
  genomes.clear();
  if (genomes.size() != 0) { Log::error("The worst island is not fully erased!\n"); }
}

int32_t Species::get_species_not_improving_count() { return species_not_improving_count; }

void Species::set_species_not_improving_count(int32_t count) { species_not_improving_count = count; }
