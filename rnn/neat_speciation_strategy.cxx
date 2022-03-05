#include <functional>
using std::function;

#include <chrono>

//#include <iostream>

#include <algorithm>
#include <algorithm>
using std::min;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <stdlib.h>

#include "common/log.hxx"
#include "examm.hxx"
#include "neat_speciation_strategy.hxx"
#include "rnn_genome.hxx"

/**
 *
 */
NeatSpeciationStrategy::NeatSpeciationStrategy(shared_ptr<const RNN_Genome> _seed_genome, double _species_threshold,
                                               double _fitness_threshold, double _neat_c1, double _neat_c2,
                                               double _neat_c3,
                                               GenomeOperators genome_operators)
    : generation_species(0),
      species_count(0),
      population_not_improving_count(0),
      species_threshold(_species_threshold),
      fitness_threshold(_fitness_threshold),
      neat_c1(_neat_c1),
      neat_c2(_neat_c2),
      neat_c3(_neat_c3),
      generated_genomes(0),
      inserted_genomes(0),
      minimal_genome(move(_seed_genome)),
      genome_operators(genome_operators) {
  Log::info("Neat speciation strategy, the species threshold is %f. \n", species_threshold);

  generated_genomes++;
  // set the fitst species with minimal genome
  neat_species.push_back(Species(species_count));
  Log::info("initialized the first species, current neat species size: %d \n", neat_species.size());
  species_count++;
  insert_genome(unique_ptr<RNN_Genome>(minimal_genome->copy()));

  global_best_genome = NULL;
}

shared_ptr<const RNN_Genome> &NeatSpeciationStrategy::get_best_genome() { return global_best_genome; }

shared_ptr<const RNN_Genome> &NeatSpeciationStrategy::get_worst_genome() {
  int32_t worst_genome_species = -1;
  double worst_fitness = -EXAMM_MAX_DOUBLE;

  for (int32_t i = 0; i < (int32_t) neat_species.size(); i++) {
    if (neat_species[i].size() > 0) {
      double island_worst_fitness = neat_species[i].get_worst_fitness();
      if (island_worst_fitness > worst_fitness) {
        worst_fitness = island_worst_fitness;
        worst_genome_species = i;
      }
    }
  }

  if (worst_genome_species < 0) {
    return minimal_genome;
  } else {
    return neat_species[worst_genome_species].get_worst_genome();
  }
}

double NeatSpeciationStrategy::get_best_fitness() { return get_best_genome()->get_fitness(); }

double NeatSpeciationStrategy::get_worst_fitness() { return get_worst_genome()->get_fitness(); }

// this will insert a COPY, original needs to be deleted
// returns 0 if a new global best, < 0 if not inserted, > 0 otherwise
pair<int32_t, const RNN_Genome *> NeatSpeciationStrategy::insert_genome(unique_ptr<RNN_Genome> genome) {
  bool inserted = false;
  bool erased_population = check_population();
  if (!erased_population) {
    check_species();
  } else {
    population_not_improving_count = 0;
  }
  vector<double> best = genome->get_best_parameters();

  if (best.size() != 0) { genome->set_weights(best); }

  Log::info("inserting genome id %d!\n", genome->get_generation_id());
  inserted_genomes++;

  int32_t insert_position;
  const RNN_Genome *g = nullptr;
  if (neat_species.size() == 1 && neat_species[0].size() == 0) {
    // insert the first genome in the evolution
    shared_ptr<const RNN_Genome> shared_genome = move(genome);
    g = shared_genome.get();
    auto [ip, ge] = neat_species[0].insert_genome(move(shared_genome));
    insert_position = ip;
    g = ge;
    Log::info("first genome of this species inserted \n");
    inserted = true;
  } else {
    vector<uint32_t> species_list = get_random_species_list();
    for (uint32_t i = 0; i < species_list.size(); i++) {
      Species &random_species = neat_species[species_list[i]];
      if (random_species.size() == 0) {
        Log::error("random_species is empty\n");
        continue;
      }

      const RNN_Genome *representative_genome = random_species.get_representative_genome();
      if (representative_genome == nullptr) {
        Log::error("species representative is null!\n");
        break;
      }

      double distance = get_distance(representative_genome, genome.get());

      // Log::error("distance is %f \n", distance);

      if (distance < species_threshold) {
        Log::info("inserting genome to species: %d\n", species_list[i]);
        shared_ptr<const RNN_Genome> shared_genome = move(genome);
        auto [ip, ge] = random_species.insert_genome(move(shared_genome));
        insert_position = ip;
        g = ge;
        inserted = true;
        break;
      }
    }
  }

  if (!inserted) {
    Species new_species = Species(species_count);
    species_count++;
    neat_species.push_back(new_species);
    if (species_count != neat_species.size()) {
      Log::error(
          "this should never happen, the species count is not the same "
          "as the number of species we have! \n");
      Log::error("num of species: %d, and species count is %d \n", neat_species.size(), species_count);
    }
    shared_ptr<const RNN_Genome> shared_genome = move(genome);
    g = shared_genome.get();
    auto [ip, ge] = neat_species.back().insert_genome(move(shared_genome));
    insert_position = ip;
    g = ge;

    inserted = true;
  }

  if (insert_position == 0) {
    // check and see if the inserted genome has the same fitness as the best
    // fitness of all islands
    double best_fitness = get_best_fitness();
    if (genome->get_fitness() == best_fitness) {
      population_not_improving_count = 0;
      return pair(0, g);
    } else {
      population_not_improving_count++;
      return pair(0, g);
    }  // was the best for the island but not the global best
  } else {
    population_not_improving_count++;
    return pair(insert_position, g);  // will be -1 if not inserted, or > 0 if
                                      // not the global best
  }
}

unique_ptr<WorkMsg> NeatSpeciationStrategy::generate_work() {
  // generate the genome from the next island in a round
  // robin fashion.
  unique_ptr<WorkMsg> work;
  // generate the genome from the next island in a round
  // robin fashion.
  if (generation_species >= neat_species.size()) generation_species = 0;
  Log::debug("getting species: %d\n", generation_species);

  Species &current_species = neat_species[generation_species];

  function<double(const RNN_Genome *, const RNN_Genome *)> distance_function =
      [=](const RNN_Genome *g1, const RNN_Genome *g2) { return this->get_distance(g1, g2); };

  Log::info(
      "generating new genome for species[%d], species_size: %d, "
      "mutation_rate: %lf, intra_island_crossover_rate: %lf, "
      "inter_island_crossover_rate: %lf\n",
      generation_species, current_species.size(), mutation_rate, intra_island_crossover_rate,
      inter_island_crossover_rate);

  if (current_species.size() < 2) {
    Log::info("current species has less than 2 genomes, doing mutation!\n");

    shared_ptr<const RNN_Genome> genome = current_species.get_random_genome(generator);

    work = make_unique<WorkMsg>(move(genome), genome_operators.get_random_n_mutations());
  } else {
    // first eliminate genomes who have low fitness sharing in this species
    if (current_species.size() > 10) current_species.fitness_sharing_remove(fitness_threshold, distance_function);

    // generate a genome via crossover or mutation
    Log::info("current species size %d, doing mutaion or crossover\n", current_species.size());

    work = generate_work_for_species(current_species);
  }

  work->set_genome_number(generated_genomes++);
  work->set_group_id(generation_species++);

  // Round robin reset / loop back to zero
  if (generation_species >= neat_species.size()) generation_species = 0;

  return work;
}

unique_ptr<WorkMsg> NeatSpeciationStrategy::generate_work_for_species(Species &species) {
  double r = rng_0_1(generator);

  if (r < GenomeOperators::mutation_p) {
    Log::info("performing mutation\n");

    shared_ptr<const RNN_Genome> genome = species.get_random_genome(generator);

    return make_unique<WorkMsg>(move(genome), genome_operators.get_random_n_mutations());
  } else {
    r = rng_0_1(generator);
    if (r < GenomeOperators::intra_co_p || neat_species.size() == 1) {
      // intra-island crossover
      Log::info("performing intra-species crossover\n");

      // select two distinct parent genomes in the same island
      int32_t n_parents = min(species.size(), genome_operators.get_random_n_parents_intra());
      vector<shared_ptr<const RNN_Genome>> parents(n_parents);
      species.get_n_random_genomes(generator, n_parents, parents);
      return make_unique<WorkMsg>(parents);
    } else {
      // inter-island crossover
      Log::info("performing inter-species crossover\n");

      // get a random genome from this island
      shared_ptr<const RNN_Genome> parent = species.get_random_genome(generator);

      vector<shared_ptr<const RNN_Genome>> parents;
      parents.push_back(move(parent));

      // select a different island randomly
      uint32_t other_island = rng_0_1(generator) * (neat_species.size() - 1);
      if (species.get_id() == neat_species[other_island].get_id()) other_island++;

      Species &other = neat_species[other_island];
      int32_t n = min(other.size(), genome_operators.get_random_n_parents_inter() - 1);

      other.get_n_random_genomes(generator, n, parents);

      return make_unique<WorkMsg>(parents);
    }
  }
}

void NeatSpeciationStrategy::print(string indent) const {
  Log::info("NEAT Species: \n");
  for (int32_t i = 0; i < (int32_t) neat_species.size(); i++) {
    Log::info("%sSpecies %d:\n", indent.c_str(), i);
    neat_species[i].print(indent + "\t");
  }
}

/**
 * Gets speciation strategy information headers for logs
 */
string NeatSpeciationStrategy::get_strategy_information_headers() const {
  string info_header = "";
  for (int32_t i = 0; i < (int32_t) neat_species.size(); i++) {
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
  string info_value = "";
  for (int32_t i = 0; i < (int32_t) neat_species.size(); i++) {
    double best_fitness = neat_species[i].get_best_fitness();
    // double worst_fitness = neat_species[i].get_worst_fitness();
    info_value.append(",");
    info_value.append(to_string(best_fitness));
    // info_value.append(",");
    // info_value.append(to_string(worst_fitness));
  }
  return info_value;
}

shared_ptr<const RNN_Genome> &NeatSpeciationStrategy::get_global_best_genome() { return global_best_genome; }

vector<uint32_t> NeatSpeciationStrategy::get_random_species_list() {
  vector<uint32_t> species_list;
  for (uint32_t i = 0; i < neat_species.size(); i++) species_list.push_back(i);

  shuffle(species_list.begin(), species_list.end(), generator);
  return species_list;
}

double NeatSpeciationStrategy::get_distance(const RNN_Genome *g1, const RNN_Genome *g2) {
  double distance;
  int E;
  int D;
  int32_t N;
  // d = c1*E/N + c2*D/N + c3*w
  vector<edge_inon> innovation1 = g1->get_edge_inons();
  vector<edge_inon> innovation2 = g2->get_edge_inons();
  double weight1 = g1->get_avg_edge_weight();
  double weight2 = g2->get_avg_edge_weight();
  double w = abs(weight1 - weight2);
  Log::debug("weight difference: %f \n", w);
  if (innovation1.size() >= innovation2.size()) {
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

  std::vector<edge_inon> setunion;
  std::vector<edge_inon> intersec;
  std::set_union(innovation1.begin(), innovation1.end(), innovation2.begin(), innovation2.end(),
                 std::inserter(setunion, setunion.begin()));
  std::set_intersection(innovation1.begin(), innovation1.end(), innovation2.begin(), innovation2.end(),
                        std::inserter(intersec, intersec.begin()));

  D = setunion.size() - intersec.size() - E;
  distance = neat_c1 * E / N + neat_c2 * D / N + neat_c3 * w;
  Log::debug("distance is %f \n", distance);
  return distance;
}
// v1.max > v2.max
int NeatSpeciationStrategy::get_exceed_number(vector<edge_inon> v1, vector<edge_inon> v2) {
  int exceed = 0;

  for (auto it = v1.rbegin(); it != v1.rend(); ++it) {
    if (*it > v2.back()) {
      exceed++;
    } else {
      break;
    }
  }
  return exceed;
}

void NeatSpeciationStrategy::rank_species() {
  double fitness_j1, fitness_j2;

  auto by = [&](Species &a, Species &b) -> bool { return a.get_best_fitness() > b.get_best_fitness(); };

  sort(neat_species.begin(), neat_species.end(), by);

  for (uint32_t i = 0; i < neat_species.size() - 1; i++) {
    Log::error("Neat specis rank: %f \n", neat_species[i].get_best_fitness());
  }
}

bool NeatSpeciationStrategy::check_population() {
  bool erased = false;
  // check if the population fitness is not improving for 3000 genomes,
  // if so only save the top 2 species and erase the rest

  if (population_not_improving_count >= 3000) {
    Log::error(
        "the population fitness has not been improved for 3000 genomes, "
        "start to erasing \n");
    rank_species();

    neat_species.erase(neat_species.begin(), neat_species.end() - 2);
    if (neat_species.size() != 2) {
      Log::error(
          "It should never happen, the population has %d number of "
          "species instead of 2! \n",
          neat_species.size());
    }
    for (int i = 0; i < 2; i++) {
      Log::error("species %d size %d\n", i, neat_species[i].size());
      Log::error("species %d fitness %f\n", i, neat_species[i].get_best_fitness());
      neat_species[i].set_species_not_improving_count(0);
    }
    Log::error("erase finished!\n");
    Log::error("current number of species: %d \n", neat_species.size());
    erased = true;
  }
  return erased;
}

void NeatSpeciationStrategy::check_species() {
  Log::info("checking speies \n");
  auto it = neat_species.begin();

  while (it != neat_species.end()) {
    if (neat_species[it - neat_species.begin()].get_species_not_improving_count() >= 2250) {
      Log::error(
          "Species at position %d hasn't been improving for 2250 "
          "genomes, erasing it \n",
          it - neat_species.begin());
      Log::error("current number of species: %d \n", neat_species.size());
      // neat_species[neat_species.begin() - it]->erase_species();
      it = neat_species.erase(it);
    } else {
      ++it;
    }
  }
  Log::info("finished checking species, current number of species: %d \n", neat_species.size());
}
