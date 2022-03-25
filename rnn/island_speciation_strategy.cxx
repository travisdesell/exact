#include <functional>
using std::function;

#include <chrono>

//#include <iostream>

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <optional>
using std::nullopt;

#include <string>
using std::string;

#include "common/log.hxx"
#include "examm.hxx"
#include "island_speciation_strategy.hxx"
#include "rnn_genome.hxx"

ArgumentSet IslandSpeciationStrategy::arguments = ArgumentSet(
    "island_args",
    {
        new Argument("n_islands", "--n_islands", "number of islands", false, Argument::INT, 8),

        new Argument("island_size", "--island_size", "maximum number of genomes per island", false, Argument::INT, 8),

        new Argument("start_filled", "--start-filled",
                     "Whether to start the algorithm with islands full of random genomes.", false, Argument::BOOL,
                     false),

        new Argument("extinction_event_generation_number", "--extinction-event-generation-number",
                     "When to perform an island extinction.", false, Argument::INT, INT_MAX),

        new Argument("repeat_extinction", "--repeat-extinction",
                     "Whether or not to repeat the extinction event every $extinction_event_generation_number genomes.",
                     false, Argument::BOOL, false),
    });

IslandSpeciationStrategy::IslandSpeciationStrategy(
    uint32_t _number_of_islands, uint32_t _max_island_size, shared_ptr<const RNN_Genome> _seed_genome,
    string _island_ranking_method, string _repopulation_method, uint32_t _extinction_event_generation_number,
    uint32_t _repopulation_mutations, uint32_t _islands_to_exterminate, bool _seed_genome_was_minimal,
    optional<function<void(RNN_Genome *)>> modify, GenomeOperators &genome_operators)
    : generation_island(0),
      number_of_islands(_number_of_islands),
      max_island_size(_max_island_size),
      seed_genome(move(_seed_genome)),
      island_ranking_method(_island_ranking_method),
      extinction_event_generation_number(_extinction_event_generation_number),
      repopulation_mutations(_repopulation_mutations),
      islands_to_exterminate(_islands_to_exterminate),
      seed_genome_was_minimal(_seed_genome_was_minimal),
      genome_operators(genome_operators) {
  double rate_sum =
      GenomeOperators::mutation_p + (GenomeOperators::intra_co_p + GenomeOperators::inter_co_p) * GenomeOperators::co_p;

  mutation_rate = GenomeOperators::mutation_p / rate_sum;
  intra_island_crossover_rate = GenomeOperators::intra_co_p * GenomeOperators::co_p / rate_sum;
  inter_island_crossover_rate = GenomeOperators::inter_co_p * GenomeOperators::co_p / rate_sum;

  if (number_of_islands == 1 && max_island_size == 1) {
    mutation_rate = 1.0;
    intra_island_crossover_rate = 0.0;
    inter_island_crossover_rate = 0.0;
  } else if (number_of_islands == 1) {
    intra_island_crossover_rate += inter_island_crossover_rate;
  } else if (max_island_size == 1) {
    inter_island_crossover_rate += intra_island_crossover_rate;
  }

  intra_island_crossover_rate += mutation_rate;
  inter_island_crossover_rate += intra_island_crossover_rate;

  if (modify == nullopt) {
    for (int i = 0; i < (int) number_of_islands; i += 1) { islands.push_back(Island(i, max_island_size)); }
  } else {
    auto make_filled_island = [](int32_t id, shared_ptr<const RNN_Genome> seed_genome, int32_t size,
                                 function<void(RNN_Genome *)> &modify) {
      vector<shared_ptr<const RNN_Genome>> genomes;
      genomes.reserve(size);
      for (int i = 0; i < size; i += 1) {
        RNN_Genome *clone = seed_genome->copy();
        modify(clone);
        clone->set_generation_id(0);
        genomes.push_back(shared_ptr<RNN_Genome>(clone));
      }

      return Island(id, genomes);
    };
    for (uint32_t i = 0; i < number_of_islands; i += 1)
      islands.push_back(make_filled_island(i, _seed_genome, max_island_size, *modify));
  }
  set_repopulation_method(_repopulation_method);

  generated_genomes++;
  global_best_genome = seed_genome;
}

shared_ptr<const RNN_Genome> &IslandSpeciationStrategy::get_best_genome() {
  // the global_best_genome is updated every time a genome is inserted
  return global_best_genome;
}

shared_ptr<const RNN_Genome> &IslandSpeciationStrategy::get_worst_genome() {
  int32_t worst_genome_island = -1;
  double worst_fitness = -EXAMM_MAX_DOUBLE;

  for (int32_t i = 0; i < (int32_t) islands.size(); i++) {
    if (islands[i].size() > 0) {
      double island_worst_fitness = islands[i].get_worst_fitness();
      if (island_worst_fitness > worst_fitness) {
        worst_fitness = island_worst_fitness;
        worst_genome_island = i;
      }
    }
  }

  if (worst_genome_island < 0) {
    return seed_genome;
  } else {
    return islands[worst_genome_island].get_worst_genome();
  }
}

double IslandSpeciationStrategy::get_best_fitness() { return global_best_genome->get_fitness(); }

double IslandSpeciationStrategy::get_worst_fitness() { return get_worst_genome()->get_fitness(); }

void IslandSpeciationStrategy::set_repopulation_method(string repopulation_method_str) {
  std::transform(repopulation_method_str.begin(), repopulation_method_str.end(), repopulation_method_str.begin(),
                 ::tolower);

  static const map<string, RepopulationMethod> str2repop{
      {"bestparents",   RepopulationMethod::BEST_PARENTS  },
      {"randomparents", RepopulationMethod::RANDOM_PARENTS},
      {"bestgenome",    RepopulationMethod::BEST_GENOME   },
      {"bestisland",    RepopulationMethod::BEST_ISLAND   },
      {"",              RepopulationMethod::NONE          }
  };

  auto it = str2repop.find(repopulation_method_str);
  if (it != str2repop.end())
    repopulation_method = it->second;
  else {
    Log::fatal("Invalid repopulation method specified.");
    exit(0);
  }
}

bool IslandSpeciationStrategy::islands_full() const {
  return generated_genomes > (int) (islands.size() * max_island_size * 2);
}

// returns 0 if a new global best, < 0 if not inserted, > 0 otherwise
pair<int32_t, const RNN_Genome *> IslandSpeciationStrategy::insert_genome(unique_ptr<RNN_Genome> genome_unique) {
  Log::debug("inserting genome!\n");
  if (extinction_event_generation_number != 0) {
    if (generated_genomes > 1 && generated_genomes % extinction_event_generation_number == 0 &&
        max_genomes - evaluated_genomes >= extinction_event_generation_number) {
      if (island_ranking_method.compare("EraseWorst") == 0 || island_ranking_method.compare("") == 0) {
        vector<int32_t> rank = rank_islands();
        for (uint32_t i = 0; i < islands_to_exterminate; i++) {
          if (rank[i] >= 0) {
            Log::info("found island: %d is the worst island \n", rank[0]);
            islands[rank[i]].erase_island();
            islands[rank[i]].erase_structure_map();
            islands[rank[i]].set_status(IslandStatus::REPOPULATING);
          } else
            Log::error("Didn't find the worst island!");
          // set this so the island would not be re-killed in 5 rounds
          if (!repeat_extinction) { set_erased_islands_status(); }
        }
      }
    }
  }

  double fitness = genome_unique->get_fitness();
  int32_t island = genome_unique->get_group_id();
  shared_ptr<const RNN_Genome> genome_shared = std::move(genome_unique);

  bool new_global_best = global_best_genome->get_fitness() > fitness;
  if (new_global_best) global_best_genome = genome_shared;

  auto [insert_position, g] = islands[island].insert_genome(move(genome_shared));

  evaluated_genomes++;
  
  if (insert_position >= 0) {
    inserted_genomes++;
    return pair(insert_position, g);
  } else {
    return pair(-1, nullptr);
  }
}

int32_t IslandSpeciationStrategy::get_worst_island_by_best_genome() {
  int32_t worst_island = -1;
  double worst_best_fitness = 0;
  for (int32_t i = 0; i < (int32_t) islands.size(); i++) {
    if (islands[i].size() > 0) {
      if (islands[i].get_erase_again_num() > 0) continue;
      double island_best_fitness = islands[i].get_best_fitness();
      if (island_best_fitness > worst_best_fitness) {
        worst_best_fitness = island_best_fitness;
        worst_island = i;
      }
    }
  }
  return worst_island;
}

vector<int32_t> IslandSpeciationStrategy::rank_islands() {
  vector<int32_t> island_rank;
  int32_t temp;
  double fitness_j1, fitness_j2;
  Log::info("ranking islands \n");
  Log::info("repeat extinction: %s \n", repeat_extinction ? "true" : "false");
  for (uint32_t i = 0; i < number_of_islands; i++) {
    if (repeat_extinction) {
      island_rank.push_back(i);
    } else {
      if (islands[i].get_erase_again_num() == 0) { island_rank.push_back(i); }
    }
  }

  for (uint32_t i = 0; i < island_rank.size() - 1; i++) {
    for (uint32_t j = 0; j < island_rank.size() - i - 1; j++) {
      fitness_j1 = islands[island_rank[j]].get_best_fitness();
      fitness_j2 = islands[island_rank[j + 1]].get_best_fitness();
      if (fitness_j1 < fitness_j2) {
        temp = island_rank[j];
        island_rank[j] = island_rank[j + 1];
        island_rank[j + 1] = temp;
      }
    }
  }
  Log::info("island rank: \n");
  for (uint32_t i = 0; i < island_rank.size(); i++) {
    Log::info("island: %d fitness %f \n", island_rank[i], islands[island_rank[i]].get_best_fitness());
  }
  return island_rank;
}

unique_ptr<WorkMsg> IslandSpeciationStrategy::generate_work() {
  // generate the genome from the next island in a round robin fashion.
  unique_ptr<WorkMsg> work;

  Log::info("getting island: %d\n", generation_island);
  Island &island = islands[generation_island];

  Log::debug("islands.size(): %d\n", islands.size());

  if (island.is_initializing()) {
    // Generate via mutation only + use the seed genome for empty islands
    work = generate_work_for_initializing_island(island);
  } else if (island.is_full()) {
    // Mutation or crossover
    work = generate_work_for_filled_island(island);
  } else if (island.is_repopulating()) {
    // select two other islands (non-overlapping) at random, and select genomes
    // from within those islands and generate a child via crossover
    work = generate_work_for_reinitializing_island(island);
  } else {
    Log::fatal("ERROR: island was neither initializing, repopulating or full.\n");
    Log::fatal("This should never happen!\n");
    exit(1);
  }

  work->set_genome_number(++generated_genomes);
  work->set_group_id(generation_island++);

  if (generation_island >= (int) number_of_islands) generation_island = 0;

  if (work == nullptr) {
    Log::fatal("ERROR: genome was NULL at the end of generate genome!\n");
    Log::fatal("This should never happen.\n");
    exit(1);
  }

  return work;
}

unique_ptr<WorkMsg> IslandSpeciationStrategy::generate_work_for_initializing_island(Island &island) {
  Log::info("island is initializing!\n");
  shared_ptr<const RNN_Genome> genome;
  int32_t n_mutations;

  if (island.size() == 0) {
    Log::debug("starting with minimal genome\n");
    genome = seed_genome;
    n_mutations = 0;
  } else {
    Log::info("island is not empty, mutating a random genome\n");
    genome = island.get_random_genome(generator);
    n_mutations = genome_operators.get_random_n_mutations();
  }

  return make_unique<WorkMsg>(move(genome), n_mutations);
}

unique_ptr<WorkMsg> IslandSpeciationStrategy::generate_work_for_reinitializing_island(Island &island) {
  Log::info("island is repopulating through %s method!\n", repopulation_method_str.c_str());

  switch (repopulation_method) {
    case RepopulationMethod::RANDOM_PARENTS:
      return parents_repopulation(RepopulationMethod::RANDOM_PARENTS);
    case RepopulationMethod::BEST_PARENTS:
      return parents_repopulation(RepopulationMethod::BEST_PARENTS);
    case RepopulationMethod::BEST_GENOME:
      return make_unique<WorkMsg>(get_global_best_genome(), repopulation_mutations);
    case RepopulationMethod::BEST_ISLAND: {
      // copy the best island to the worst at once
      // after the worst island is filled, set the island status to filled
      // then generate a genome for filled status, so this function still return
      // a generated genome
      uint32_t best_island_id = get_best_genome()->get_group_id();
      copy_island(best_island_id, island.get_id());

      if (island.is_full()) {
        Log::info("island is full now, and generating a new one!\n");
        island.set_status(IslandStatus::FILLED);
        return generate_work_for_filled_island(island);
      } else {
        Log::error("Island is not full after coping the best island over!\n");
        island.set_status(IslandStatus::INITIALIZING);
        return generate_work_for_initializing_island(island);
      }
    }
    case RepopulationMethod::NONE:
      Log::fatal(
          "repopulating an island even though the selected repopulation "
          "method is NONE");
      exit(0);
    default:
      Log::fatal("This should be unreachable\n");
      exit(0);
  }
}

unique_ptr<WorkMsg> IslandSpeciationStrategy::generate_work_for_filled_island(Island &island) {
  // if we haven't filled ALL of the island populations yet, only use mutation
  // otherwise do mutation at %, crossover at %, and island crossover at %
  double r = rng_0_1(generator);
  if (!islands_full() || r < mutation_rate) {
    Log::info("performing mutation\n");
    shared_ptr<const RNN_Genome> genome = island.get_random_genome(generator);
    return make_unique<WorkMsg>(move(genome), genome_operators.get_random_n_mutations());
  } else {
    vector<shared_ptr<const RNN_Genome>> parents;

    if (r < intra_island_crossover_rate || number_of_islands == 1) {
      Log::info("performing intra-island crossover\n");
      island.get_n_random_genomes(generator, genome_operators.get_random_n_parents_intra(), parents);
    } else {
      Log::info("performing inter-island crossover\n");
      int32_t n = genome_operators.get_random_n_parents_inter();
      parents.push_back(island.get_random_genome(generator));
      int32_t other_island = rng_0_1(generator) * (number_of_islands - 1);
      if (other_island >= generation_island) other_island += 1;
      islands[other_island].get_n_random_genomes(generator, n - 1, parents);
    }
    return make_unique<WorkMsg>(parents);
  }
}

void IslandSpeciationStrategy::print(string indent) const {
  Log::info("%sIslands: \n", indent.c_str());
  for (int32_t i = 0; i < (int32_t) islands.size(); i++) {
    Log::info("%sIsland %d:\n", indent.c_str(), i);
    islands[i].print(indent + "\t");
  }
}

/**
 * Gets speciation strategy information headers for logs
 */
string IslandSpeciationStrategy::get_strategy_information_headers() const {
  string info_header = "";
  for (int32_t i = 0; i < (int32_t) islands.size(); i++) {
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
string IslandSpeciationStrategy::get_strategy_information_values() const {
  string info_value = "";
  for (int32_t i = 0; i < (int32_t) islands.size(); i++) {
    double best_fitness = islands[i].get_best_fitness();
    double worst_fitness = islands[i].get_worst_fitness();
    info_value.append(",");
    info_value.append(to_string(best_fitness));
    info_value.append(",");
    info_value.append(to_string(worst_fitness));
  }
  return info_value;
}

unique_ptr<WorkMsg> IslandSpeciationStrategy::parents_repopulation(RepopulationMethod method) {
  Log::info("generation island: %d \n", generation_island);
  int32_t parent_island1 = (number_of_islands - 1) * rng_0_1(generator);
  if (parent_island1 >= generation_island) parent_island1 += 1;

  Log::info("parent island 1: %d \n", parent_island1);

  int32_t parent_island2;
  do {
    parent_island2 = (number_of_islands - 1) * rng_0_1(generator);
  } while (parent_island2 == generation_island || parent_island2 == parent_island1);

  Log::info("parent island 2: %d \n", parent_island2);

  vector<shared_ptr<const RNN_Genome>> parents;

  switch (method) {
    case RepopulationMethod::RANDOM_PARENTS:
      islands[parent_island1].get_n_random_genomes(generator, 1, parents);
      islands[parent_island2].get_n_random_genomes(generator, 1, parents);
      break;

    case RepopulationMethod::BEST_PARENTS:
      parents.push_back(islands[parent_island1].get_best_genome());
      parents.push_back(islands[parent_island2].get_best_genome());
      break;

    default:
      Log::fatal("Invalid me supplied to IslandSpeciationStrategy::method");
      exit(0);
  }

  Log::info("current island is %d, the parent1 island is %d, parent 2 island is %d\n", generation_island,
            parent_island1, parent_island2);

  // swap so the first parent is the more fit parent
  if (parents[0]->get_fitness() > parents[1]->get_fitness()) std::swap(parents[0], parents[1]);

  return make_unique<WorkMsg>(parents);
}

void IslandSpeciationStrategy::copy_island(uint32_t src_island, uint32_t dst_island) {
  vector<shared_ptr<const RNN_Genome>> &src_genomes = islands[src_island].get_genomes();
  for (uint32_t i = 0; i < src_genomes.size(); i++) {
    RNN_Genome *copy = src_genomes[i]->copy();
    generated_genomes++;
    copy->set_generation_id(generated_genomes);
    islands[generation_island].set_latest_generation_id(generated_genomes);
    copy->set_group_id(dst_island);
    if (repopulation_mutations > 0) {
      Log::info(
          "Doing %d mutations to genome %d before inserted to the "
          "repopulating island\n",
          repopulation_mutations, copy->generation_id);
      genome_operators.mutate(copy, repopulation_mutations);
    }
    insert_genome(unique_ptr<RNN_Genome>(copy));
  }
}

shared_ptr<const RNN_Genome> &IslandSpeciationStrategy::get_global_best_genome() { return global_best_genome; }

void IslandSpeciationStrategy::set_erased_islands_status() {
  for (uint32_t i = 0; i < islands.size(); i++) {
    if (islands[i].get_erase_again_num() > 0) {
      islands[i].set_erase_again_num();
      Log::info("Island %d can be removed in %d rounds.\n", i, islands[i].get_erase_again_num());
    }
  }
}
