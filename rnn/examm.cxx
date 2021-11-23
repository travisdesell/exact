#include <algorithm>
using std::sort;

#include <chrono>
#include <cstring>

#include <functional>
using std::bind;
using std::function;

#include <fstream>
using std::ofstream;

#include <iomanip>
using std::setprecision;
using std::setw;

#include <iostream>
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <string>
using std::string;
using std::to_string;

#include "examm.hxx"
#include "generate_nn.hxx"
#include "island_speciation_strategy.hxx"
#include "neat_speciation_strategy.hxx"
#include "rnn_genome.hxx"
#include "speciation_strategy.hxx"

// INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING
#include "delta_node.hxx"
#include "gru_node.hxx"
#include "lstm_node.hxx"
#include "mgu_node.hxx"
#include "rnn.hxx"
#include "rnn_node.hxx"
#include "ugrnn_node.hxx"

#include "common/files.hxx"
#include "common/log.hxx"

EXAMM::~EXAMM() {}

EXAMM::EXAMM(int32_t population_size, int32_t number_islands,
             int32_t max_genomes, int32_t _max_time_minutes, int32_t extinction_event_generation_number,
             int32_t islands_to_exterminate, string island_ranking_method,
             string repopulation_method, int32_t repopulation_mutations,
             bool repeat_extinction, int32_t epochs_acc_freq,
             string speciation_method, double species_threshold,
             double fitness_threshold, double neat_c1, double neat_c2,
             double neat_c3, WeightType weight_initialize,
             WeightType weight_inheritance, WeightType mutated_component_weight,
             string output_directory, GenomeOperators _genome_operators,
             DatasetMeta dataset_meta, TrainingParameters training_parameters,
             RNN_Genome *_seed_genome, bool start_filled)
    : population_size(population_size), number_islands(number_islands),
      max_genomes(max_genomes), dataset_meta(dataset_meta),
      max_time_minutes(_max_time_minutes),
      training_parameters(training_parameters),
      genome_operators(_genome_operators),
      extinction_event_generation_number(extinction_event_generation_number),
      start_filled(start_filled), seed_genome(_seed_genome),
      island_ranking_method(island_ranking_method),
      repopulation_method(repopulation_method),
      repopulation_mutations(repopulation_mutations),
      repeat_extinction(repeat_extinction), epochs_acc_freq(epochs_acc_freq),
      //                         use_regression(use_regression),
      output_directory(output_directory), weight_initialize(weight_initialize),
      weight_inheritance(weight_inheritance),
      mutated_component_weight(mutated_component_weight) {
  total_bp_epochs = 0;

  uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator = minstd_rand0(seed);
  rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

  // rng_crossover_weight = uniform_real_distribution<double>(0.0, 0.0);
  // rng_crossover_weight = uniform_real_distribution<double>(-0.10, 0.1);
  rng_crossover_weight = uniform_real_distribution<double>(-0.5, 1.5);
  // rng_crossover_weight = uniform_real_distribution<double>(0.45, 0.55);

  check_weight_initialize_validity();

  Log::info("weight initialize: %s\n",
            WEIGHT_TYPES_STRING[weight_initialize].c_str());
  Log::info("weight inheritance: %s \n",
            WEIGHT_TYPES_STRING[weight_inheritance].c_str());
  Log::info("mutated component weight: %s\n",
            WEIGHT_TYPES_STRING[mutated_component_weight].c_str());

  Log::info("Speciation method is: \"%s\" (Default is the island-based "
            "speciation strategy).\n",
            speciation_method.c_str());
  Log::info("Repeat extinction is set to %s\n",
            repeat_extinction ? "true" : "false");
  bool seed_genome_was_minimal = seed_genome == NULL;
  if (seed_genome_was_minimal) {
    seed_genome = create_ff(dataset_meta.input_parameter_names, 0, 0,
                            dataset_meta.output_parameter_names, 0,
                            training_parameters, weight_initialize,
                            weight_inheritance, mutated_component_weight);
    seed_genome->initialize_randomly();
  }

  seed_genome->set_generated_by("initial");

  // insert a copy of it into the population so
  // additional requests can mutate it

  seed_genome->best_validation_mse = EXAMM_MAX_DOUBLE;
  seed_genome->best_validation_mae = EXAMM_MAX_DOUBLE;

  Log::info("Speciation method is: \"%s\" (Default is the island-based "
            "speciation strategy).\n",
            speciation_method.c_str());
  if (speciation_method.compare("island") == 0 ||
      speciation_method.compare("") == 0) {
    double mutation_rate = 0.70, intra_island_co_rate = 0.20,
           inter_island_co_rate = 0.10;

    if (number_islands == 1) {
      intra_island_co_rate += inter_island_co_rate;
      inter_island_co_rate = 0.0;
    }
    if (population_size == 1) {
        mutation_rate += intra_island_co_rate;
        intra_island_co_rate = 0.0;
    }

    // Only difference here is that the apply_stir_mutations lambda is passed if
    // the island is supposed to start filled.
    if (start_filled) {
      // Only used if start_filled is enabled
      function<void(RNN_Genome *)> apply_stir_mutations =
          [&](RNN_Genome *genome) {
            RNN_Genome *copy = genome->copy();
            genome_operators.mutate(copy, repopulation_mutations);
            return copy;
          };

      speciation_strategy = new IslandSpeciationStrategy(
          number_islands, population_size, mutation_rate, intra_island_co_rate,
          inter_island_co_rate, seed_genome, island_ranking_method,
          repopulation_method, extinction_event_generation_number,
          repopulation_mutations, islands_to_exterminate,
          seed_genome_was_minimal, apply_stir_mutations, genome_operators);
    } else {
      speciation_strategy = new IslandSpeciationStrategy(
          number_islands, population_size, mutation_rate, intra_island_co_rate,
          inter_island_co_rate, seed_genome, island_ranking_method,
          repopulation_method, extinction_event_generation_number,
          repopulation_mutations, islands_to_exterminate, max_genomes,
          repeat_extinction, seed_genome_was_minimal, genome_operators);
    }
  } else if (speciation_method.compare("neat") == 0) {
    // make sure we don't duplicate node or edge innovation numbers
    double mutation_rate = 0.70, intra_island_co_rate = 0.20,
           inter_island_co_rate = 0.10;

    if (number_islands == 1) {
      inter_island_co_rate = 0.0;
      intra_island_co_rate = 0.30;
    }
    // no transfer learning for NEAT
    speciation_strategy = new NeatSpeciationStrategy(
        mutation_rate, intra_island_co_rate, inter_island_co_rate, seed_genome,
        species_threshold, fitness_threshold, neat_c1, neat_c2, neat_c3,
        generator);
  }

  if (output_directory != "") {
    mkpath(output_directory.c_str(), 0777);
    log_file = new ofstream(output_directory + "/" + "fitness_log.csv");
    (*log_file)
        << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE, Best Val. "
           "MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges, Fitness";
    // memory_log << "Inserted Genomes, Total BP Epochs, Time, Best Val. MAE,
    // Best Val. MSE, Enabled Nodes, Enabled Edges, Enabled Rec. Edges";

    (*log_file) << speciation_strategy->get_strategy_information_headers();
    //(memory_log) << speciation_strategy->get_strategy_information_headers();

    (*log_file) << endl;
    // memory_log << endl;

    op_log_file = new ofstream(output_directory + "/op_log.csv");

    op_log_ordering = {
        "genomes",     "crossover",          "island_crossover", "clone",
        "add_edge",    "add_recurrent_edge", "enable_edge",      "disable_edge",
        "enable_node", "disable_node",
    };

    // To get data about these ops without respect to node type,
    // you'll have to calculate the sum, e.g. sum split_node(x) for all node
    // types x to get information about split_node as a whole.
    vector<string> ops_with_node_type = {"add_node", "split_node", "merge_node",
                                         "split_edge"};

    for (uint32_t i = 0; i < ops_with_node_type.size(); i++) {
      string op = ops_with_node_type[i];
      for (uint32_t j = 0;
           j < genome_operators.get_possible_node_types().size(); j++)
        op_log_ordering.push_back(
            op + "(" +
            NODE_TYPES[genome_operators.get_possible_node_types()[j]] + ")");
    }

    for (uint32_t i = 0; i < op_log_ordering.size(); i++) {
      string op = op_log_ordering[i];
      (*op_log_file) << op;
      (*op_log_file) << " Generated, ";
      (*op_log_file) << op;
      (*op_log_file) << " Inserted, ";

      inserted_counts[op] = 0;
      generated_counts[op] = 0;
    }

    map<string, int>::iterator it;

    (*op_log_file) << endl;

  } else {
    log_file = NULL;
    op_log_file = NULL;
  }

  start_clock = std::chrono::system_clock::now();
}

void EXAMM::print() {
  if (Log::at_level(Log::INFO)) {
    speciation_strategy->print();
  }
}

void EXAMM::update_log() {
  if (log_file != NULL) {

    // make sure the log file is still good
    if (!log_file->good()) {
      log_file->close();
      delete log_file;

      string output_file = output_directory + "/fitness_log.csv";
      log_file = new ofstream(output_file, std::ios_base::app);

      if (!log_file->is_open()) {
        Log::error("could not open EXAMM output log: '%s'\n",
                   output_file.c_str());
        exit(1);
      }
    }

    if (!op_log_file->good()) {
      op_log_file->close();
      delete op_log_file;

      string output_file = output_directory + "/op_log.csv";
      op_log_file = new ofstream(output_file, std::ios_base::app);

      if (!op_log_file->is_open()) {
        Log::error("could not open EXAMM output log: '%s'\n",
                   output_file.c_str());
        exit(1);
      }
    }

    RNN_Genome *best_genome = get_best_genome();
    if (best_genome == NULL) {
      best_genome = speciation_strategy->get_global_best_genome();
    }

    std::chrono::time_point<std::chrono::system_clock> currentClock =
        std::chrono::system_clock::now();
    long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                            currentClock - start_clock)
                            .count();

    (*log_file) << speciation_strategy->get_generated_genomes() << ","
                << total_bp_epochs << "," << milliseconds << ","
                << best_genome->best_validation_mae << ","
                << best_genome->best_validation_mse << ","
                << best_genome->get_enabled_node_count() << ","
                << best_genome->get_enabled_edge_count() << ","
                << best_genome->get_enabled_recurrent_edge_count()
                << speciation_strategy->get_strategy_information_values() << ","
                << best_genome->get_fitness() << ","
                << endl;

    /*
    memory_log << speciation_strategy->get_evaluated_genomes()
        << "," << total_bp_epochs
        << "," << milliseconds
        << "," << best_genome->best_validation_mae
        << "," << best_genome->best_validation_mse
        << "," << best_genome->get_enabled_node_count()
        << "," << best_genome->get_enabled_edge_count()
        << "," << best_genome->get_enabled_recurrent_edge_count()
        << speciation_strategy->get_strategy_information_values()
        << endl;
    */

    for (uint32_t i = 0; i < op_log_ordering.size(); i++) {
      string op = op_log_ordering[i];
      (*op_log_file) << generated_counts[op] << ", " << inserted_counts[op]
                     << ", ";
    }

    (*op_log_file) << endl;
  }
}

void EXAMM::write_memory_log(string filename) {
  ofstream log_file(filename);
  log_file << memory_log.str();
  log_file.close();
}

string EXAMM::get_output_directory() const { return output_directory; }

double EXAMM::get_best_fitness() {
  return speciation_strategy->get_best_fitness();
}

double EXAMM::get_worst_fitness() {
  return speciation_strategy->get_worst_fitness();
}

RNN_Genome *EXAMM::get_best_genome() {
  return speciation_strategy->get_best_genome();
}

RNN_Genome *EXAMM::get_worst_genome() {
  return speciation_strategy->get_worst_genome();
}

// this will insert a COPY, original needs to be deleted
bool EXAMM::insert_genome(RNN_Genome *genome) {
  total_bp_epochs += genome->get_bp_iterations();

  // Log::info("genomes evaluated: %10d , attempting to insert: %s\n",
  // (speciation_strategy->get_evaluated_genomes() + 1),
  // parse_fitness(genome->get_fitness()).c_str());

  if (!genome->sanity_check()) {
    Log::error("genome failed sanity check on insert!\n");
    exit(1);
  }

  // updates EXAMM's mapping of which genomes have been generated by what
  genome->update_generation_map(generated_from_map);

  int32_t insert_position = speciation_strategy->insert_genome(genome);
  // write this genome to disk if it was a new best found genome
  if (insert_position == 0) {
    genome->normalize_type = dataset_meta.normalize_type;
    genome->write_graphviz(output_directory + "/rnn_genome_" +
                           to_string(genome->get_generation_id()) + ".gv");
    genome->write_to_file(output_directory + "/rnn_genome_" +
                          to_string(genome->get_generation_id()) + ".bin");
  }

  // Name of the operator
  const map<string, int> *generated_by_map = genome->get_generated_by_map();

  for (auto it = generated_by_map->begin(); it != generated_by_map->end();
       it++) {
    string generated_by = it->first;

    if (generated_counts.count(generated_by) > 0) {
      generated_counts["nenomes"] += 1;

      // Add one to the number of genomes generated by this operator
      generated_counts[generated_by] += 1;

      // If it was inserted add one to the number of genomes generated AND
      // inserted by this operator
      if (insert_position >= 0) {
        inserted_counts["genomes"] += 1;
        inserted_counts[generated_by] += 1;
      }
    } else {
      if (generated_by != "initial")
        Log::error("unrecognized generated_by string '%s'\n",
                   generated_by.c_str());
    }
  }

  // This is the culprit right here...
  // speciation_strategy->print();
  update_log();

  return insert_position >= 0;
}

Work *EXAMM::get_initialize_work() {
  Log::info("seed_genome %p\n", seed_genome);
  int max_node_innovation_number =
      seed_genome->get_max_node_innovation_number();
  int max_edge_innovation_number =
      seed_genome->get_max_edge_innovation_number();
  return new InitializeWork(max_node_innovation_number + 1,
                            max_edge_innovation_number + 1);
}

bool EXAMM::time_limit_reached() {
    if (max_time_minutes < 0)
        return false;
    std::chrono::time_point<std::chrono::system_clock> now = chrono::system_clock::now();
    long minutes_elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - start_clock).count();
    return minutes_elapsed >= max_time_minutes;
}

Work *EXAMM::generate_work() {
  if (speciation_strategy->get_inserted_genomes() > max_genomes || time_limit_reached())
    return new TerminateWork();
  return speciation_strategy->generate_work(rng_0_1, generator);
}

RNN_Genome *EXAMM::get_seed_genome() { return seed_genome; }

void EXAMM::check_weight_initialize_validity() {
  if (weight_initialize < 0) {
    Log::fatal(
        "Weight initalization is set to NONE, this should not happen! \n");
    exit(1);
  }
  if (weight_inheritance < 0) {
    Log::fatal("Weight inheritance is set to NONE, this should not happen! \n");
    exit(1);
  }
  if (mutated_component_weight < 0) {
    Log::fatal(
        "Mutated component weight is set to NONE, this should not happen! \n");
    exit(1);
  }
  if (weight_initialize == WeightType::LAMARCKIAN) {
    Log::fatal("Weight initialization method is set to Lamarckian! \n");
    exit(1);
  }
  if (weight_inheritance != weight_initialize &&
      weight_inheritance != WeightType::LAMARCKIAN) {
    Log::fatal("Weight initialize is %s, weight inheritance is %s\n",
               WEIGHT_TYPES_STRING[weight_initialize].c_str(),
               WEIGHT_TYPES_STRING[weight_inheritance].c_str());
    exit(1);
  }
  if (mutated_component_weight != weight_initialize &&
      mutated_component_weight != WeightType::LAMARCKIAN) {
    Log::fatal("Weight initialize is %s, new component weight is %s\n",
               WEIGHT_TYPES_STRING[weight_initialize].c_str(),
               WEIGHT_TYPES_STRING[mutated_component_weight].c_str());
    exit(1);
  }
}
