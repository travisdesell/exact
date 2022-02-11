#ifndef EXAMM_HXX
#define EXAMM_HXX

#include <fstream>
using std::ofstream;

#include <map>
using std::map;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include "common/weight_initialize.hxx"
#include "genome_operators.hxx"
#include "rnn_genome.hxx"
#include "speciation_strategy.hxx"
#include "training_parameters.hxx"

#include "msg.hxx"

#include "common/dataset_meta.hxx"

class EXAMM {
private:
  int32_t population_size;
  int32_t number_islands;

  int32_t max_genomes;
  int32_t max_time_minutes;
  int32_t total_bp_epochs;

  int32_t genome_id = 0;

  DatasetMeta dataset_meta;
  TrainingParameters training_parameters;
  GenomeOperators genome_operators;

  int32_t extinction_event_generation_number;

  bool start_filled;

  RNN_Genome *seed_genome;

  string island_ranking_method;
  // string speciation_method;
  string repopulation_method;
  int32_t repopulation_mutations;
  bool repeat_extinction;
  int32_t epochs_acc_freq;

  SpeciationStrategy *speciation_strategy;

  map<string, int32_t> inserted_from_map;
  map<string, int32_t> generated_from_map;

  minstd_rand0 generator;
  uniform_real_distribution<double> rng_0_1{0.0, 1.0};
  uniform_real_distribution<double> rng_crossover_weight;

  vector<string> op_log_ordering;
  map<string, int32_t> inserted_counts;
  map<string, int32_t> generated_counts;

  string output_directory;
  ofstream *log_file;
  ofstream *op_log_file;

  WeightType weight_initialize;
  WeightType weight_inheritance;
  WeightType mutated_component_weight;

  ostringstream memory_log;

  std::chrono::time_point<std::chrono::system_clock> start_clock;

  string genome_file_name;

  bool time_limit_reached();

public:
  EXAMM(int32_t _population_size, int32_t _number_islands, int32_t _max_genomes, int32_t _max_time_minutes,
        int32_t _extinction_event_generation_number,
        int32_t _islands_to_exterminate, string _island_ranking_method,
        string repopulation_method, int32_t _repopulation_mutations,
        bool _repeat_extinction, int32_t _epochs_acc_freq,
        string speciation_method, double _species_threshold,
        double _fitness_threshold, double _neat_c1, double _neat_c2,
        double _neat_c3, WeightType _weight_initialize,
        WeightType _weight_inheritance, WeightType _mutated_component_weight,
        string _output_directory, GenomeOperators _genome_operators,
        DatasetMeta _dataset_meta, TrainingParameters _training_parameters,
        // int32_t _bp_iterations,
        // double _learning_rate,
        // bool _use_high_threshold,
        // double _high_threshold,
        // bool _use_low_threshold,
        // double _low_threshold,
        // bool _use_dropout,
        // double _dropout_probability,
        // int32_t _min_recurrent_depth,
        // int32_t _max_recurrent_depth,
        // bool _use_regression,
        // string _output_directory,
        RNN_Genome *seed_genome, bool _start_filled);

  ~EXAMM();

  void print();
  void update_log();
  void write_memory_log(string filename);

  uniform_int_distribution<int32_t> get_recurrent_depth_dist();

  Msg *get_initialize_work();
  Msg *generate_work();

  int get_random_node_type();

  bool insert_genome(RNN_Genome *genome);

  double get_best_fitness();
  double get_worst_fitness();
  RNN_Genome *get_best_genome();
  RNN_Genome *get_worst_genome();
  RNN_Genome *get_seed_genome();

  string get_output_directory() const;
  RNN_Genome *generate_for_transfer_learning(string file_name, int extra_inputs,
                                             int extra_outputs);

  void check_weight_initialize_validity();
};

#endif
