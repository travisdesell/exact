#ifndef GENOME_OPERATORS_HXX
#define GENOME_OPERATORS_HXX

#include <random>
#include <vector>

#include "common/dataset_meta.hxx"
#include "common/weight_initialize.hxx"
#include "rnn_genome.hxx"
#include "rnn_node_interface.hxx"
#include "training_parameters.hxx"

#include "common/log.hxx"
#include "common/weight_initialize.hxx"
using namespace std;

class GenomeOperators {
private:
  // Crossover hyperparameters
  static constexpr double more_fit_crossover_rate = 1.00;
  static constexpr double less_fit_crossover_rate = 0.50;

  // Mutation hyperparameters
  static constexpr double clone_rate = 1.0;

  static constexpr double add_edge_rate = 1.0;
  static constexpr double add_recurrent_edge_rate = 1.0;
  static constexpr double enable_edge_rate = 1.0;
  static constexpr double disable_edge_rate = 1.0;
  static constexpr double split_edge_rate = 0.0;

#define NODE_OPS 1
#ifdef NODE_OPS
  static constexpr double add_node_rate = 1.0;
  static constexpr double enable_node_rate = 1.0;
  static constexpr double disable_node_rate = 1.0;
  static constexpr double split_node_rate = 1.0;
  static constexpr double merge_node_rate = 1.0;
#else
  static constexpr double add_node_rate = 0.0;
  static constexpr double enable_node_rate = 0.0;
  static constexpr double disable_node_rate = 0.0;
  static constexpr double split_node_rate = 0.0;
  static constexpr double merge_node_rate = 0.0;
#endif

  static constexpr double mutation_rates_total =
      clone_rate + add_edge_rate + add_recurrent_edge_rate + enable_edge_rate +
      disable_edge_rate + split_edge_rate + add_node_rate + enable_node_rate +
      disable_node_rate + split_node_rate + merge_node_rate;

  // Instance variables
  const DatasetMeta dataset_meta;
  vector<int> possible_node_types;

  int32_t number_workers;
  int32_t worker_id;

  int32_t number_inputs;
  int32_t number_outputs;

  WeightType weight_initialize;
  WeightType weight_inheritance;
  WeightType mutated_component_weight;

  minstd_rand0 generator;
  uniform_int_distribution<int> node_index_dist;
  uniform_real_distribution<double> rng_0_1{0.0, 1.0};
  uniform_real_distribution<double> rng_crossover_weight{-0.5, 1.5};
  uniform_int_distribution<int32_t> recurrent_depth_dist;

  int32_t get_next_node_innovation_number();
  int32_t get_next_edge_innovation_number();
  void set_possible_node_types(vector<string> &node_types);
  int get_random_node_type();

  void attempt_node_insert(vector<RNN_Node_Interface *> &child_nodes,
                           const RNN_Node_Interface *node,
                           const vector<double> &new_weights);
  void attempt_edge_insert(vector<RNN_Edge *> &child_edges,
                           vector<RNN_Node_Interface *> &child_nodes,
                           RNN_Edge *edge, RNN_Edge *second_edge,
                           bool set_enabled);
  void attempt_recurrent_edge_insert(
      vector<RNN_Recurrent_Edge *> &child_recurrent_edges,
      vector<RNN_Node_Interface *> &child_nodes,
      RNN_Recurrent_Edge *recurrent_edge, RNN_Recurrent_Edge *second_edge,
      bool set_enabled);

public:
  const TrainingParameters training_parameters;
  int32_t edge_innovation_count;
  int32_t node_innovation_count;
  function<int32_t()> next_edge_innovation_number;
  function<int32_t()> next_node_innovation_number;

  GenomeOperators(int32_t _number_workers, int32_t _worker_id,
                  int32_t _number_inputs, int32_t _number_outputs,
                  int32_t _edge_innovation_count,
                  int32_t _node_innovation_count, int32_t _min_recurrent_depth,
                  int32_t _max_recurrent_depth, WeightType _weight_initialize,
                  WeightType _weight_inheritance,
                  WeightType _mutated_component_weight,
                  DatasetMeta _dataset_meta,
                  TrainingParameters _training_parameters,
                  vector<string> possible_node_types);

  RNN_Genome *mutate(RNN_Genome *g, int32_t n_mutations);
  RNN_Genome *crossover(RNN_Genome *more_fit, RNN_Genome *less_fit);

  void finalize_genome(RNN_Genome *g);
  void set_edge_innovation_count(int32_t);
  void set_node_innovation_count(int32_t);
  const vector<int> &get_possible_node_types();

  int get_number_inputs();
  int get_number_outputs();
};

#endif
