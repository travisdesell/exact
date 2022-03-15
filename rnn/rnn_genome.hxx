#ifndef RNN_BPTT_HXX
#define RNN_BPTT_HXX

#include <fstream>
#include <memory>
using std::ifstream;
using std::istream;
using std::ofstream;
using std::ostream;

#include <map>
using std::map;

#include <random>
using std::minstd_rand0;
using std::mt19937;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <vector>
using std::vector;
#include <functional>

#include "common/random.hxx"
#include "common/weight_initialize.hxx"
#include "rnn.hxx"
#include "rnn_edge.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_recurrent_edge.hxx"
#include "time_series/time_series.hxx"
#include "training_parameters.hxx"
#include "word_series/word_series.hxx"

// mysql can't handle the max float value for some reason
#define EXAMM_MAX_DOUBLE 10000000

string parse_fitness(double fitness);

class RNN_Genome {
 private:
  int32_t generation_id;
  int32_t group_id;

  TrainingParameters training_parameters;

  string structural_hash;

  string log_filename;

  WeightType weight_initialize;
  WeightType weight_inheritance;
  WeightType mutated_component_weight;

  map<string, int> generated_by_map;

  vector<double> initial_parameters;

  double best_validation_mse = EXAMM_MAX_DOUBLE;
  double best_validation_mae = EXAMM_MAX_DOUBLE;
  double training_mse;
  double training_mae;
  vector<double> best_parameters;

  minstd_rand0 generator;

  uniform_real_distribution<double> rng;
  uniform_real_distribution<double> rng_0_1;
  uniform_real_distribution<double> rng_1_1;
  NormalDistribution normal_distribution;

  vector<RNN_Node_Interface *> nodes;
  vector<RNN_Edge *> edges;
  vector<RNN_Recurrent_Edge *> recurrent_edges;

  vector<string> input_parameter_names;
  vector<string> output_parameter_names;

  string normalize_type;
  map<string, double> normalize_mins;
  map<string, double> normalize_maxs;
  map<string, double> normalize_avgs;
  map<string, double> normalize_std_devs;

 public:
  // Bit flags.
  enum transfer_learning_version { v1 = 1, v2 = 2, v3 = 4 };
  static const inline map<string, int> TRANSFER_LEARNING_MAP = {
      {"v1", v1},
      {"v2", v2},
      {"v3", v3}
  };

  bool tl_with_epigenetic;
  void sort_nodes_by_depth();
  void sort_edges_by_depth();
  void sort_recurrent_edges_by_depth();

  RNN_Genome(vector<RNN_Node_Interface *> &_nodes, vector<RNN_Edge *> &_edges,
             vector<RNN_Recurrent_Edge *> &_recurrent_edges, TrainingParameters training_parameters,
             WeightType _weight_initialize, WeightType _weight_inheritance, WeightType _mutated_component_weight);
  RNN_Genome(vector<RNN_Node_Interface *> &_nodes, vector<RNN_Edge *> &_edges,
             vector<RNN_Recurrent_Edge *> &_recurrent_edges, TrainingParameters training_parameters, uint16_t seed,
             WeightType _weight_initialize, WeightType _weight_inheritance, WeightType _mutated_component_weight);
  RNN_Genome(string binary_filename, bool _tl_with_epigenetic = false);
  RNN_Genome(char *array, uint32_t length);
  RNN_Genome(istream &bin_infile);

  ~RNN_Genome();

  void read_from_array(char *array, uint32_t length);
  void read_from_stream(istream &bin_istream);

  void write_to_array(char **array, uint32_t &length) const;
  void write_to_file(string bin_filename) const;
  void write_to_stream(ostream &bin_stream) const;

  RNN_Genome *copy() const;

  static string print_statistics_header();
  string print_statistics() const;

  void set_parameter_names(const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names);

  string generated_by_string() const;

  string get_edge_count_str(bool recurrent) const;
  string get_node_count_str(int node_type) const;

  const map<string, int> &get_generated_by_map() const;

  double get_avg_recurrent_depth() const;

  int32_t get_enabled_edge_count() const;
  int32_t get_enabled_recurrent_edge_count() const;
  int32_t get_enabled_node_count(int node_type) const;
  int32_t get_node_count(int node_type) const;
  int32_t get_enabled_node_count() const;
  int32_t get_node_count() const;

  void calculate_fitness(const vector<vector<vector<double>>> &training_inputs,
                         const vector<vector<vector<double>>> &training_outputs,
                         const vector<vector<vector<double>>> &validation_inputs,
                         const vector<vector<vector<double>>> &validation_outputs);
  double get_fitness() const;
  double get_best_validation_softmax() const;
  double get_best_validation_mse() const;
  double get_best_validation_mae() const;

  void set_normalize_bounds(string _normalize_type, const map<string, double> &_normalize_mins,
                            const map<string, double> &_normalize_maxs, const map<string, double> &_normalize_avgs,
                            const map<string, double> &_normalize_std_devs);

  string get_normalize_type() const;
  map<string, double> get_normalize_mins() const;
  map<string, double> get_normalize_maxs() const;
  map<string, double> get_normalize_avgs() const;
  map<string, double> get_normalize_std_devs() const;

  const vector<string> &get_input_parameter_names() const;
  const vector<string> &get_output_parameter_names() const;

  int32_t get_group_id() const;
  void set_group_id(int32_t _group_id);

  int32_t get_bp_iterations() const;

  void set_log_filename(string _log_filename);
  void get_weights(vector<double> &parameters) const;
  void set_weights(const vector<double> &parameters);

  uint32_t get_number_weights() const;
  uint32_t get_number_inputs() const;
  uint32_t get_number_outputs() const;

  double get_avg_edge_weight() const;
  void initialize_randomly();
  void initialize_xavier(RNN_Node_Interface *n);
  void initialize_kaiming(RNN_Node_Interface *n);
  void initialize_node_randomly(RNN_Node_Interface *n);
  double get_xavier_weight(RNN_Node_Interface *output_node);
  double get_kaiming_weight(RNN_Node_Interface *output_node);
  double get_random_weight();

  void get_input_edges(node_inon node_innovation, vector<RNN_Edge *> &input_edges,
                       vector<RNN_Recurrent_Edge *> &input_recurrent_edges) const;
  int32_t get_fan_in(node_inon node_innovation) const;
  int32_t get_fan_out(node_inon node_innovation) const;

  int32_t get_generation_id() const;
  void set_generation_id(int32_t generation_id);

  void clear_generated_by();
  void update_generation_map(map<string, int32_t> &generation_map);
  void set_generated_by(string type);
  int32_t get_generated_by(string type) const;

  RNN *get_rnn();
  vector<double> get_best_parameters() const;

  void set_best_parameters(vector<double> parameters);     // INFO: ADDED BY ABDELRAHMAN TO USE FOR
                                                           // TRANSFER LEARNING
  void set_initial_parameters(vector<double> parameters);  // INFO: ADDED BY ABDELRAHMAN TO USE FOR
                                                           // TRANSFER LEARNING

  void get_analytic_gradient(vector<RNN *> &rnns, const vector<double> &parameters,
                             const vector<vector<vector<double>>> &inputs,
                             const vector<vector<vector<double>>> &outputs, double &mse,
                             vector<double> &analytic_gradient, bool training);

  void backpropagate(const vector<vector<vector<double>>> &inputs, const vector<vector<vector<double>>> &outputs,
                     const vector<vector<vector<double>>> &validation_inputs,
                     const vector<vector<vector<double>>> &validation_outputs);

  void backpropagate_stochastic(const vector<vector<vector<double>>> &inputs,
                                const vector<vector<vector<double>>> &outputs,
                                const vector<vector<vector<double>>> &validation_inputs,
                                const vector<vector<vector<double>>> &validation_outputs);

  vector<vector<double>> slice_time_series(uint32_t start_index, uint32_t sequence_length, uint32_t num_parameter,
                                           const vector<vector<double>> &inputs);

  double get_softmax(const vector<double> &parameters, const vector<vector<vector<double>>> &inputs,
                     const vector<vector<vector<double>>> &outputs);
  double get_mse(const vector<double> &parameters, const vector<vector<vector<double>>> &inputs,
                 const vector<vector<vector<double>>> &outputs);
  double get_mae(const vector<double> &parameters, const vector<vector<vector<double>>> &inputs,
                 const vector<vector<vector<double>>> &outputs);

  vector<vector<double>> get_predictions(const vector<double> &parameters, const vector<vector<vector<double>>> &inputs,
                                         const vector<vector<vector<double>>> &outputs);
  void write_predictions(string output_directory, const vector<string> &input_filenames,
                         const vector<double> &parameters, const vector<vector<vector<double>>> &inputs,
                         const vector<vector<vector<double>>> &outputs, TimeSeriesSets *time_series_sets);
  void write_predictions(string output_directory, const vector<string> &input_filenames,
                         const vector<double> &parameters, const vector<vector<vector<double>>> &inputs,
                         const vector<vector<vector<double>>> &outputs, Corpus *word_series_sets);

  void get_mu_sigma(const vector<double> &p, double &mu, double &sigma) const;

  bool sanity_check() const;
  void assign_reachability();
  bool outputs_unreachable();

  RNN_Node_Interface *create_node(double mu, double sigma, int node_type, double depth);

  bool attempt_edge_insert(RNN_Node_Interface *n1, RNN_Node_Interface *n2, double mu, double sigma);
  bool attempt_recurrent_edge_insert(RNN_Node_Interface *n1, RNN_Node_Interface *n2, double mu, double sigma,
                                     uniform_int_distribution<int32_t> dist);

  // after adding an Elman or Jordan node, generate the circular RNN edge for
  // Elman and the edges from output to this node for Jordan.
  void generate_recurrent_edges(RNN_Node_Interface *node, double mu, double sigma,
                                uniform_int_distribution<int32_t> dist);

  bool add_edge(double mu, double sigma);
  bool add_recurrent_edge(double mu, double sigma, uniform_int_distribution<int32_t> rec_depth_dist);
  bool disable_edge();
  bool enable_edge();
  bool split_edge(double mu, double sigma, int node_type, uniform_int_distribution<int32_t> rec_depth_dist);

  bool add_node(double mu, double sigma, int node_type, uniform_int_distribution<int32_t> dist);

  bool enable_node();
  bool disable_node();
  bool split_node(double mu, double sigma, int node_type, uniform_int_distribution<int32_t> dist);
  bool merge_node(double mu, double sigma, int node_type, uniform_int_distribution<int32_t> dist);

  /**
   * Determines if the genome contains a node with the given innovation number
   *
   * @param the innovation number to fine
   *
   * @return true if the genome has a node with the provided innovation
   * number, false otherwise.
   */
  bool has_node_with_inon(node_inon inon) const;

  bool equals(const RNN_Genome *other) const;

  string get_color(double weight, bool is_recurrent) const;
  void write_graphviz(string filename) const;

  bool connect_new_input_node(double mu, double sig, RNN_Node_Interface *new_node,
                              uniform_int_distribution<int32_t> dist, bool not_all_hidden);
  bool connect_new_output_node(double mu, double sig, RNN_Node_Interface *new_node,
                               uniform_int_distribution<int32_t> dist, bool not_all_hidden);

  vector<RNN_Node_Interface *> pick_possible_nodes(int layer_type, bool not_all_hidden, string node_type);

  vector<edge_inon> get_edge_inons() const;
  /**
   * \return the structural hash (calculated when assign_reachaability is
   * called)
   */
  string get_structural_hash() const;

  /**
   * \return the max innovation number of any node in the genome.
   */
  node_inon get_max_node_inon() const;

  /**
   * \return the max innovation number of any edge or recurrent edge in the
   * genome.
   */
  edge_inon get_max_edge_inon() const;

  void transfer_to(const vector<string> &new_input_parameter_names, const vector<string> &new_output_parameter_names,
                   string transfer_learning_version, bool epigenetic_weights, int32_t min_recurrent_depth,
                   int32_t max_recurrent_depth);

  friend class EXAMM;
  friend class GenomeOperators;
  friend class IslandSpeciationStrategy;
  friend class NeatSpeciationStrategy;
  friend class RecDepthFrequencyTable;
};

struct sort_genomes_by_fitness {
  bool operator()(const RNN_Genome *const g1, const RNN_Genome *const g2) {
    return g1->get_fitness() < g2->get_fitness();
  }
  bool operator()(const shared_ptr<const RNN_Genome> &a, const shared_ptr<const RNN_Genome> &b) {
    return a->get_fitness() < b->get_fitness();
  }
};

void write_binary_string(ostream &out, string s, string name);
void read_binary_string(istream &in, string &s, string name);

#endif
