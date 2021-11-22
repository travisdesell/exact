#ifndef GENOME_OPERATORS_HXX
#define GENOME_OPERATORS_HXX

#include <random>
#include <utility>
#include <vector>
using std::pair;

#include "common/dataset_meta.hxx"
#include "common/log.hxx"
#include "common/weight_initialize.hxx"
#include "rnn_genome.hxx"
#include "rnn_node_interface.hxx"
#include "training_parameters.hxx"
using namespace std;

class GenomeOperators {
   private:
    // Genome rank is how well it is doing, relative to some group of genomes.
    // 0 is the best, 1 second best, etc.
    // It usually corresponds to the index of the genome, as long as the genomes are sorted.
    typedef uint32_t GenomeRank;

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
        clone_rate + add_edge_rate + add_recurrent_edge_rate + enable_edge_rate + disable_edge_rate + split_edge_rate +
        add_node_rate + enable_node_rate + disable_node_rate + split_node_rate + merge_node_rate;

    // Instance variables
    const DatasetMeta dataset_meta;
    vector<int> possible_node_types;

    int32_t number_workers;
    int32_t worker_id;

    int32_t number_inputs;
    int32_t number_outputs;

    // Number of genomes used for crossover within an island. Max value is the population size of an island
    int32_t n_parents_intra;

    // Number of genomes used for crossover between different islands. Max value is 1 + population size of an island, 1
    // from local island and population size genomes from foreign island.
    int32_t n_parents_inter;

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

    void attempt_node_insert(vector<RNN_Node_Interface *> &child_nodes, const RNN_Node_Interface *node,
                             const vector<double> &new_weights);
    void attempt_edge_insert(vector<RNN_Edge *> &child_edges, vector<RNN_Node_Interface *> &child_nodes, RNN_Edge *edge,
                             RNN_Edge *second_edge, bool set_enabled);
    void attempt_recurrent_edge_insert(vector<RNN_Recurrent_Edge *> &child_recurrent_edges,
                                       vector<RNN_Node_Interface *> &child_nodes, RNN_Recurrent_Edge *recurrent_edge,
                                       RNN_Recurrent_Edge *second_edge, bool set_enabled);

    // Perform the "reflect" step of simplex to obtain a new weight
    double simplex_weight_crossover_1d(vector<pair<GenomeRank, double> > &weights);
    // Similar, but works on x in R^N instead of just R
    double simplex_weight_crossover_2d(vector<pair<GenomeRank, vector<double> > > &weights);

   public:
    const TrainingParameters training_parameters;
    int32_t edge_innovation_count;
    int32_t node_innovation_count;
    function<int32_t()> next_edge_innovation_number;
    function<int32_t()> next_node_innovation_number;

    GenomeOperators(int32_t _number_workers, int32_t _worker_id, int32_t _number_inputs, int32_t _number_outputs,
                    int32_t n_parents_inra, int32_t n_parents_inter, int32_t _edge_innovation_count,
                    int32_t _node_innovation_count, int32_t _min_recurrent_depth, int32_t _max_recurrent_depth,
                    WeightType _weight_initialize, WeightType _weight_inheritance, WeightType _mutated_component_weight,
                    DatasetMeta _dataset_meta, TrainingParameters _training_parameters,
                    vector<string> possible_node_types);

    RNN_Genome *mutate(RNN_Genome *g, int32_t n_mutations);
    RNN_Genome *crossover(RNN_Genome *more_fit, RNN_Genome *less_fit);
    RNN_Genome *ncrossover(vector<RNN_Genome *> &parents);

    void finalize_genome(RNN_Genome *g);
    void set_edge_innovation_count(int32_t);
    void set_node_innovation_count(int32_t);
    const vector<int> &get_possible_node_types();

    int32_t get_n_parents_intra();
    int32_t get_n_parents_inter();

    int get_number_inputs();
    int get_number_outputs();
};

#endif
