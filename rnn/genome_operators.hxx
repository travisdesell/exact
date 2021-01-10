#ifndef MUTATOR_HXX
#define MUTATOR_HXX

using namespace std;

#include <vector>
#include <random>

#include "rnn_genome.hxx"
#include "rnn_node_interface.hxx"

#include "common/log.hxx"

class GenomeOperators {
    private:
        // Crossover hyperparameters
        static constexpr double more_fit_crossover_rate = 1.00;
        static constexpr double less_fit_crossover_rate = 0.50;


        // Mutation hyperparameters
        static constexpr double clone_rate          = 1.0;

        static constexpr double add_edge_rate       = 1.0;
        static constexpr double add_recurrent_edge_rate = 1.0;
        static constexpr double enable_edge_rate    = 1.0;
        static constexpr double disable_edge_rate   = 1.0;
        static constexpr double split_edge_rate     = 0.0;

#define NODE_OPS 1
#ifdef NODE_OPS
        static constexpr double add_node_rate      = 1.0;
        static constexpr double enable_node_rate   = 1.0;
        static constexpr double disable_node_rate  = 1.0;
        static constexpr double split_node_rate    = 1.0;
        static constexpr double merge_node_rate    = 1.0;
#else
        static constexpr double add_node_rate      = 0.0;
        static constexpr double enable_node_rate   = 0.0;
        static constexpr double disable_node_rate  = 0.0;
        static constexpr double split_node_rate    = 0.0;
        static constexpr double merge_node_rate    = 0.0;
#endif

        static constexpr double mutation_rates_total = clone_rate + add_edge_rate + add_recurrent_edge_rate + enable_edge_rate + disable_edge_rate + split_edge_rate + add_node_rate + enable_node_rate + disable_node_rate + split_node_rate + merge_node_rate;


        // Instance variables
        const TrainingParameters _training_parameters;
        const DatasetMeta _dataset_meta,
        vector<int> possible_node_types;

        int32_t number_workers;
        int32_t worker_id;

        int32_t number_inputs;
        int32_t number_output;

        int32_t edge_innovation_count;
        int32_t node_innovation_count;

        minstd_rand0 generator{(unsigned long) time(0)};
        uniform_int_distribution<int> node_index_dist;
        uniform_real_distribution<double> unit_dist{0.0, 1.0};
        uniform_int_distribution<int32_t> recurrent_depth_dist;

        int32_t get_next_node_innovation_number();
        int32_t get_next_edge_innovation_number();
        void set_possible_node_types(vector<string> &node_types);
        int get_random_node_type();
        void finalize_genome(RNN_Genome *g);

    public:
        GenomeOperators(
                int32_t _number_workers,
                int32_t _worker_id,
                int32_t _number_inputs,
                int32_t _number_outputs,
                int32_t _edge_innovation_count,
                int32_t _node_innovation_count,
                int32_t _min_recurrent_depth,
                int32_t _max_recurrent_depth,
                double _dropout_probability,
                DatasetMeta _dataset_meta,
                TrainingParameters _training_parameters,
                vector<string> possible_node_types);

        RNN_Genome *mutate(RNN_Genome *g, int32_t n_mutations);
        RNN_Genome *crossover(RNN_Genome *more_fit, RNN_Genome *less_fit);
};

#endif
