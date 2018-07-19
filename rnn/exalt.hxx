#ifndef EXALT_HXX
#define EXALT_HXX

#include <fstream>
using std::ofstream;

#include <map>
using std::map;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include "rnn_genome.hxx"

class EXALT {
    private:
        int32_t population_size;
        vector<RNN_Genome*> genomes;

        int32_t max_genomes;
        int32_t generated_genomes;
        int32_t inserted_genomes;

        int32_t edge_innovation_count;
        int32_t node_innovation_count;

        map<string, int32_t> inserted_from_map;
        map<string, int32_t> generated_from_map;

        int32_t number_inputs;
        int32_t number_outputs;
        int32_t bp_iterations;
        double learning_rate;

        bool use_high_threshold;
        double high_threshold;

        bool use_low_threshold;
        double low_threshold;

        bool use_dropout;
        double dropout_probability;

        minstd_rand0 generator;
        uniform_real_distribution<double> rng_0_1;
        uniform_real_distribution<double> rng_crossover_weight;

        int32_t max_recurrent_depth;

        bool epigenetic_weights;
        double crossover_rate;
        double more_fit_crossover_rate;
        double less_fit_crossover_rate;

        double lstm_node_rate;

        double clone_rate;

        double add_edge_rate;
        double add_recurrent_edge_rate;
        double enable_edge_rate;
        double disable_edge_rate;
        double split_edge_rate;

        double add_node_rate;
        double enable_node_rate;
        double disable_node_rate;
        double split_node_rate;
        double merge_node_rate;

        string log_filename;
        ofstream *log_file;

        vector<string> input_parameter_names;
        vector<string> output_parameter_names;


    public:
        EXALT(int32_t _population_size, int32_t _max_genomes, int32_t _number_inputs, int32_t _number_outputs, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names, int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold, double _low_threshold, bool _use_dropout, double _dropout_probability, string _log_filename);

        void print_population();
        int32_t population_contains(RNN_Genome* genome);

        bool insert_genome(RNN_Genome* genome);

        void initialize_genome_parameters(RNN_Genome* genome);
        RNN_Genome* generate_genome();
        void mutate(RNN_Genome *p1);

        void attempt_node_insert(vector<RNN_Node_Interface*> &child_nodes, const RNN_Node_Interface *node, const vector<double> &new_weights);
        void attempt_edge_insert(vector<RNN_Edge*> &child_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Edge *edge, RNN_Edge *second_edge, bool set_enabled);
        void attempt_recurrent_edge_insert(vector<RNN_Recurrent_Edge*> &child_recurrent_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Recurrent_Edge *recurrent_edge, RNN_Recurrent_Edge *second_edge, bool set_enabled);
        RNN_Genome* crossover(RNN_Genome *p1, RNN_Genome *p2);

        double get_best_fitness() const;
        double get_worst_fitness() const;

        RNN_Genome* get_best_genome();
};

#endif
