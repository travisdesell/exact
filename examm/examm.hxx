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

#include "rnn/rnn_genome.hxx"
#include "speciation_strategy.hxx"
#include "weights/weight_rules.hxx"
#include "time_series/time_series.hxx"
#include "rnn/genome_property.hxx"

class EXAMM {
    private:
        int32_t population_size;
        int32_t number_islands;

        vector< vector<RNN_Genome*> > genomes;

        int32_t max_genomes;
        int32_t total_bp_epochs;
        SpeciationStrategy *speciation_strategy;
        TimeSeriesSets *time_series_sets;
        WeightRules *weight_rules;
        GenomeProperty *genome_property;


        int32_t edge_innovation_count;
        int32_t node_innovation_count;

        map<string, int32_t> inserted_from_map;
        map<string, int32_t> generated_from_map;

        // int32_t number_inputs;
        // int32_t number_outputs;

        minstd_rand0 generator;
        uniform_real_distribution<double> rng_0_1;
        uniform_real_distribution<double> rng_crossover_weight;

        bool epigenetic_weights;

        double more_fit_crossover_rate;
        double less_fit_crossover_rate;

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

        vector<int32_t> possible_node_types;

        vector<string> op_log_ordering;
        map<string, int32_t> inserted_counts;
        map<string, int32_t> generated_counts;

        string output_directory;
        ofstream *log_file;
        ofstream *op_log_file;

        // vector<string> input_parameter_names;
        // vector<string> output_parameter_names;

        // string normalize_type;
        // map<string,double> normalize_mins;
        // map<string,double> normalize_maxs;
        // map<string,double> normalize_avgs;
        // map<string,double> normalize_std_devs;



        ostringstream memory_log;

        std::chrono::time_point<std::chrono::system_clock> startClock;

        string  genome_file_name;

        // bool start_filled;

    public:
        EXAMM(  int32_t _population_size,
                int32_t _number_islands,
                int32_t _max_genomes,
                SpeciationStrategy *_speciation_strategy,
                TimeSeriesSets *_time_series_sets,
                WeightRules *_weight_rules,
                GenomeProperty *_genome_property,
                string _output_directory,
                RNN_Genome *seed_genome);


        ~EXAMM();

        void print();
        void update_log();
        void write_memory_log(string filename);

        void set_possible_node_types(vector<string> possible_node_type_strings);

        uniform_int_distribution<int32_t> get_recurrent_depth_dist();

        int32_t get_random_node_type();

        RNN_Genome* generate_genome(int32_t seed_genome_stirs = 0);
        bool insert_genome(RNN_Genome* genome);

        void mutate(int32_t max_mutations, RNN_Genome *p1);

        void attempt_node_insert(vector<RNN_Node_Interface*> &child_nodes, const RNN_Node_Interface *node, const vector<double> &new_weights);
        void attempt_edge_insert(vector<RNN_Edge*> &child_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Edge *edge, RNN_Edge *second_edge, bool set_enabled);
        void attempt_recurrent_edge_insert(vector<RNN_Recurrent_Edge*> &child_recurrent_edges, vector<RNN_Node_Interface*> &child_nodes, RNN_Recurrent_Edge *recurrent_edge, RNN_Recurrent_Edge *second_edge, bool set_enabled);
        RNN_Genome* crossover(RNN_Genome *p1, RNN_Genome *p2);

        double get_best_fitness();
        double get_worst_fitness();
        RNN_Genome* get_best_genome();
        RNN_Genome* get_worst_genome();

        string get_output_directory() const;
        // RNN_Genome* generate_for_transfer_learning(string file_name, int32_t extra_inputs, int32_t extra_outputs) ;

        void check_weight_initialize_validity();

        // void get_time_series_parameters();
};

#endif
