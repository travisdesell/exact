#ifndef EXALT_HXX
#define EXALT_HXX

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

#include "rnn_genome.hxx"
#include "edge_pheromone.hxx"
#include "node_pheromone.hxx"
#include "rnn_genome.hxx"
#include "generate_nn.hxx"

#define ANTS 50
#define PHEROMONES_THERESHOLD 20
#define PERIODIC_PHEROMONE_REDUCTION_RATIO 0.75

class ACNNTO {
    private:
        int32_t population_size;
        int32_t number_islands;
        vector<RNN_Genome*> population;

        map <int32_t, NODE_Pheromones*> colony;

        int32_t max_genomes;
        int32_t generated_genomes;
        int32_t inserted_genomes;
        int32_t total_bp_epochs;

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
        double mutation_rate;
        double crossover_rate;
        double island_crossover_rate;

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

        vector<int> possible_node_types;

        string output_directory;
        ofstream *log_file;

        vector<string> input_parameter_names;
        vector<string> output_parameter_names;

        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;


        ostringstream memory_log;

        std::chrono::time_point<std::chrono::system_clock> startClock;

        void prepare_new_genome(RNN_Genome* genome);
        RNN_Node* check_node_existance(vector<RNN_Node_Interface*> &rnn_nodes,   EDGE_Pheromone* pheromone_line);
        void check_edge_existance(vector<RNN_Edge*> &rnn_edges, int32_t innovation_number, RNN_Node* in_node, RNN_Node* out_node);
        void check_recurrent_edge_existance(vector<RNN_Recurrent_Edge*> &recurrent_edges, int32_t innovation_number, int32_t depth, RNN_Node* in_node, RNN_Node* out_node);

        EDGE_Pheromone* pick_line(vector<EDGE_Pheromone*>* edges_pheromones);
        int pick_node_type(double* type_pheromones);
        void reward_colony(RNN_Genome* g, double treat);
        void reduce_pheromones();



    public:
        ACNNTO(int32_t _population_size, int32_t _max_genomes, const vector<string> &_input_parameter_names,
          const vector<string> &_output_parameter_names, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs,
          int32_t _bp_iterations, double _learning_rate, bool _use_high_threshold, double _high_threshold, bool _use_low_threshold,
          double _low_threshold, string _output_directory);

        ~ACNNTO();

        void print_population();
        void write_memory_log(string filename);

        void set_possible_node_types(vector<string> possible_node_type_strings);

        int32_t population_contains(RNN_Genome* genome);
        bool populations_full() const;

        bool insert_genome(RNN_Genome* genome);


        int get_random_node_type();

        void initialize_genome_parameters(RNN_Genome* genome);


        RNN_Genome* get_best_genome();
        RNN_Genome* get_worst_genome();

        double get_best_fitness();
        double get_worst_fitness();

        string get_output_directory() const;

        RNN_Genome* ants_march();
};

#endif
