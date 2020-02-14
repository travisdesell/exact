#ifndef RNN_BPTT_HXX
#define RNN_BPTT_HXX

#include <fstream>
using std::istream;
using std::ifstream;
using std::ostream;
using std::ofstream;

#include <map>
using std::map;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;
using std::mt19937;

#include <vector>
using std::vector;

#include "rnn.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_edge.hxx"
#include "rnn_recurrent_edge.hxx"

#include "common/random.hxx"

//mysql can't handl the max float value for some reason
#define EXAMM_MAX_DOUBLE 10000000

string parse_fitness(double fitness);

// This needs to be forward declared here
class Distribution;

class RNN_Genome {
    private:
        int32_t generation_id;
        int32_t group_id;

        int32_t bp_iterations;
        double learning_rate;
        bool adapt_learning_rate;
        bool use_nesterov_momentum;
        bool use_reset_weights;

        bool use_high_norm;
        double high_threshold;
        bool use_low_norm;
        double low_threshold;

        bool use_dropout;
        double dropout_probability;

        string log_filename;

        map<string, int> generated_by_map;

        vector<double> initial_parameters;

        double best_validation_mse;
        double best_validation_mae;
        vector<double> best_parameters;

        minstd_rand0 generator;
        uniform_real_distribution<double> rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);
        NormalDistribution normal_distribution;

        vector<RNN_Node_Interface*> nodes;
        vector<RNN_Edge*> edges;
        vector<RNN_Recurrent_Edge*> recurrent_edges;

        vector<string> input_parameter_names;
        vector<string> output_parameter_names;

        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;

    public:
        void sort_nodes_by_depth();
        void sort_edges_by_depth();

        RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges);
        RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges);
        RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges, uint16_t seed);


        RNN_Genome* copy();

        ~RNN_Genome();

        static string print_statistics_header();
        string print_statistics();

        void set_parameter_names(const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names);

        string generated_by_string();

        string get_edge_count_str(bool recurrent);
        string get_node_count_str(int node_type);

        double get_avg_recurrent_depth() const;

        int32_t get_enabled_edge_count();
        int32_t get_enabled_recurrent_edge_count();
        int32_t get_enabled_node_count(int node_type);
        int32_t get_node_count(int node_type);
        int32_t get_enabled_node_count();
        int32_t get_node_count();

        double get_fitness() const;
        double get_best_validation_mse() const;
        double get_best_validation_mae() const;


        void set_normalize_bounds(const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs);

        map<string,double> get_normalize_mins() const;
        map<string,double> get_normalize_maxs() const;

        vector<string> get_input_parameter_names() const;
        vector<string> get_output_parameter_names() const;

        int32_t get_group_id() const;
        void set_group_id(int32_t _group_id);


        void set_bp_iterations(int32_t _bp_iterations);
        int32_t get_bp_iterations();

        void set_learning_rate(double _learning_rate);
        void set_adapt_learning_rate(bool _adapt_learning_rate);
        void set_nesterov_momentum(bool _use_nesterov_momentum);
        void set_reset_weights(bool _use_reset_weights);
        void disable_high_threshold();
        void enable_high_threshold(double _high_threshold);
        void disable_low_threshold();
        void enable_low_threshold(double _low_threshold);
        void disable_dropout();
        void enable_dropout(double _dropout_probability);
        void set_log_filename(string _log_filename);

        void get_weights(vector<double> &parameters);
        void set_weights(const vector<double> &parameters);

        uint32_t get_number_weights();
        uint32_t get_number_inputs();
        uint32_t get_number_outputs();

        void initialize_randomly();

        int32_t get_generation_id() const;
        void set_generation_id(int32_t generation_id);

        void clear_generated_by();
        void update_generation_map(map<string, int32_t> &generation_map);
        void set_generated_by(string type);
        int32_t get_generated_by(string type);


        RNN* get_rnn();
        vector<double> get_best_parameters() const;

        void set_best_parameters( vector<double> parameters);    //INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING
        void set_initial_parameters( vector<double> parameters);  //INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING

        void get_analytic_gradient(vector<RNN*> &rnns, const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, double &mse, vector<double> &analytic_gradient, bool training);

        void backpropagate(const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, const vector< vector< vector<double> > > &validation_inputs, const vector< vector< vector<double> > > &validation_outputs);

        void backpropagate_stochastic(const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, const vector< vector< vector<double> > > &validation_inputs, const vector< vector< vector<double> > > &validation_outputs);


        double get_mse(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs);
        double get_mae(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs);


        vector< vector<double> > get_predictions(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs);
        void write_predictions(const vector<string> &input_filenames, const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs);

        void get_mu_sigma(const vector<double> &p, double &mu, double &sigma);

        bool sanity_check();
        void assign_reachability();
        bool outputs_unreachable();

        RNN_Node_Interface* create_node(double mu, double sigma, int node_type, int32_t &node_innovation_count, double depth);

        bool attempt_edge_insert(RNN_Node_Interface *n1, RNN_Node_Interface *n2, double mu, double sigma, int32_t &edge_innovation_count);
        bool attempt_recurrent_edge_insert(RNN_Node_Interface *n1, RNN_Node_Interface *n2, double mu, double sigma, Distribution *dist, int32_t &edge_innovation_count);

        //after adding an Elman or Jordan node, generate the circular RNN edge for Elman and the
        //edges from output to this node for Jordan.
        void generate_recurrent_edges(RNN_Node_Interface *node, double mu, double sigma, Distribution *dist, int32_t &edge_innovation_count);

        bool add_edge(double mu, double sigma, int32_t &edge_innovation_count);
        bool add_recurrent_edge(double mu, double sigma, Distribution *d, int32_t &edge_innovation_count);
        bool disable_edge();
        bool enable_edge();
        bool split_edge(double mu, double sigma, int node_type, Distribution *dist, int32_t &edge_innovation_count, int32_t &node_innovation_count);


        bool add_node(double mu, double sigma, int node_type, Distribution *dist, int32_t &edge_innovation_count, int32_t &node_innovation_count);

        bool enable_node();
        bool disable_node();
        bool split_node(double mu, double sigma, int node_type, Distribution *dist, int32_t &edge_innovation_count, int32_t &node_innovation_count);

        bool merge_node(double mu, double sigma, int node_type, Distribution *dist, int32_t &edge_innovation_count, int32_t &node_innovation_count);


        /**
         * Determines if the genome contains a node with the given innovation number
         *
         * @param the innovation number to fine
         *
         * @return true if the genome has a node with the provided innovation number, false otherwise.
         */
        bool has_node_with_innovation(int32_t innovation_number) const;


        bool equals(RNN_Genome *other);

        string get_color(double weight, bool is_recurrent);
        void write_graphviz(string filename);

        RNN_Genome(string binary_filename);
        RNN_Genome(char* array, int32_t length);
        RNN_Genome(istream &bin_infile);

        void read_from_array(char *array, int32_t length);
        void read_from_stream(istream &bin_istream);

        void write_to_array(char **array, int32_t &length);
        void write_to_file(string bin_filename);
        void write_to_stream(ostream &bin_stream);

        bool connect_new_input_node( double mu, double sig, RNN_Node_Interface *new_node, Distribution *dist, int32_t &edge_innovation_count );
        bool connect_new_output_node( double mu, double sig, RNN_Node_Interface *new_node, Distribution *dist, int32_t &edge_innovation_count );
        bool connect_node_to_hid_nodes( double mu, double sig, RNN_Node_Interface *new_node, Distribution *dist, int32_t &edge_innovation_count, bool from_input );

        friend class EXAMM;
        friend class IslandSpeciationStrategy;
        friend class RecDepthFrequencyTable;
};

struct sort_genomes_by_fitness {
    bool operator()(RNN_Genome *g1, RNN_Genome *g2) {
        return g1->get_fitness() < g2->get_fitness();
    }
};

#endif
