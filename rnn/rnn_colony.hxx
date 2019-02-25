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

#include <vector>
using std::vector;

#include "rnn.hxx"
#include "rnn_node_interface.hxx"
#include "rnn_edge.hxx"
#include "rnn_recurrent_edge.hxx"

#include "common/random.hxx"

//mysql can't handl the max float value for some reason
#define EXALT_MAX_DOUBLE 10000000

string parse_fitness(double fitness);


class RNN_Colony {
    private:
        int32_t bp_iterations;
        double learning_rate;
        bool adapt_learning_rate;
        bool use_reset_weights;

        bool use_high_norm;
        double high_threshold;
        bool use_low_norm;
        double low_threshold;

        string log_filename;

        map<string, int> generated_by_map;

        vector<double> initial_parameters;

        double best_validation_mse;
        double best_validation_mae;
        vector<double> best_parameters;

        uniform_real_distribution<double> rng_0_1;
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

        RNN_Colony(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges);
        RNN_Colony(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges);


        RNN_Colony* copy();

        ~RNN_Colony();

        static string print_statistics_header();
        string print_statistics();

        void set_parameter_names(const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names);

        string generated_by_string();

        string get_edge_count_str(bool recurrent);
        string get_node_count_str(int node_type);

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

        int32_t get_island() const;


        void set_bp_iterations(int32_t _bp_iterations);
        int32_t get_bp_iterations();

        void set_learning_rate(double _learning_rate);
        void set_adapt_learning_rate(bool _adapt_learning_rate);
        void set_reset_weights(bool _use_reset_weights);
        void disable_high_threshold();
        void enable_high_threshold(double _high_threshold);
        void disable_low_threshold();
        void enable_low_threshold(double _low_threshold);
        void set_log_filename(string _log_filename);

        void get_weights(vector<double> &parameters);
        void set_weights(const vector<double> &parameters);
        uint32_t get_number_weights();
        void initialize_randomly();

        int32_t get_generation_id() const;
        void set_generation_id(int32_t generation_id);

        void clear_generated_by();
        void update_generation_map(map<string, int32_t> &generation_map);
        void set_generated_by(string type);
        int32_t get_generated_by(string type);


        RNN* get_rnn();
        vector<double> get_best_parameters() const;

        void get_analytic_gradient(vector<RNN*> &rnns, const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, double &mse, vector<double> &analytic_gradient, bool training);

        void backpropagate(const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, const vector< vector< vector<double> > >   &validation_inputs, const vector< vector< vector<double> > > &validation_outputs);

        void backpropagate_stochastic(const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, const vector< vector< vector<double> > > &validation_inputs, const vector< vector< vector<double> > > &validation_outputs);


        double get_mse(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, bool verbose = false);
        double get_mae(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, bool verbose = false);

        void write_predictions(const vector<string> &input_filenames, const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs);

        bool sanity_check();
        void assign_reachability();
        bool outputs_unreachable();

        void print_information();

        bool equals(RNN_Colony *other);

        string get_color(double weight, bool is_recurrent);
        void write_graphviz(string filename);

        RNN_Colony(string binary_filename, bool verbose = false);
        RNN_Colony(char* array, int32_t length, bool verbose = false);
        RNN_Colony(istream &bin_infile, bool verbose = false);

        void read_from_array(char *array, int32_t length, bool verbose = false);
        void read_from_stream(istream &bin_istream, bool verbose = false);

        void write_to_array(char **array, int32_t &length, bool verbose = false);
        void write_to_file(string bin_filename, bool verbose = false);
        void write_to_stream(ostream &bin_stream, bool verbose = false);

        friend class ACNNTO;
};

struct sort_Colonys_by_fitness {
    bool operator()(RNN_Colony *g1, RNN_Colony *g2) {
        return g1->get_fitness() < g2->get_fitness();
    }
};


#endif
