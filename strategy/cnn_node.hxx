#ifndef CNN_NEAT_NODE_H
#define CNN_NEAT_NODE_H

#include <fstream>
using std::ofstream;
using std::ifstream;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::istream;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <sstream>
using std::istringstream;

#include <string>
using std::string;

#include <vector>
using std::vector;


#include "image_tools/image_set.hxx"

#ifdef _MYSQL_
#include "common/db_conn.hxx"
#endif

#include "common/random.hxx"

#define RELU_MIN 0
#define RELU_MIN_LEAK 0.005

#define RELU_MAX 5.5
#define RELU_MAX_LEAK 0.00

#define INPUT_NODE 0
#define HIDDEN_NODE 1
#define OUTPUT_NODE 2
#define SOFTMAX_NODE 3


class CNN_Node {
    private:
        int node_id;
        int exact_id;
        int genome_id;

        int innovation_number;
        double depth;

        int batch_size, size_y, size_x;

        int weight_count;

        int type;

        int total_inputs;
        int inputs_fired;

        int total_outputs;
        int outputs_fired;

        bool visited;
        bool needs_initialization;

        double gamma;
        double best_gamma;
        double previous_velocity_gamma;

        double beta;
        double best_beta;
        double previous_velocity_beta;

        double inverse_variance;

        double running_mean;
        double best_running_mean;
        double running_variance;
        double best_running_variance;

        //batch number x size_y x size_x
        vector< vector< vector<double> > > values;   //after RELU
        vector< vector< vector<double> > > errors;
        vector< vector< vector<double> > > gradients;


    public:
        CNN_Node();

        CNN_Node(int _innovation_number, double _depth, int _batch_size, int _input_size_x, int _input_size_y, int type);

        CNN_Node* copy() const;

#ifdef _MYSQL_
        CNN_Node(int node_id);
        void export_to_database(int exact_id, int genome_id);

        int get_node_id() const;
#endif

        bool needs_init() const;
        int get_batch_size() const;
        int get_size_x() const;
        int get_size_y() const;

        void initialize();

        void reset_velocities();

        void reset_weight_count();
        void add_weight_count(int _weight_count);
        int get_weight_count() const;

        int get_innovation_number() const;

        double get_depth() const;

        bool is_fixed() const;
        bool is_hidden() const;
        bool is_input() const;
        bool is_output() const;
        bool is_softmax() const;

        bool is_visited() const;
        void visit();
        void set_unvisited();

        bool has_nan() const;

        void set_values(const vector<const Image> &image, int channel, bool perform_dropout, double input_dropout_probability, minstd_rand0 &generator);
        double get_value(int batch_number, int y, int x);
        void set_value(int batch_number, int y, int x, double value);
        vector< vector< vector<double> > >& get_values();

        void set_error(int batch_number, int y, int x, double error);
        vector< vector< vector<double> > >& get_errors();

        void set_gradient(int batch_number, int y, int x, double gradient);
        vector< vector< vector<double> > >& get_gradients();

        void print(ostream &out);

        void reset();
        void save_best_weights();
        void set_weights_to_best();

        void resize_arrays();
        bool modify_size_x(int change);
        bool modify_size_y(int change);

        void add_input();
        void disable_input();
        int get_number_inputs() const;
        int get_inputs_fired() const;
        void input_fired(bool training, double epsilon, double alpha, bool perform_dropout, double hidden_dropout_probability, minstd_rand0 &generator);

        void add_output();
        void disable_output();
        int get_number_outputs() const;
        int get_outputs_fired() const;
        void output_fired(double mu, double learning_rate);

        //void batch_normalize(bool training, double epsilon, double alpha);
        void apply_relu();
        void apply_dropout(bool perform_dropout, double dropout_probability, minstd_rand0 &generator);

        /*
        void backpropagate_dropout();
        void backpropagate_relu(vector< vector< vector<double> > > &values);
        void backpropagate_batch_normalization(double mu, double learning_rate);
        */

        void print_statistics();

        friend ostream &operator<<(ostream &os, const CNN_Node* node);
        friend istream &operator>>(istream &is, CNN_Node* node);
};

double read_hexfloat(istream &infile);
void write_hexfloat(ostream &outfile, double value);

struct sort_CNN_Nodes_by_depth {
    bool operator()(const CNN_Node *n1, const CNN_Node *n2) {
        if (n1->get_depth() < n2->get_depth()) {
            return true;
        } else if (n1->get_depth() == n2->get_depth()) {
            //make sure the order of the nodes is *always* the same
            //going through the nodes in different orders may change
            //the output of backpropagation
            if (n1->get_innovation_number() < n2->get_innovation_number()) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
};

#endif
