#ifndef CNN_EDGE_H
#define CNN_EDGE_H

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
using std::normal_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "cnn_node.hxx"
#include "image_tools/image_set.hxx"
#include "common/random.hxx"

#define CONVOLUTIONAL 0
#define POOLING 1

class CNN_Edge {
    private:
        int edge_id;
        int exact_id;
        int genome_id;

        int type;
        int innovation_number;

        int input_node_innovation_number;
        int output_node_innovation_number;

        CNN_Node *input_node;
        CNN_Node *output_node;

        int batch_size;
        int filter_x, filter_y;
        int filter_size;
        float *weights;
        float *weight_updates;
        float *best_weights;

        float *previous_velocity;
        float *best_velocity;

        float scale;
        float best_scale;
        float previous_velocity_scale;
        float best_velocity_scale;

        vector<int> y_pools;
        vector<int> y_pool_offset;
        vector<int> x_pools;
        vector<int> x_pool_offset;

        bool fixed;
        bool disabled;
        bool forward_visited;
        bool reverse_visited;

        bool reverse_filter_x;
        bool reverse_filter_y;
        bool needs_initialization;

        float propagate_backward_time;
        float propagate_forward_time;
        float weight_update_time;

    public:
        CNN_Edge();

        CNN_Edge(CNN_Node *_input_node, CNN_Node *_output_node, bool _fixed, int _innovation_number, int _type);

#ifdef _MYSQL_
        CNN_Edge(int edge_id);
        void export_to_database(int exact_id, int genome_id);

        int get_edge_id() const;
#endif


        CNN_Edge* copy() const;

        ~CNN_Edge();

        bool equals(CNN_Edge *other) const;

        int get_type() const;

        bool has_nan() const;

        void reset_times();
        void accumulate_times(float &total_forward_time, float &total_backward_time, float &total_weight_update_time);

        void set_needs_init();
        bool needs_init() const;
        int get_filter_size() const;
        int get_filter_x() const;
        int get_filter_y() const;

        bool is_reverse_filter_x() const;
        bool is_reverse_filter_y() const;

        float get_weight(int i) const;
        float get_weight_update(int i) const;
        void update_weight(int i, float diff);

        float get_scale() const;

        void propagate_weight_count();

        void save_best_weights();
        void set_weights_to_best();

        bool set_nodes(const vector<CNN_Node*> nodes);
        void set_pools();

        void initialize_weights(minstd_rand0 &generator, NormalDistribution &normal_distribution);
        void reset_velocities();
        void resize();

        void alter_edge_type();
        void disable();
        void enable();
        bool is_enabled() const;
        bool is_disabled() const;
        void update_batch_size(int new_batch_size);

        bool is_reachable() const;
        bool is_forward_visited() const;
        bool is_reverse_visited() const;
        void forward_visit();
        void reverse_visit();
        void set_unvisited();

        bool is_filter_correct() const;

        int get_number_weights() const;
        int get_batch_size() const;

        int get_innovation_number() const;
        int get_input_innovation_number() const;
        int get_output_innovation_number() const;

        bool connects(int n1, int n2) const;

        bool has_zero_weight() const;
        bool has_zero_best_weight() const;

        CNN_Node* get_input_node();

        CNN_Node* get_output_node();

        void print(ostream &out);

        void check_output_update(const vector< vector< vector<float> > > &output, const vector< vector< vector<float> > > &input, float value, float weight, float previous_output, int batch_number, int in_y, int in_x, int out_y, int out_x);

        void check_weight_update(const vector< vector< vector<float> > > &input, const vector< vector< vector<float> > > &input_deltas, float delta, float previous_delta, float weight_update, float previous_weight_update, int batch_number, int out_y, int out_x, int in_y, int in_x);

        void propagate_forward(bool training, bool accumulate_test_statistics, float epsilon, float alpha, bool perform_dropout, float hidden_dropout_probability, minstd_rand0 &generator);

        void propagate_backward(bool training, float mu, float learning_rate, float epsilon);
        void update_weights(float mu, float learning_rate, float weight_decay);

        void print_statistics();

        bool is_identical(const CNN_Edge *other, bool testing_checkpoint);

        friend ostream &operator<<(ostream &os, const CNN_Edge* flight);
        friend istream &operator>>(istream &is, CNN_Edge* flight);
};

int random_edge_type(float random_value);

void parse_float_2d(float **output, istringstream &iss, int size_x, int size_y);
void parse_vector_2d(vector<vector<float>> &output, istringstream &iss, int size_x, int size_y);

struct sort_CNN_Edges_by_depth {
    bool operator()(CNN_Edge *n1, CNN_Edge *n2) {
        if (n1->get_input_node()->get_depth() < n2->get_input_node()->get_depth()) {
            return true;

        } else if (n1->get_input_node()->get_depth() == n2->get_input_node()->get_depth()) {
            //make sure the order of the edges is *always* the same
            //going through the edges in different orders may effect the output
            //of backpropagation
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

struct sort_CNN_Edges_by_output_depth {
    bool operator()(CNN_Edge *n1, CNN_Edge *n2) {
        if (n1->get_output_node()->get_depth() < n2->get_output_node()->get_depth()) {
            return true;

        } else if (n1->get_output_node()->get_depth() == n2->get_output_node()->get_depth()) {
            //make sure the order of the edges is *always* the same
            //going through the edges in different orders may effect the output
            //of backpropagation
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


struct sort_CNN_Edges_by_innovation {
    bool operator()(CNN_Edge *n1, CNN_Edge *n2) {
        return n1->get_innovation_number() < n2->get_innovation_number();
    }   
};



#endif
