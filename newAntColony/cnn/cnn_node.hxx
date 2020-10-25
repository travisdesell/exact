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
        float depth;

        int batch_size, size_y, size_x;
        int total_size;

        int weight_count;

        int type;

        int total_inputs;
        int inputs_fired;

        int total_outputs;
        int outputs_fired;

        bool forward_visited;
        bool reverse_visited;
        bool needs_initialization;

        bool disabled;

        float gamma;
        float best_gamma;
        float previous_velocity_gamma;

        float beta;
        float best_beta;
        float previous_velocity_beta;

        float batch_mean;
        float batch_variance; //sigma squared
        float batch_std_dev; //sqrt(batch_variance + epsilon);
        float inverse_variance;

        float running_mean;
        float best_running_mean;
        float running_variance;
        float best_running_variance;

        //batch number x size_y x size_x
        float *values_out;
        float *errors_out;
        float *relu_gradients;
        float *pool_gradients;

        float *values_in;
        float *errors_in;

        float input_fired_time;
        float output_fired_time;


    public:
        CNN_Node();
        ~CNN_Node();

        CNN_Node(int _innovation_number, float _depth, int _batch_size, int _input_size_x, int _input_size_y, int type);

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
        bool vectors_correct() const;

        void initialize();

        void reset_velocities();

        void reset_weight_count();
        void add_weight_count(int _weight_count);
        int get_weight_count() const;

        int get_innovation_number() const;

        float get_depth() const;

        bool is_fixed() const;
        bool is_hidden() const;
        bool is_input() const;
        bool is_output() const;
        bool is_softmax() const;

        bool is_reachable() const;
        bool is_forward_visited() const;
        bool is_reverse_visited() const;
        void forward_visit();
        void reverse_visit();
        void set_unvisited();

        bool has_nan() const;

        void set_values(const ImagesInterface &images, const vector<int> &batch, int channel, bool perform_dropout, bool accumulate_test_statistics, float input_dropout_probability, minstd_rand0 &generator);

        float get_value_in(int batch_number, int y, int x);
        void set_value_in(int batch_number, int y, int x, float value);
        float* get_values_in();

        float get_value_out(int batch_number, int y, int x);
        void set_value_out(int batch_number, int y, int x, float value);
        float* get_values_out();


        void set_error_in(int batch_number, int y, int x, float error);
        float* get_errors_in();

        void set_error_out(int batch_number, int y, int x, float error);
        float* get_errors_out();


        float* get_relu_gradients();
        float* get_pool_gradients();

        void print(ostream &out);

        void reset_times();
        void accumulate_times(float &total_input_time, float &total_output_time);

        void reset();
        void save_best_weights();
        void set_weights_to_best();

        void resize_arrays();
        void update_batch_size(int new_batch_size);
        bool modify_size_x(int change);
        bool modify_size_y(int change);

        bool is_disabled() const;
        bool is_enabled() const;
        void disable();
        void enable();

        void add_input();
        void disable_input();
        int get_number_inputs() const;
        int get_inputs_fired() const;
        void input_fired(bool training, bool accumulate_test_statistics, float epsilon, float alpha, bool perform_dropout, float hidden_dropout_probability, minstd_rand0 &generator);

        void add_output();
        void disable_output();
        int get_number_outputs() const;
        int get_outputs_fired() const;
        void output_fired(bool training, float mu, float learning_rate, float epsilon);

        void zero_test_statistics();
        void divide_test_statistics(int number_batches);

        void print_batch_statistics();

        void batch_normalize(bool training, bool accumulating_test_statistics, float epsilon, float alpha);
        void apply_relu(float* values, float* gradients);
        void apply_dropout(float* values, float* gradients, bool perform_dropout, bool accumulate_test_statistics, float dropout_probability, minstd_rand0 &generator);

        //void backpropagate_dropout();
        void backpropagate_relu(float* errors, float* gradients);
        void backpropagate_batch_normalization(bool training, float mu, float learning_rate, float epsilon);

        void print_statistics();
        void print_statistics(const float* values, const float* errors, const float* gradients);

        bool is_identical(const CNN_Node *other, bool testing_checkpoint);

        friend ostream &operator<<(ostream &os, const CNN_Node* node);
        friend istream &operator>>(istream &is, CNN_Node* node);
};

float read_hexfloat(istream &infile);
void write_hexfloat(ostream &outfile, float value);

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
