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

#define RELU_MAX 5
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

        int size_x, size_y;

        //int stride;
        //int max_pool;
        //int output_size_x, output_size_y;

        double **values;
        double **errors;
        double **bias;
        double **best_bias;
        double **bias_velocity;


        int type;

        int total_inputs;
        int inputs_fired;

        bool visited;

    public:
        CNN_Node();

        CNN_Node(int _innovation_number, double _depth, int _size_x, int _size_y, int type, minstd_rand0 &generator, NormalDistribution &normal_distribution);

        ~CNN_Node();

        CNN_Node* copy() const;

#ifdef _MYSQL_
        CNN_Node(int node_id);
        void export_to_database(int exact_id, int genome_id);
#endif

        int get_size_x() const;
        int get_size_y() const;

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

        void initialize_bias(minstd_rand0 &generator, NormalDistribution &normal_disribution);
        void initialize_velocities();

        bool has_zero_bias() const;
        bool has_zero_best_bias() const;
        void propagate_bias(double mu, double learning_rate, double weight_decay);

        void set_values(const Image &image, int rows, int cols);
        double** get_values();

        double get_error(int y, int x);
        void set_error(int y, int x, double value);
        double** get_errors();

        void print(ostream &out);

        void reset();

        double get_value(int y, int x);

        double set_value(int y, int x, double value);

        void save_best_bias();
        void set_bias_to_best();

        void resize_arrays(int previous_size_x, int previous_size_y);
        bool modify_size_x(int change, minstd_rand0 &generator, NormalDistribution &normal_distribution);
        bool modify_size_y(int change, minstd_rand0 &generator, NormalDistribution &normal_distribution);

        void add_input();
        void disable_input();
        int get_number_inputs() const;
        int get_inputs_fired() const;

        void input_fired();

        friend ostream &operator<<(ostream &os, const CNN_Node* node);
        friend istream &operator>>(istream &is, CNN_Node* node);
};

template<class T>
void parse_array_2d(T ***output, istringstream &iss, int size_x, int size_y);


struct sort_CNN_Nodes_by_depth {
    bool operator()(const CNN_Node *n1, const CNN_Node *n2) {
        return n1->get_depth() < n2->get_depth();
    }
};

#endif
