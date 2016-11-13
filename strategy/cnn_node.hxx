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

#include <string>
using std::string;

#include <vector>
using std::vector;


#include "image_tools/image_set.hxx"

#define RELU_MIN 0
#define RELU_MIN_LEAK 0.005

#define RELU_MAX 5
#define RELU_MAX_LEAK 0.00


class CNN_Node {
    private:
        int innovation_number;
        int depth;

        int size_x, size_y;

        //int stride;
        //int max_pool;
        //int output_size_x, output_size_y;

        double **values;
        double **errors;

        bool input;
        bool output;
        bool softmax;

        int total_inputs;
        int inputs_fired;

    public:
        CNN_Node();

        CNN_Node(int _innovation_number, int _depth, int _size_x, int _size_y, bool _input, bool _output, bool _softmax);

        CNN_Node* copy() const;

        int get_size_x() const;
        int get_size_y() const;

        int get_innovation_number() const;

        int get_depth() const;

        bool is_fixed() const;
        bool is_hidden() const;
        bool is_input() const;
        bool is_output() const;
        bool is_softmax() const;

        void set_values(const Image &image, int rows, int cols);
        double** get_values();

        double get_error(int y, int x);
        void set_error(int y, int x, double value);
        double** get_errors();

        void print(ostream &out);

        void reset();

        double get_value(int y, int x);

        double set_value(int y, int x, double value);

        void add_input();

        void input_fired();

        friend ostream &operator<<(ostream &os, const CNN_Node* node);
        friend istream &operator>>(istream &is, CNN_Node* node);
};


#endif
