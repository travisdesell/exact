#ifndef CNN_GENOME_H
#define CNN_GENOME_H

#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"

#include <vector>
using std::vector;

class CNN_NEAT_Genome {
    private:
        CNN_Node *input_node;
        vector<CNN_Node*> nodes;
        vector< vector<CNN_Node*> > hidden_nodes;
        vector<CNN_Node*> output_nodes;
        vector<CNN_Node*> softmax_nodes;

        vector<CNN_Edge*> edges;

        mt19937 generator;

        double initial_mu;
        double mu;
        int epoch;
        int epochs;
        int best_predictions;

        bool started_from_checkpoint;
        vector<long> backprop_order;

    public:
        CNN_NEAT_Genome();

        /**
         *  Iniitalize a genome from a set of nodes and edges
         */
        CNN_NEAT_Genome(int seed, int _epochs, vector<CNN_Node*> _nodes, vector<CNN_Edge*> _edges);

        /**
         *  Initialize the initial genotype for the CNN_NEAT algorithm from
         *  a set of training images
         */
        CNN_NEAT_Genome(int number_classes, int rows, int cols, int seed, int _epochs);

        int evaluate_image(const Image &image, vector<double> &class_error, bool do_backprop);

        void stochastic_backpropagation(const Images &images, string checkpoint_filename, string output_filename);

        void write_to_file(string filename);
        void read_from_file(string filename);

};


#endif
