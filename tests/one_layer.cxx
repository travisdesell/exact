#include <iomanip>
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <random>
using std::minstd_rand0;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#include "cnn/exact.hxx"
#include "cnn/cnn_genome.hxx"
#include "cnn/cnn_edge.hxx"
#include "cnn/cnn_node.hxx"

#include "image_tools/image_set.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string training_filename;
    get_argument(arguments, "--training_file", true, training_filename);

    string validation_filename;
    get_argument(arguments, "--validation_file", true, validation_filename);

    string testing_filename;
    get_argument(arguments, "--testing_file", true, testing_filename);

    int padding;
    get_argument(arguments, "--padding", true, padding);

    int max_epochs;
    get_argument(arguments, "--max_epochs", true, max_epochs);

    double learning_rate;
    get_argument(arguments, "--learning_rate", true, learning_rate);

    double learning_rate_delta;
    get_argument(arguments, "--learning_rate_delta", true, learning_rate_delta);

    double weight_decay;
    get_argument(arguments, "--weight_decay", true, weight_decay);

    double weight_decay_delta;
    get_argument(arguments, "--weight_decay_delta", true, weight_decay_delta);

    double mu;
    get_argument(arguments, "--mu", true, mu);

    double mu_delta;
    get_argument(arguments, "--mu_delta", true, mu_delta);

    int velocity_reset;
    get_argument(arguments, "--velocity_reset", true, velocity_reset);

    int batch_size;
    get_argument(arguments, "--batch_size", true, batch_size);

    double alpha;
    get_argument(arguments, "--alpha", true, alpha);


    double input_dropout_probability;
    get_argument(arguments, "--input_dropout_probability", true, input_dropout_probability);

    double hidden_dropout_probability;
    get_argument(arguments, "--hidden_dropout_probability", true, hidden_dropout_probability);

    double epsilon = 1.0e-7;

    Images training_images(training_filename, padding);
    Images validation_images(validation_filename, padding, training_images.get_average(), training_images.get_std_dev());
    Images testing_images(testing_filename, padding, training_images.get_average(), training_images.get_std_dev());

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    vector<CNN_Node*> nodes;
    vector<CNN_Node*> layer1_nodes;
    vector<CNN_Node*> softmax_nodes;

    vector<CNN_Edge*> edges;

    minstd_rand0 generator(time(NULL));
    NormalDistribution normal_distribution;

    CNN_Node *input_node = new CNN_Node(++node_innovation_count, 0, batch_size, training_images.get_image_height(), training_images.get_image_width(), INPUT_NODE);
    nodes.push_back(input_node);

    for (int32_t i = 0; i < training_images.get_number_classes(); i++) {
        CNN_Node *softmax_node = new CNN_Node(++node_innovation_count, 2, batch_size, 1, 1, SOFTMAX_NODE);
        nodes.push_back(softmax_node);
        softmax_nodes.push_back(softmax_node);

        edges.push_back( new CNN_Edge(input_node, softmax_node, false, ++edge_innovation_count, CONVOLUTIONAL) );
    }

    long genome_seed = generator();
    cout << "seeding genome with: " << genome_seed << endl;

    CNN_Genome *genome = new CNN_Genome(1, padding, training_images.get_number_images(), validation_images.get_number_images(), testing_images.get_number_images(), genome_seed, max_epochs, true, velocity_reset, mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, batch_size, epsilon, alpha, input_dropout_probability, hidden_dropout_probability, nodes, edges);
    //save the weights and bias of the initially generated genome for reuse
    genome->initialize();

    cout << "number edges: " << edges.size() << ", total weights: " << genome->get_number_weights() << endl;

    ofstream outfile("one_layer.gv");
    genome->print_graphviz(outfile);
    outfile.close();

    genome->check_gradients(training_images);

    genome->stochastic_backpropagation(training_images, validation_images);
    genome->evaluate_test(testing_images);
}
