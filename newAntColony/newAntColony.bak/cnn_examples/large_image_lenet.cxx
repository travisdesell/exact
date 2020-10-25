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

#include "image_tools/large_image_set.hxx"

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

    LargeImages training_images(training_filename, padding, 64, 64);
    LargeImages validation_images(validation_filename, padding, 64, 64, training_images.get_average(), training_images.get_std_dev());
    LargeImages testing_images(testing_filename, padding, 64, 64, training_images.get_average(), training_images.get_std_dev());

    int node_innovation_count = 0;
    int edge_innovation_count = 0;
    vector<CNN_Node*> nodes;
    vector<CNN_Node*> input_nodes;
    vector<CNN_Node*> layer1_nodes;
    vector<CNN_Node*> layer2_nodes;
    vector<CNN_Node*> layer3_nodes;
    vector<CNN_Node*> layer4_nodes;
    vector<CNN_Node*> layer5_nodes;
    vector<CNN_Node*> layer6_nodes;
    vector<CNN_Node*> layer7_nodes;
    vector<CNN_Node*> layer8_nodes;
    vector<CNN_Node*> softmax_nodes;

    vector<CNN_Edge*> edges;

    minstd_rand0 generator(time(NULL));
    NormalDistribution normal_distribution;

    for (int32_t i = 0; i < training_images.get_image_channels(); i++) {
        CNN_Node *input_node = new CNN_Node(++node_innovation_count, 0, batch_size, training_images.get_image_height(), training_images.get_image_width(), INPUT_NODE);
        nodes.push_back(input_node);
        input_nodes.push_back(input_node);
    }

    //first layer of filters
    //input node goes to 5 filters
    for (int32_t i = 0; i < 6; i++) {
        CNN_Node *layer1_node = new CNN_Node(++node_innovation_count, 1, batch_size, 64, 64, HIDDEN_NODE);
        nodes.push_back(layer1_node);
        layer1_nodes.push_back(layer1_node);

        for (int32_t j = 0; j < training_images.get_image_channels(); j++) {
            edges.push_back( new CNN_Edge(input_nodes[j], layer1_node, false, ++edge_innovation_count, CONVOLUTIONAL) );
        }
    }

    for (int32_t i = 0; i < 6; i++) {
        CNN_Node *layer2_node = new CNN_Node(++node_innovation_count, 2, batch_size, 32, 32, HIDDEN_NODE);
        nodes.push_back(layer2_node);
        layer2_nodes.push_back(layer2_node);

        edges.push_back( new CNN_Edge(layer1_nodes[i], layer2_node, false, ++edge_innovation_count, POOLING) );
    }


    for (int32_t i = 0; i < 6; i++) {
        CNN_Node *layer3_node = new CNN_Node(++node_innovation_count, 2, batch_size, 28, 28, HIDDEN_NODE);
        nodes.push_back(layer3_node);
        layer3_nodes.push_back(layer3_node);

        edges.push_back( new CNN_Edge(layer2_nodes[i], layer3_node, false, ++edge_innovation_count, CONVOLUTIONAL) );
    }


    for (int32_t i = 0; i < 6; i++) {
        CNN_Node *layer4_node = new CNN_Node(++node_innovation_count, 2, batch_size, 14, 14, HIDDEN_NODE);
        nodes.push_back(layer4_node);
        layer4_nodes.push_back(layer4_node);

        edges.push_back( new CNN_Edge(layer3_nodes[i], layer4_node, false, ++edge_innovation_count, POOLING) );
    }


    for (int32_t i = 0; i < 16; i++) {
        CNN_Node *layer5_node = new CNN_Node(++node_innovation_count, 3, batch_size, 10, 10, HIDDEN_NODE);
        nodes.push_back(layer5_node);
        layer5_nodes.push_back(layer5_node);
    }

    //0  to 0 1 2
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[0], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[0], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[0], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //1  to 1 2 3
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[1], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[1], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[1], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //2  to 2 3 4
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[2], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[2], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[2], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //3  to 3 4 5
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[3], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[3], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[3], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //4  to 4 5 0
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[4], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[4], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[4], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //5  to 5 0 1
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[5], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[5], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[5], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //6  to 0 1 2 3
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[6], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[6], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[6], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[6], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //7  to 1 2 3 4
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[7], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[7], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[7], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[7], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //8  to 2 3 4 5
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[8], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[8], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[8], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[8], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //9  to 3 4 5 0
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[9], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[9], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[9], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[9], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //10 to 4 5 0 1
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[10], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[10], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[10], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[10], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //11 to 5 0 1 2
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[11], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[11], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[11], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[11], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //12 to 0 1 3 4
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[12], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[12], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[12], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[12], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //13 to 1 2 4 5
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[13], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[13], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[13], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[13], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //14 to 0 2 3 5
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[14], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[14], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[14], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[14], false, ++edge_innovation_count, CONVOLUTIONAL) );

    //15 to 0 1 2 3 4 5
    edges.push_back( new CNN_Edge(layer4_nodes[0], layer5_nodes[15], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[1], layer5_nodes[15], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[2], layer5_nodes[15], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[3], layer5_nodes[15], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[4], layer5_nodes[15], false, ++edge_innovation_count, CONVOLUTIONAL) );
    edges.push_back( new CNN_Edge(layer4_nodes[5], layer5_nodes[15], false, ++edge_innovation_count, CONVOLUTIONAL) );

    for (int32_t i = 0; i < 16; i++) {
        CNN_Node *layer6_node = new CNN_Node(++node_innovation_count, 4, batch_size, 5, 5, HIDDEN_NODE);
        nodes.push_back(layer6_node);
        layer6_nodes.push_back(layer6_node);

        edges.push_back( new CNN_Edge(layer5_nodes[i], layer6_node, false, ++edge_innovation_count, POOLING) );
    }


    //fully connected to 100 layer
    for (int32_t i = 0; i < 120; i++) {
        CNN_Node *layer7_node = new CNN_Node(++node_innovation_count, 6, batch_size, 1, 1, HIDDEN_NODE);
        nodes.push_back(layer7_node);
        layer7_nodes.push_back(layer7_node);

        for (int32_t j = 0; j < 16; j++) {
            edges.push_back( new CNN_Edge(layer6_nodes[j], layer7_node, false, ++edge_innovation_count, CONVOLUTIONAL) );
        }
    }


    for (int32_t i = 0; i < 84; i++) {
        CNN_Node *layer8_node = new CNN_Node(++node_innovation_count, 7, batch_size, 1, 1, HIDDEN_NODE);
        nodes.push_back(layer8_node);
        layer8_nodes.push_back(layer8_node);

        for (int32_t j = 0; j < layer7_nodes.size(); j++) {
            edges.push_back( new CNN_Edge(layer7_nodes[j], layer8_node, false, ++edge_innovation_count, CONVOLUTIONAL) );
        }
    }


    for (int32_t i = 0; i < training_images.get_number_classes(); i++) {
        CNN_Node *softmax_node = new CNN_Node(++node_innovation_count, 8, batch_size, 1, 1, SOFTMAX_NODE);
        nodes.push_back(softmax_node);
        softmax_nodes.push_back(softmax_node);

        for (int32_t j = 0; j < layer8_nodes.size(); j++) {
            edges.push_back( new CNN_Edge(layer8_nodes[j], softmax_node, false, ++edge_innovation_count, CONVOLUTIONAL) );

        }
    }


    long genome_seed = generator();
    cout << "seeding genome with: " << genome_seed << endl;

    CNN_Genome *genome = new CNN_Genome(1, padding, training_images.get_number_images(), validation_images.get_number_images(), testing_images.get_number_images(), genome_seed, max_epochs, true, velocity_reset, mu, mu_delta, learning_rate, learning_rate_delta, weight_decay, weight_decay_delta, batch_size, epsilon, alpha, input_dropout_probability, hidden_dropout_probability, nodes, edges);
    //save the weights and bias of the initially generated genome for reuse
    genome->initialize();

    cout << "number edges: " << edges.size() << ", total weights: " << genome->get_number_weights() << endl;

    ofstream outfile("large_image_lenet.gv");
    genome->print_graphviz(outfile);
    outfile.close();

    genome->stochastic_backpropagation(training_images, validation_images);

    cout << "writing genome to file!" << endl;
    genome->write_to_file("./large_image_lenet.txt");

    cout << endl << "getting training images predictions." << endl;
    genome->evaluate_large_images(training_images, "./prediction_results_training/");

    cout << endl << "getting testing images predictions." << endl;
    genome->evaluate_large_images(testing_images, "./prediction_results_testing/");
    
    cout << endl << "getting statistics for test images:" << endl;
    genome->evaluate_test(testing_images);
}
