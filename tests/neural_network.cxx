#include <iostream>
using std::cout;
using std::endl;

#include "node.hxx"
#include "edge.hxx"
#include "neural_network.hxx"

NeuralNetwork::NeuralNetwork() {
    learning_rate = 0.5;

    nodes = vector< vector<Node*> >(3, vector<Node*>());

    nodes[0].push_back( new Node(0.05, 0.0) );
    nodes[0].push_back( new Node(0.10, 0.0) );
    nodes[1].push_back( new Node(0.00, 0.35) );
    nodes[1].push_back( new Node(0.00, 0.35) );
    nodes[2].push_back( new Node(0.00, 0.60) );
    nodes[2].push_back( new Node(0.00, 0.60) );

    cout << "created nodes!" << endl;

    edges.push_back( new Edge(0.15, nodes[0][0], nodes[1][0]) );
    edges.push_back( new Edge(0.20, nodes[0][1], nodes[1][0]) );
    edges.push_back( new Edge(0.25, nodes[0][0], nodes[1][1]) );
    edges.push_back( new Edge(0.30, nodes[0][1], nodes[1][1]) );
    edges.push_back( new Edge(0.40, nodes[1][0], nodes[2][0]) );
    edges.push_back( new Edge(0.45, nodes[1][1], nodes[2][0]) );
    edges.push_back( new Edge(0.50, nodes[1][0], nodes[2][1]) );
    edges.push_back( new Edge(0.55, nodes[1][1], nodes[2][1]) );

    cout << "created edges!" << endl;

    o_expected.push_back(0.01);
    o_expected.push_back(0.99);

    cout << "set expected output!" << endl;
}

void NeuralNetwork::forward_pass() {
    for (uint32_t i = 0; i < nodes[0].size(); i++) {
        nodes[0][i]->fire();
        //cout << "nodes[0][" << i << "] out: " << nodes[0][i]->out << endl;
    }

    for (uint32_t i = 0; i < nodes[1].size(); i++) {
        //cout << "nodes[1][" << i << "] out (before activation): " << nodes[1][i]->out << endl;
        nodes[1][i]->activation_function();
        nodes[1][i]->fire();
        //cout << "nodes[1][" << i << "] out: " << nodes[1][i]->out << endl;
    }

    for (uint32_t i = 0; i < nodes[2].size(); i++) {
        //cout << "nodes[2][" << i << "] out (before activation): " << nodes[2][i]->out << endl;
        nodes[2][i]->activation_function();
        cout << "nodes[2][" << i << "] out: " << nodes[2][i]->out << endl;
    }
}

void NeuralNetwork::backward_pass() {
    total_error = 0.0;
    //do output layer
    for (uint32_t i = 0; i < nodes[2].size(); i++) {
        double error_difference = (o_expected[i] - nodes[2][i]->out);
        nodes[2][i]->error = 0.5 * error_difference * error_difference;

        //cout << "nodes[2][" << i << "]->error = " << nodes[2][i]->error << endl;

        total_error += nodes[2][i]->error;

        nodes[2][i]->dtotal_dout = (nodes[2][i]->out - o_expected[i]);
        //cout << "nodes[2][" << i << "]->dtotal_dout = " << nodes[2][i]->dtotal_dout << endl;

        nodes[2][i]->dout_dnet = nodes[2][i]->out * (1 - nodes[2][i]->out);
        //cout << "nodes[2][" << i << "]->dout_dnet = " << nodes[2][i]->dout_dnet << endl;

        for (uint32_t j = 0; j < nodes[2][i]->input_edges.size(); j++) {
            nodes[2][i]->input_edges[j]->next_weight = nodes[2][i]->input_edges[j]->weight - (nodes[2][i]->input_edges[j]->input->out * nodes[2][i]->dout_dnet * nodes[2][i]->dtotal_dout * learning_rate);
            //cout << "next_weight: " << nodes[2][i]->input_edges[j]->next_weight << endl;
        }
        cout << endl;
    }

    cout << "total error: " << total_error << endl;
    cout << endl;

    //do hidden layer
    for (uint32_t i = 0; i < nodes[1].size(); i++) {
        nodes[1][i]->dtotal_dout = 0.0;
        for (uint32_t j = 0; j < nodes[1][i]->output_edges.size(); j++) {
            double partial_dtotal_dout = nodes[1][i]->output_edges[j]->output->dtotal_dout * nodes[1][i]->output_edges[j]->output->dout_dnet * nodes[1][i]->output_edges[j]->weight;
            //cout << "nodes[1][i]->dtotal_dout[" << j << "] = " << partial_dtotal_dout << endl;
            nodes[1][i]->dtotal_dout += partial_dtotal_dout;
        }
        //cout << "nodes[1][" << i << "]->dtotal_dout = " << nodes[1][i]->dtotal_dout << endl;

        nodes[1][i]->dout_dnet = nodes[1][i]->out * (1 - nodes[1][i]->out);
        //cout << "nodes[1][" << i << "]->dout_dnet = " << nodes[1][i]->dout_dnet << endl;

        for (uint32_t j = 0; j < nodes[1][i]->input_edges.size(); j++) {
            nodes[1][i]->input_edges[j]->next_weight = nodes[1][i]->input_edges[j]->weight - (nodes[1][i]->input_edges[j]->input->out * nodes[1][i]->dout_dnet * nodes[1][i]->dtotal_dout * learning_rate);
            //cout << "next_weight: " << nodes[1][i]->input_edges[j]->next_weight << endl;
        }
        cout << endl;
    }
}

void NeuralNetwork::update_weights() {
    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->weight = edges[i]->next_weight;
    }
}
