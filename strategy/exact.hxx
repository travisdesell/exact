#ifndef EXACT_H
#define EXACT_H

#include <algorithm>
using std::sort;

#include <iostream>
using std::ostream;
using std::istream;

#include <limits>
using std::numeric_limits;

#include <random>
using std::mt19937;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <string>
using std::to_string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"

class EXACT {
    private:
        int population_size;
        int node_innovation_count;
        int edge_innovation_count;

        uniform_int_distribution<long> rng_long;
        uniform_real_distribution<double> rng_double;

        vector <CNN_Node* > all_nodes;
        vector <CNN_Edge* > all_edges;

        vector< CNN_Genome* > genomes;


        int epochs;
        mt19937 generator;

    public:

        EXACT(const Images &images, int _population_size, int _epochs);

        CNN_Genome* get_best_genome();

        CNN_Genome* create_mutation();
        CNN_Genome* create_child();

        void insert_genome(CNN_Genome* genome);
};



#endif
