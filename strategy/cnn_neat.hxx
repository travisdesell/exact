#ifndef CNN_NEAT_H
#define CNN_NEAT_H

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

#include <string>
using std::to_string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"

class CNN_NEAT {
    private:
        int population_size;
        int last_node_innovation_number;
        int last_edge_innovation_number;

        uniform_int_distribution<long> rng_long;
        uniform_real_distribution<double> rng_double;

        vector <CNN_Node* > all_nodes;
        vector <CNN_Edge* > all_edges;

        vector< CNN_NEAT_Genome* > genomes;


    public:

        CNN_NEAT(const Images &images, int epochs) {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            //unsigned seed = 10;
            mt19937 generator(seed);

            rng_long = uniform_int_distribution<long>(-numeric_limits<long>::max(), numeric_limits<long>::max());
            rng_double = uniform_real_distribution<double>(0, 1.0);

            long genome_seed = rng_long(generator);
            cout << "seeding genome with: " << genome_seed << endl;

            CNN_Node *input_node = new CNN_Node(


            CNN_NEAT_Genome *first_genome = new CNN_NEAT_Genome(genome_seed, epochs, all_nodes, all_edges);

            genomes.push_back(first_genome);

            for (uint32_t i = 1; i < population_size; i++) {
                CNN_NEAT_Genome* genome = generate_mutation();

                genomes.push_back(genome);
            }
        }

        CNN_NEAT_Genome* get_best_genome() {
            return genomes[0];
        }
};



#endif
