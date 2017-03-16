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
using std::minstd_rand0;
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
        int id;
        string search_name;

        string output_directory;

        int number_images;
        int image_rows;
        int image_cols;
        int number_classes;

        int population_size;
        int node_innovation_count;
        int edge_innovation_count;

        minstd_rand0 generator;
        NormalDistribution normal_distribution;
        uniform_int_distribution<long> rng_long;
        uniform_real_distribution<double> rng_double;

        vector <CNN_Node* > all_nodes;
        vector <CNN_Edge* > all_edges;

        vector< CNN_Genome* > genomes;

        int genomes_generated;
        int inserted_genomes;
        int max_genomes;

        bool reset_weights;
        int max_epochs;
<<<<<<< HEAD
=======
        int improvement_required_epochs;
>>>>>>> 9d06146dad72324abd2295b530070590d941ed0e


        double initial_mu_min;
        double initial_mu_max;
        double mu_min;
        double mu_max;

        double initial_mu_delta_min;
        double initial_mu_delta_max;
        double mu_delta_min;
        double mu_delta_max;

        double initial_learning_rate_min;
        double initial_learning_rate_max;
        double learning_rate_min;
        double learning_rate_max;

        double initial_learning_rate_delta_min;
        double initial_learning_rate_delta_max;
        double learning_rate_delta_min;
        double learning_rate_delta_max;

        double initial_weight_decay_min;
        double initial_weight_decay_max;
        double weight_decay_min;
        double weight_decay_max;

        double initial_weight_decay_delta_min;
        double initial_weight_decay_delta_max;
        double weight_decay_delta_min;
        double weight_decay_delta_max;

        double initial_input_dropout_probability_min;
        double initial_input_dropout_probability_max;
        double input_dropout_probability_min;
        double input_dropout_probability_max;

        double initial_hidden_dropout_probability_min;
        double initial_hidden_dropout_probability_max;
        double hidden_dropout_probability_min;
        double hidden_dropout_probability_max;

        int initial_velocity_reset_min;
        int initial_velocity_reset_max;
        int velocity_reset_min;
        int velocity_reset_max;

        double reset_weights_chance;

        double crossover_rate;
        double more_fit_parent_crossover;
        double less_fit_parent_crossover;

        int number_mutations;
        double edge_disable;
        double edge_enable;
        double edge_split;
        double edge_add;
        double edge_change_stride;
        double node_change_size;
        double node_change_size_x;
        double node_change_size_y;
        double node_change_pool_size;

        int inserted_from_disable_edge;
        int inserted_from_enable_edge;
        int inserted_from_split_edge;
        int inserted_from_add_edge;
        int inserted_from_change_size;
        int inserted_from_change_size_x;
        int inserted_from_change_size_y;
        int inserted_from_crossover;
        int inserted_from_reset_weights;

        int generated_from_disable_edge;
        int generated_from_enable_edge;
        int generated_from_split_edge;
        int generated_from_add_edge;
        int generated_from_change_size;
        int generated_from_change_size_x;
        int generated_from_change_size_y;
        int generated_from_crossover;
        int generated_from_reset_weights;

<<<<<<< HEAD

=======
        bool sort_by_fitness;
>>>>>>> 9d06146dad72324abd2295b530070590d941ed0e
    public:
#ifdef _MYSQL_
        static bool exists_in_database(int exact_id);
        EXACT(int exact_id);

        void export_to_database();
        void update_database();
#endif

<<<<<<< HEAD
        EXACT(const Images &images, int _population_size, int _max_epochs, int _max_genomes, string _output_directory, string _search_name, bool _reset_weights);
=======
        EXACT(const Images &images, int _population_size, int _min_epochs, int _max_epochs, int _improvement_required_epochs, bool _reset_weights, double _mu, double _mu_decay, double _learning_rate, double _learning_rate_decay, double _weight_decay, double _weight_decay_decay, int _max_genomes, string _output_directory, string _search_name);
>>>>>>> 9d06146dad72324abd2295b530070590d941ed0e

        int32_t population_contains(CNN_Genome *genome) const;
        CNN_Genome* get_best_genome();

        void generate_initial_hyperparameters(double &mu, double &mu_delta, double &learning_rate, double &learning_rate_delta, double &weight_decay, double &weight_decay_delta, double &input_dropout_probability, double &hidden_dropout_probability, int &velocity_reset);

        void generate_simplex_hyperparameters(double &mu, double &mu_delta, double &learning_rate, double &learning_rate_delta, double &weight_decay, double &weight_decay_delta, double &input_dropout_probability, double &hidden_dropout_probability, int &velocity_reset);


        CNN_Genome* generate_individual();
        CNN_Genome* create_mutation();
        CNN_Genome* create_child();

        bool insert_genome(CNN_Genome* genome);

        int get_inserted_genomes() const;
        int get_max_genomes() const;

        void write_statistics(int new_generation_id, double new_fitness);
        void write_statistics_header();

        int get_id() const;
        string get_search_name() const;
        string get_output_directory() const;
        int get_number_images() const;
};

#endif
