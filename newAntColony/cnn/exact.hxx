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
        string training_filename;
        string validation_filename;
        string test_filename;

        int number_training_images;
        int number_validation_images;
        int number_test_images;

        int padding;

        int image_channels;
        int image_rows;
        int image_cols;
        int number_classes;

        int population_size;
        int node_innovation_count;
        int edge_innovation_count;

        minstd_rand0 generator;
        NormalDistribution normal_distribution;
        uniform_int_distribution<long> rng_long;
        uniform_real_distribution<float> rng_float;

        vector< CNN_Genome* > genomes;

        int best_predictions_genome_id;
        CNN_Genome *best_predictions_genome;

        int genomes_generated;
        int inserted_genomes;
        int max_genomes;

        bool reset_weights;
        int max_epochs;
        bool use_sfmp;
        bool use_node_operations;

        float initial_batch_size_min;
        float initial_batch_size_max;
        float batch_size_min;
        float batch_size_max;

        float initial_mu_min;
        float initial_mu_max;
        float mu_min;
        float mu_max;

        float initial_mu_delta_min;
        float initial_mu_delta_max;
        float mu_delta_min;
        float mu_delta_max;

        float initial_learning_rate_min;
        float initial_learning_rate_max;
        float learning_rate_min;
        float learning_rate_max;

        float initial_learning_rate_delta_min;
        float initial_learning_rate_delta_max;
        float learning_rate_delta_min;
        float learning_rate_delta_max;

        float initial_weight_decay_min;
        float initial_weight_decay_max;
        float weight_decay_min;
        float weight_decay_max;

        float initial_weight_decay_delta_min;
        float initial_weight_decay_delta_max;
        float weight_decay_delta_min;
        float weight_decay_delta_max;

        float epsilon;

        float initial_alpha_min;
        float initial_alpha_max;
        float alpha_min;
        float alpha_max;

        int initial_velocity_reset_min;
        int initial_velocity_reset_max;
        int velocity_reset_min;
        int velocity_reset_max;

        float initial_input_dropout_probability_min;
        float initial_input_dropout_probability_max;
        float input_dropout_probability_min;
        float input_dropout_probability_max;

        float initial_hidden_dropout_probability_min;
        float initial_hidden_dropout_probability_max;
        float hidden_dropout_probability_min;
        float hidden_dropout_probability_max;

        float reset_weights_chance;

        float crossover_rate;
        float more_fit_parent_crossover;
        float less_fit_parent_crossover;
        float crossover_alter_edge_type;

        int number_mutations;
        float edge_alter_type;
        float edge_disable;
        float edge_enable;
        float edge_split;
        float edge_add;
        float node_change_size;
        float node_change_size_x;
        float node_change_size_y;
        float node_add;
        float node_split;
        float node_merge;
        float node_enable;
        float node_disable;

        map<string, int> inserted_from_map;
        map<string, int> generated_from_map;

    public:
#ifdef _MYSQL_
        static bool exists_in_database(int exact_id);
        EXACT(int exact_id);

        void export_to_database();
        void update_database();
#endif

        EXACT(const ImagesInterface &training_images, const ImagesInterface &validation_images, const ImagesInterface &test_images, int _padding, int _population_size, int _max_epochs, bool _use_sfmp, bool _use_node_operations, int _max_genomes, string _output_directory, string _search_name, bool _reset_weights);

        int32_t population_contains(CNN_Genome *genome) const;
        CNN_Genome* get_best_genome();

        int get_number_genomes() const;
        CNN_Genome* get_genome(int i);

        void generate_initial_hyperparameters(float &mu, float &mu_delta, float &learning_rate, float &learning_rate_delta, float &weight_decay, float &weight_decay_delta, float &alpha, int &velocity_reset, float &input_dropout_probability, float &hidden_dropout_probability, int &batch_size);

        void generate_simplex_hyperparameters(float &mu, float &mu_delta, float &learning_rate, float &learning_rate_delta, float &weight_decay, float &weight_decay_delta, float &alpha, int &velocity_reset, float &input_dropout_probability, float &hidden_dropout_probability, int &batch_size);


        bool add_edge(CNN_Genome *child, CNN_Node *node1, CNN_Node *node2, int edge_type);

        CNN_Genome* generate_individual();
        CNN_Genome* create_mutation();
        CNN_Genome* create_child();

        bool insert_genome(CNN_Genome* genome);

        int get_inserted_genomes() const;
        int get_max_genomes() const;

        void write_individual_hyperparameters(CNN_Genome *individual);
        void write_statistics(int new_generation_id, float new_fitness);
        void write_statistics_header();
        void write_hyperparameters_header();

        int get_id() const;
        string get_search_name() const;
        string get_output_directory() const;
        string get_training_filename() const;
        string get_validation_filename() const;
        string get_test_filename() const;
        int get_number_training_images() const;

        bool is_identical(EXACT *other, bool testing_checkpoint);
};

#endif
