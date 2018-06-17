#ifndef CNN_GENOME_H
#define CNN_GENOME_H

#include <map>
using std::map;

#include <random>
using std::minstd_rand0;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "image_tools/large_image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "common/random.hxx"

#define SANITY_CHECK_BEFORE_INSERT 0
#define SANITY_CHECK_AFTER_GENERATION 1

//mysql can't handl the max float value for some reason
#define EXACT_MAX_FLOAT 10000000

class CNN_Genome {
    private:
        string version_str;
        int exact_id;
        int genome_id;

        vector<CNN_Node*> nodes;
        vector<CNN_Edge*> edges;

        vector<CNN_Node*> input_nodes;
        vector<CNN_Node*> softmax_nodes;

        NormalDistribution normal_distribution;
        minstd_rand0 generator;

        int velocity_reset;
        int batch_size;

        float epsilon;
        float alpha;

        float input_dropout_probability;
        float hidden_dropout_probability;

        float initial_mu;
        float mu;
        float mu_delta;

        float initial_learning_rate;
        float learning_rate;
        float learning_rate_delta;

        float initial_weight_decay;
        float weight_decay;
        float weight_decay_delta;

        int epoch;
        int max_epochs;
        bool reset_weights;

        int padding;

        int best_epoch;
        int number_validation_images;
        float best_validation_error;
        int best_validation_predictions;

        int number_training_images;
        float training_error;
        int training_predictions;

        int number_test_images;
        float test_error;
        int test_predictions;

        bool started_from_checkpoint;
        vector<long> backprop_order;

        int generation_id;
        string name;
        string checkpoint_filename;
        string output_filename;

        map<string, int> generated_by_map;

        int (*progress_function)(float);

    public:
        /**
         *  Initialize a genome from a file
         */
        CNN_Genome(string filename, bool is_checkpoint);
        CNN_Genome(istream &in, bool is_checkpoint);

        /**
         *  Iniitalize a genome from a set of nodes and edges
         */
        CNN_Genome(int _generation_id, int _padding, int _number_training_images, int _number_validation_images, int _number_test_images, int seed, int _max_epochs, bool _reset_weights, int velocity_reset, float _mu, float _mu_delta, float _learning_rate, float _learning_rate_delta, float _weight_decay, float _weight_decay_delta, int _batch_size, float _epsilon, float _alpha, float _input_dropout_probability, float _hidden_dropout_probability, const vector<CNN_Node*> &_nodes, const vector<CNN_Edge*> &_edges);

        ~CNN_Genome();

#ifdef _MYSQL_
        CNN_Genome(int genome_id);
        void export_to_database(int exact_id);
#endif

        float get_version() const;
        string get_version_str() const;


        int get_genome_id() const;
        int get_exact_id() const;

        bool equals(CNN_Genome *other) const;

        void print_results(ostream &out) const;

        void print_progress(ostream &out, string progress_name, float total_error, int correct_predictions, int number_images) const;

        int get_number_training_images() const;
        int get_number_validation_images() const;
        int get_number_test_images() const;

        int get_number_weights() const;

        int get_padding() const;

        int get_operations_estimate() const;

        void set_progress_function(int (*_progress_function)(float));

        int get_generation_id() const;

        float get_best_validation_error() const;
        float get_best_validation_rate() const;
        int get_best_validation_predictions() const;

        float get_training_error() const;
        float get_training_rate() const;
        int get_training_predictions() const;

        float get_test_error() const;
        float get_test_rate() const;
        int get_test_predictions() const;

        int get_best_epoch() const;

        int get_epoch() const;
        int get_max_epochs() const;

        bool sanity_check(int type);
        bool visit_nodes();

        void get_node_copies(vector<CNN_Node*> &node_copies) const;
        void get_edge_copies(vector<CNN_Edge*> &edge_copies) const;

        const vector<CNN_Node*> get_nodes() const;
        const vector<CNN_Edge*> get_edges() const;

        CNN_Node* get_node(int node_position);
        CNN_Edge* get_edge(int edge_position);

        float get_initial_mu() const;
        float get_mu() const;
        float get_mu_delta() const;
        float get_initial_learning_rate() const;
        float get_learning_rate() const;
        float get_learning_rate_delta() const;
        float get_initial_weight_decay() const;
        float get_weight_decay() const;
        float get_weight_decay_delta() const;
        int get_batch_size() const;

        float get_alpha() const;
        int get_velocity_reset() const;

        float get_input_dropout_probability() const;
        float get_hidden_dropout_probability() const;

        int get_number_enabled_pooling_edges() const;
        int get_number_enabled_convolutional_edges() const;
        int get_number_enabled_edges() const;
        int get_number_enabled_nodes() const;

        int get_number_edges() const;
        int get_number_nodes() const;
        int get_number_softmax_nodes() const;
        int get_number_input_nodes() const;

        vector<CNN_Node*> get_reachable_nodes();
        vector<CNN_Node*> get_disabled_nodes();
        vector<CNN_Node*> get_reachable_hidden_nodes();
        vector<CNN_Edge*> get_reachable_edges();
        vector<CNN_Edge*> get_disabled_edges();

        void add_node(CNN_Node* node);
        void add_edge(CNN_Edge* edge);
        bool disable_edge(int edge_position);

        void resize_edges_around_node(int node_position);
 
        void evaluate_images(const ImagesInterface &images, const vector<int> &batch, vector< vector<float> > &predictions, int offset);
        void evaluate_images(const ImagesInterface &images, const vector<int> &batch, bool training, float &total_error, int &correct_predictions, bool accumulate_test_statistics);

        void set_to_best();
        void save_to_best();

        void reset(bool _reset_weights);
        void initialize();

        void check_gradients(const ImagesInterface &images);

        void evaluate_large_images(const LargeImages &images, string output_directory);

        void evaluate(const ImagesInterface &images, vector< vector<float> > &predictions);
        void evaluate(const ImagesInterface &images, const vector<long> &order, float &total_error, int &correct_predictions, bool perform_backprop, bool accumulate_test_statistics);
        void evaluate(string progress_name, const ImagesInterface &images, float &total_error, int &correct_predictions);

        void stochastic_backpropagation(const ImagesInterface &training_images, int training_resize, const ImagesInterface &validation_images);
        void stochastic_backpropagation(const ImagesInterface &training_images, const ImagesInterface &validation_images);

        void evaluate_test(const ImagesInterface &test_images);

        void set_name(string _name);
        void set_output_filename(string _output_filename);
        void set_checkpoint_filename(string _checkpoint_filename);

        void write(ostream &outfile);
        void write_to_file(string filename);

        void read(istream &infile);

        void print_graphviz(ostream &out) const;

        void set_generated_by(string type);
        int get_generated_by(string type);

        bool is_identical(CNN_Genome *other, bool testing_checkpoint);

        void get_prediction_matrix(const MultiImagesInterface &images, int image_number, int stride, vector< vector< vector<float> > > &prediction_matrix);
        void get_expanded_prediction_matrix(const MultiImagesInterface &images, int image_number, int stride, int prediction_class, vector< vector<float> > &extended_prediction_matrix);

};

void write_map(ostream &out, map<string, int> &m);
void read_map(istream &in, map<string, int> &m);

struct sort_genomes_by_validation_error {
    bool operator()(CNN_Genome *g1, CNN_Genome *g2) {
        return g1->get_best_validation_error() < g2->get_best_validation_error();
    }   
};

struct sort_genomes_by_predictions {
    bool operator()(CNN_Genome *g1, CNN_Genome *g2) {
        return g1->get_best_validation_predictions() > g2->get_best_validation_predictions();
    }   
};



#endif
