#ifndef ONLINE_SERIES_HXX
#define ONLINE_SERIES_HXX

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <map>
using std::map;

#include <vector>
using std::vector;

#include <random>
using std::normal_distribution;
using std::default_random_engine;

class OnlineSeries {
    private:

        vector< double > mean;
        vector< double > std;
        // vector< vector< vector<double> > > input_time_series;
        // vector< vector< vector<double> > > output_time_series;
        normal_distribution<double> gaussian;
        default_random_engine noise_generator;
        // vector< vector< vector<double> > > input_sets;
        // vector< vector< vector<double> > > output_sets;
        vector< int32_t> scores;
        int32_t total_num_sets;
        int32_t sequence_length;
        int32_t current_index;
        vector< int32_t > avalibale_training_index;
        vector< int32_t > training_index;
        vector< int32_t > validation_index;
        int32_t num_training_sets;
        int32_t num_validataion_sets;
        string get_training_data_method;
        
    public:
        OnlineSeries(int32_t _num_sets, const vector<string> &arguments);
        ~OnlineSeries();

        void get_mean_std();
        void shuffle_data(int32_t num_recent_sets);
        void fill_training_index(int32_t num_sets, int32_t num_recent_sets);
        void validate_generation_number(int32_t num_generation);
        void set_current_index(int32_t _current_gen);
        void get_online_arguments(const vector<string> &arguments);
        void add_gaussian_noise(vector< vector<double> > &noisy_values, double noise_percent);
        void slice_series_to_sets();
        // void set_sequence_length(int32_t _sequence_length);
        // vector< vector <double> > get_input_values();
        // vector< vector <double> > get_output_values();
        vector< int32_t > get_training_index();
        vector< int32_t > get_validation_index();
        int32_t get_test_index();
        void add_score(int32_t index, int32_t score);

    
};



#endif