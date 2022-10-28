#include <string>
using std::string;

#include <vector>
using std::vector;

using std::min;

#include <algorithm> 
using std::shuffle;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "online_series.hxx"

OnlineSeries::OnlineSeries(const int32_t _total_num_sets,const vector<string> &arguments) {
    total_num_sets = _total_num_sets;
    current_index = 0;
    get_online_arguments(arguments);
}

void OnlineSeries::get_online_arguments(const vector<string> &arguments) {
    get_argument(arguments, "--num_validataion_sets", true, num_validataion_sets);
    get_argument(arguments, "--num_training_sets", true, num_training_sets);
    get_argument(arguments, "--get_train_data_by", true, get_training_data_method);
    get_argument(arguments, "--time_series_length", true, sequence_length);
    // get_argument(arguments, "--generation_genomes", true, generation_genomes);
    // get_argument(arguments, "--elite_population_size", true, elite_population_size);
}

void OnlineSeries::set_current_index(int32_t _current_gen) {
    //current index is the begining of validation index
    current_index = _current_gen + num_training_sets;
    Log::debug("current generation is %d, current index is %d\n", _current_gen, current_index);
}

void OnlineSeries::shuffle_data(int32_t num_recent_sets) {
    avalibale_training_index.clear();
    if (num_recent_sets <= 0) {
        for (int32_t i = 0; i < current_index; i++) {
            avalibale_training_index.push_back(i);
        }
    } else {
        for (int32_t i = 0; i < current_index - num_recent_sets; i++) {
            avalibale_training_index.push_back(i);
        }
    }

    auto rng = std::default_random_engine {};
    shuffle(avalibale_training_index.begin(), avalibale_training_index.end(), rng);
}

void OnlineSeries::fill_training_index(int32_t num_random_sets, int32_t num_recent_sets) {
    shuffle_data(num_recent_sets);
    training_index.clear();
    for (int32_t i = 0; i < num_random_sets; i++) {
        training_index.push_back(avalibale_training_index[i]);
    }
    for (int32_t i = 0; i < num_recent_sets; i++) {
        training_index.push_back(current_index - i -1);
    }
}

vector<int32_t> OnlineSeries::get_training_index() {

    if (get_training_data_method.compare("v1") == 0) {
        Log::debug("getting historical data with V1\n");
        // V1 means all the generated genome has different random historical data
        int32_t num_random_sets = min(num_training_sets, current_index);
        fill_training_index(num_random_sets, 0);
    } else if (get_training_data_method.compare("v2") == 0) {
        // V2 means all the genomes in the same generation share the same training data
        Log::debug("getting historical data with V2\n");
        if ((int32_t)avalibale_training_index.size() != current_index) {
            int32_t num_random_sets = min(num_training_sets, current_index);
            fill_training_index(num_random_sets, 0);
        }  else Log::error("training set generated\n");
    } else if (get_training_data_method.compare("v3") == 0) {
        // V3 means trained with half most recent data and half random historical data
        Log::debug("getting historical data with V3\n");
        int32_t num_random_sets = min(num_training_sets, current_index) / 2;
        fill_training_index(num_random_sets, num_training_sets - num_random_sets);
    }

    return training_index;
    
}

vector< int32_t > OnlineSeries::get_validation_index() {
    validation_index.clear();
    for (int32_t i = 0; i < num_validataion_sets; i++) {
        validation_index.push_back(current_index + i);
    }
    return validation_index;
}

int32_t OnlineSeries::get_test_index() {
    return current_index + num_validataion_sets;
}
