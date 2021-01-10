#ifndef DATASET_META_HXX
#define DATASET_META_HXX 1

#include <vector>
#include <string>
#include <map>

/**
 * Meta information about the dataset that is needed to train a genome
 **/
class DatasetMeta {
    public:
        const vector<string> input_parameter_names;
        const vector<string> output_parameter_names;

        const string normalize_type;
        const map<string, double> normalize_mins;
        const map<string, double> normalize_maxs;
        const map<string, double> normalize_avgs;
        const map<string, double> normalize_std_devs;

        DatasetMeta(
                const vector<string> &_input_parameter_names,
                const vector<string> &_output_parameter_names,
                const string _normalize_type,
                const map<string,double> &_normalize_mins,
                const map<string,double> &_normalize_maxs,
                const map<string,double> &_normalize_avgs,
                const map<string,double> &_normalize_std_devs);
};

#endif
