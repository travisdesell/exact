#include <cmath>

#include <algorithm>
using std::find;

#include <fstream>
using std::ifstream;

#include <iomanip>
using std::setw;

#include <iostream>
using std::endl;
using std::ostream;
using std::cout;

#include <limits>
using std::numeric_limits;

#include <sstream>
using std::stringstream;

#include <stdexcept>
using std::invalid_argument;

#include <string>
using std::string;
using std::getline;

#include <vector>
using std::vector;

#include <set>
using std::set;

#include<regex>
using std::regex_replace;

#include "../common/arguments.hxx"
#include "../common/log.hxx"

#include "character_series.hxx"

CharacterSeries::CharacterSeries(string _name) {
	name = _name;
}

void CharacterSeries::add_value(double value) {
	values.push_back(value);
}

double CharacterSeries::get_value(int i) {
	return values[i];
}

void CharacterSeries::calculate_statistics() {
    min = numeric_limits<double>::max();
    min_change = numeric_limits<double>::max();
    average = 0.0;
    max = -numeric_limits<double>::max();
    max_change = -numeric_limits<double>::max();
    std_dev = 0.0;
    variance = 0.0;

    for (uint32_t i = 0; i < values.size(); i++) {
        average += values[i];

        if (min > values[i]) min = values[i];
        if (max < values[i]) max = values[i];
        if (i > 0) {
            double diff = values[i] - values[i-1];
            if (diff < min_change) min_change = diff;
            if (diff > max_change) max_change = diff;
        }
    }
    average /= values.size();
    for (uint32_t i = 0; i < values.size(); i++) {
        double diff = values[i] - average;
        variance += diff * diff;
    }
    variance /= values.size() - 1;
    std_dev = sqrt(variance);
}

void CharacterSeries::print_statistics() {
    Log::info ("\t%25s stats, min: %lf, avg: %lf, max: %lf, min_change: %lf, max_change: %lf, std_dev: %lf, variance: %lf\n", name.c_str(), min, average, max, min_change, max_change, std_dev, variance);
}

int CharacterSeries::get_number_values() const {
    return values.size();
}

double CharacterSeries::get_min() const {
    return min;
}

double CharacterSeries::get_average() const {
    return average;
}

double CharacterSeries::get_max() const {
    return max;
}

double CharacterSeries::get_std_dev() const {
    return std_dev;
}

double CharacterSeries::get_variance() const {
    return variance;
}

double CharacterSeries::get_min_change() const {
    return min_change;
}

double CharacterSeries::get_max_change() const {
    return max_change;
}

void CharacterSeries::normalize_min_max(double min, double max) {
    Log::debug("normalizing time series '%s' with min: %lf and max: %lf, series min: %lf, series max: %lf\n", name.c_str(), min, max, this->min, this->max);

    for (int i = 0; i < values.size(); i++) {
        if (values[i] < min) {
            Log::warning("normalizing series %s, value[%d] %lf was less than min for normalization: %lf\n", name.c_str(), i, values[i], min);
        }
        if (values[i] > max) {
            Log::warning("normalizing series %s, value[%d] %lf was greater than max for normalization: %lf\n", name.c_str(), i, values[i], max);
        }

        values[i] = (values[i] - min) / (max - min);
    }
}

//divide by the normalized max to make things between -1 and 1
void CharacterSeries::normalize_avg_std_dev(double avg, double std_dev, double norm_max) {
    Log::debug("normalizing time series '%s' with avg: %lf, std_dev: %lf and normalized max: %lf, series avg: %lf, series std_dev: %lf\n", name.c_str(), avg, std_dev, norm_max, this->average, this->std_dev);

    for (int i = 0; i < values.size(); i++) {
        values[i] = ((values[i] - avg) / std_dev) / norm_max;
    }
}

void CharacterSeries::cut(int32_t start, int32_t stop) {
    auto first = values.begin() + start;
    auto last = values.begin() + stop;
    values = vector<double>(first, last);
    //update the statistics after the cut
    calculate_statistics();
}

double CharacterSeries::get_correlation(const CharacterSeries *other, int32_t lag) const {
    double other_average = other->get_average();
    int32_t length = fmin(values.size(), other->values.size()) - lag;
    double covariance_sum = 0.0;

    for (int32_t i = 0; i < length; i++) {
        covariance_sum += (values[i + lag] - average) * (other->values[i] - other_average);
    }

    double other_variance = other->get_variance();
    double correlation;
    if (variance < 1e-12 || other_variance < 1e-12) {
        correlation = 0.0;
    } else {
        correlation = (covariance_sum / sqrt(variance * other_variance)) / length;
    }
    return correlation;
}

CharacterSeries::CharacterSeries() {
}

CharacterSeries* CharacterSeries::copy() {
    CharacterSeries *ws = new CharacterSeries();

    ws->name = name;
    ws->min = min;
    ws->average = average;
    ws->max = max;
    ws->std_dev = std_dev;
    ws->variance = variance;
    ws->min_change = min_change;
    ws->max_change = max_change;
    ws->values = values;
    return ws;
}

void CharacterSeries::copy_values(vector<double> &series) {
    series = values;
}

void string_split_character(const string &s, char delim, vector<string> &result) {
    stringstream ss;
    ss.str(s);
    string item;
    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
}

void SentenceSeries::add_character_series(string name) {
    if (character_series.count(name) == 0) {
        character_series[name] = new CharacterSeries(name);
    } else {
        Log::error("ERROR! Trying to add a time series to a time series set with name '%s' which already exists in the set!\n", name.c_str());
    }
}

SentenceSeries::SentenceSeries(const string _line, const vector<string> & _character_index , const map<string,int> &_vocab) {
    sentence = _line;
    character_index = _character_index;
    vocab = _vocab;  

    vector<string> file_characters;
    string_split_character(sentence,' ',file_characters);
    file_characters.push_back("<eos>");
    for(int i  = 0; i < character_index.size(); i++) {
       add_character_series(character_index[i]);
    }
    for (int i = 0; i < file_characters.size(); ++i) {
        int  current_character = vocab[file_characters[i]];
        for (int i = 0; i < character_index.size(); ++i) {
            if(i != current_character) {
                character_series[character_index[i]]->add_value(0);
            } else {
                character_series[character_index[i]]->add_value(1);
            }
        }
    }
    
    number_chars = file_characters.size();

    for (auto series = character_series.begin(); series != character_series.end(); series++) {
        series->second->calculate_statistics();
        if (series->second->get_min_change() == 0 && series->second->get_max_change() == 0) {
            Log::warning("WARNING: unchanging series: '%s'\n", series->first.c_str());
            Log::warning("removing unchanging series: '%s'\n", series->first.c_str());
            series->second->print_statistics();
            //character_series.erase(series);
        } else {
            series->second->print_statistics();
        }
        int series_rows = series->second->get_number_values();
        if (series_rows != number_chars) {
            Log::error("ERROR! number of rows for field '%s' (%d) doesn't equal number of rows in first field '%s' (%d)\n", series->first.c_str(), series->second->get_number_values(), character_series.begin()->first.c_str(), character_series.begin()->second->get_number_values());
        }
    }
}

SentenceSeries::~SentenceSeries() {
    for( std::map<string, CharacterSeries*>::iterator it=character_series.begin(); it!=character_series.end(); it=character_series.begin())
    {
        CharacterSeries* series = it->second;
        character_series.erase (it);
        delete series;
    }
}

int SentenceSeries::get_number_chars() const {
    return number_chars;
}

int SentenceSeries::get_number_columns() const {
    return character_index.size();
}

// string SentenceSeries::get_filename() const {
//     return filename;
// }

vector<string> SentenceSeries::get_character_index() const {
    return character_index;
}

void SentenceSeries::get_series(string character_name, vector<double> &series) {
    character_series[character_name]->copy_values(series);
}

double SentenceSeries::get_min(string character) {
    return character_series[character]->get_min();
}

double SentenceSeries::get_average(string character) {
    return character_series[character]->get_average();
}

double SentenceSeries::get_max(string character) {
    return character_series[character]->get_max();
}

double SentenceSeries::get_std_dev(string character) {
    return character_series[character]->get_std_dev();
}

double SentenceSeries::get_variance(string character) {
    return character_series[character]->get_variance();
}

double SentenceSeries::get_min_change(string character) {
    return character_series[character]->get_min_change();
}

double SentenceSeries::get_max_change(string character) {
    return character_series[character]->get_max_change();
}

void SentenceSeries::normalize_min_max(string character, double min, double max) {
    character_series[character]->normalize_min_max(min, max);
}

void SentenceSeries::normalize_avg_std_dev(string character, double avg, double std_dev, double norm_max) {
    character_series[character]->normalize_avg_std_dev(avg, std_dev, norm_max);
}

double SentenceSeries::get_correlation(string character1, string character2, int32_t lag) const {
    const CharacterSeries *first_series = character_series.at(character1);
    const CharacterSeries *second_series = character_series.at(character2);
    return first_series->get_correlation(second_series, lag);
}

/**
 *   character_offset < 0 generates input data. Do not use the last <character_offset> values
 *   character_offset > 0 generates output data. Do not use the first <character_offset> values
 */
void SentenceSeries::export_character_series(vector< vector<double> > &data, int32_t character_offset) {
    Log::debug("clearing data\n");
    data.clear();

    //for some reason fabs is not working right
    int abs_character_offset = character_offset;
    if (abs_character_offset < 0) abs_character_offset *= -1;
    Log::debug("character offset: %d\n", character_offset);
    Log::debug("abs_character_offset: %d\n", abs_character_offset);

    data.resize(character_index.size(), vector<double>(number_chars - abs_character_offset, 0.0));
    Log::debug("resized! character_offset = %d\n", character_offset);

    // output data
    if(character_offset > 0) { 
        for (int i = 0; i < character_index.size(); i++) {
            for (int j = character_offset; j < number_chars; j++) {
                data[i][j - character_offset] = character_series[character_index[i]]->get_value(j);
            }
        } 
    // input data   
    } else if (character_offset < 0) {
        for (int i = 0; i < character_index.size(); i++){
            for (int j = 0; j < number_chars + character_offset; j++) {
                data[i][j] = character_series[character_index[i]]->get_value(j);
            }
        }
    } else {
        for (int i = 0; i < character_index.size(); i++) {
            for (int j = 0; j < number_chars; j++) {
                data[i][j] = character_series[character_index[i]]->get_value(j);
            }
        }
    }
}

SentenceSeries::SentenceSeries() {

}

SentenceSeries* SentenceSeries::copy() {
    SentenceSeries* ss = new SentenceSeries();
    ss->number_chars = number_chars;
    ss->sentence = sentence;
    ss->character_index = character_index;
    ss->vocab= vocab;

    for (auto series = character_series.begin(); series != character_series.end(); series++) {
        ss->character_series[series->first] = series->second->copy();
    }
    return ss;
}

void SentenceSeries::select_parameters(const vector<string> &parameter_names) {
    for (auto series = character_series.begin(); series != character_series.end(); series++) {
        if (std::find(parameter_names.begin(), parameter_names.end(), series->first) == parameter_names.end()) {
            Log::info("removing series: '%s'\n", series->first.c_str());
            character_series.erase(series->first);
        }
    }
}

void SentenceSeries::select_parameters(const vector<string> &input_parameter_names, const vector<string> &output_parameter_names) {
    vector<string> combined_parameters = input_parameter_names;
    combined_parameters.insert(combined_parameters.end(), output_parameter_names.begin(), output_parameter_names.end());
    select_parameters(combined_parameters);
}

Corpus::Corpus() : normalize_type("none") { 

}

Corpus::~Corpus() {
    for (uint32_t i = 0; i < sentence_series.size(); i++) {
        delete sentence_series[i];
    }
}

void merge_parameter_names_character(const vector<string> &input_parameter_names, const vector<string> &output_parameter_names, vector<string> &all_parameter_names) {
    all_parameter_names.clear();

    for (int32_t i = 0; i < (int32_t)input_parameter_names.size(); i++) {
        if (find(all_parameter_names.begin(), all_parameter_names.end(), input_parameter_names[i]) == all_parameter_names.end()) {
            all_parameter_names.push_back(input_parameter_names[i]);
        }
    }

    for (int32_t i = 0; i < (int32_t)output_parameter_names.size(); i++) {
        if (find(all_parameter_names.begin(), all_parameter_names.end(), output_parameter_names[i]) == all_parameter_names.end()) {
            all_parameter_names.push_back(output_parameter_names[i]);
        }
    }
}

void Corpus::load_character_library() {
    training_sentence_indexes.clear();
    test_sentence_indexes.clear();
    for (int i = 0; i < training_file_indexes.size(); ++i) {
        vector<string> characters;
        string filename = filenames[training_file_indexes[i]];
        ifstream cs_file(filename.c_str());

        for(string line; getline(cs_file,line); ) {
            line = regex_replace(line, std::regex("^ +| +$|( ) +"), "$1");
            string_split_character(line,' ',characters);
            characters.push_back("<eos>");
        }

        set<string> character_index_set(characters.begin(), characters.end());
        for (set<string>::iterator i = character_index_set.begin(); i != character_index_set.end(); ++i) {
            character_index.push_back(*i);
            vocab[*i] = vocab.size();
        }
    }

    input_parameter_names = character_index;
    output_parameter_names = character_index;
    all_parameter_names = character_index;

    int current = 0;
    for (int i = 0; i < training_file_indexes.size(); i++) {
        Log::debug("\t%s\n", filenames[training_file_indexes[i]].c_str());
        ifstream cs_file(filenames[training_file_indexes[i]].c_str());
        for(string line; getline(cs_file,line); ) {
            line = regex_replace(line, std::regex("^ +| +$|( ) +"), "$1");
            SentenceSeries *ss = new SentenceSeries(line, character_index,vocab);
            sentence_series.push_back( ss );
            training_sentence_indexes.push_back(current);
            current++;
        }
    }
    for (int i = 0; i < test_file_indexes.size(); i++) {
        Log::debug("\t%s\n", filenames[test_file_indexes[i]].c_str());
        ifstream cs_file(filenames[test_file_indexes[i]].c_str());
        for(string line; getline(cs_file,line); ) {
            line = regex_replace(line, std::regex("^ +| +$|( ) +"), "$1");
            SentenceSeries *ss = new SentenceSeries(line, character_index,vocab);
            sentence_series.push_back( ss );
            test_sentence_indexes.push_back(current);
            current++;
        }
    }
}

Corpus* Corpus::generate_from_arguments(const vector<string> &arguments) {
	Corpus *cs = new Corpus();
	cs->filenames.clear();
	cs->character_index.clear();
	cs->vocab.clear();

	if (argument_exists(arguments, "--training_filenames") && argument_exists(arguments, "--test_filenames")) {
        vector<string> training_filenames;
        get_argument_vector(arguments, "--training_filenames", true, training_filenames);

        vector<string> test_filenames;
        get_argument_vector(arguments, "--test_filenames", true, test_filenames);

        int current = 0;
        for (int i = 0; i < training_filenames.size(); i++) {
            cs->filenames.push_back(training_filenames[i]);
            cs->training_file_indexes.push_back(current);
            current++;
        }

        for (int i = 0; i < test_filenames.size(); i++) {
            cs->filenames.push_back(test_filenames[i]);
            cs->test_file_indexes.push_back(current);
            current++;
        }

    } else {
        Log::fatal("Could not find the '--filenames' or the '--training_filenames' and '--test_filenames' command line arguments.  Usage instructions:\n");
        //help_message();
        exit(1);
    }

    cs->all_parameter_names.clear();
    cs->input_parameter_names.clear();
    cs->output_parameter_names.clear();
	cs->load_character_library();
    cs->normalize_type = "none";

	return cs;
}

double Corpus::denormalize(string field_name, double value) {
    if (normalize_type.compare("none") == 0) {
        return value;
    } else if (normalize_type.compare("min_max") == 0) {
        double min = normalize_mins[field_name];
        double max = normalize_maxs[field_name];
        value = (value * (max-min)) + min;
        return value;
    } else if (normalize_type.compare("avg_std_dev") == 0) {
        double min = normalize_mins[field_name];
        double max = normalize_maxs[field_name];
        double avg = normalize_avgs[field_name];
        double std_dev = normalize_std_devs[field_name];
        double norm_min = (min - avg) / std_dev;
        double norm_max = (max - avg) / std_dev;
        norm_max = fmax(norm_min, norm_max);
        value = (value * norm_max * std_dev) + avg;
        return value;
    } else {
        Log::fatal("Unknown normalize type on denormalize for '%s' and '%lf', '%s', this should never happen.\n", field_name.c_str(), value, normalize_type.c_str());
        exit(1);
    }
}

void Corpus::normalize_min_max() {
    Log::info("doing min/max normalization:\n");
    for (int i = 0; i < all_parameter_names.size(); i++) {
        string parameter_name = all_parameter_names[i];
        double min = numeric_limits<double>::max();
        double max = -numeric_limits<double>::max();
        //get the min of all series of the same name
        //get the max of all series of the same name

        if (normalize_mins.count(parameter_name) > 0) {
            min = normalize_mins[parameter_name];
            max = normalize_maxs[parameter_name];
            Log::info("user specified bounds for ");
        }  else {
            for (int j = 0; j < sentence_series.size(); j++) {
                double current_min = sentence_series[j]->get_min(parameter_name);
                double current_max = sentence_series[j]->get_max(parameter_name);
                if (current_min < min) min = current_min;
                if (current_max > max) max = current_max;
            }
            normalize_mins[parameter_name] = min;
            normalize_maxs[parameter_name] = max;
            Log::info("calculated bounds for     ");
        }

        Log::info_no_header("%30s, min: %22.10lf, max: %22.10lf\n", parameter_name.c_str(), min, max);
        //for each series, subtract min, divide by (max - min)
        for (int j = 0; j < sentence_series.size(); j++) {
            sentence_series[j]->normalize_min_max(parameter_name, min, max);
        }
    }
    normalize_type = "min_max";
}

void Corpus::normalize_min_max(const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs) {
    normalize_mins = _normalize_mins;
    normalize_maxs = _normalize_maxs;

    for (int32_t i = 0; i < (int32_t)all_parameter_names.size(); i++) {
        string field = all_parameter_names[i];
        if (normalize_mins.count(field) == 0) {
            //field doesn't exist in the normalize values, report an error
            Log::fatal("ERROR, couldn't find field '%s' in normalize min values.\n", field.c_str());
            Log::fatal("normalize min fields/values:\n");
            for (auto iterator = normalize_mins.begin(); iterator != normalize_mins.end(); iterator++) {
                Log::fatal("\t%s: %lf\n", iterator->first.c_str(), iterator->second);
            }
            exit(1);
        }

        if (normalize_maxs.count(field) == 0) {
            //field doesn't exist in the normalize values, report an error
            Log::fatal("ERROR, couldn't find field '%s' in normalize max values.\n", field.c_str());
            Log::fatal("normalize max fields/values:\n");
            for (auto iterator = normalize_maxs.begin(); iterator != normalize_maxs.end(); iterator++) {
                Log::fatal("\t%s: %lf\n", iterator->first.c_str(), iterator->second);
            }
            exit(1);
        }
        //for each series, subtract min, divide by (max - min)
        for (int j = 0; j < sentence_series.size(); j++) {
            sentence_series[j]->normalize_min_max(field, normalize_mins[field], normalize_maxs[field]);
        }
    }
    normalize_type = "min_max";
}

void Corpus::normalize_avg_std_dev() {
    Log::info("doing min/max normalization:\n");

    for (int i = 0; i < all_parameter_names.size(); i++) {
        string parameter_name = all_parameter_names[i];
        double min = numeric_limits<double>::max();
        double max = -numeric_limits<double>::max();
        double avg = 0.0;
        double std_dev = 0.0;
        if (normalize_avgs.count(parameter_name) > 0) {
            min = normalize_mins[parameter_name];
            max = normalize_maxs[parameter_name];
            avg = normalize_avgs[parameter_name];
            std_dev = normalize_std_devs[parameter_name];
            Log::info("user specified avg/std dev for ");
        }  else {
            double numerator_average = 0.0;
            long total_values = 0;
            for (int j = 0; j < sentence_series.size(); j++) {
                int n_values = sentence_series[j]->get_number_chars();
                numerator_average += sentence_series[j]->get_average(parameter_name) * n_values;
                total_values += n_values;
                double current_min = sentence_series[j]->get_min(parameter_name);
                double current_max = sentence_series[j]->get_max(parameter_name);
                if (current_min < min) min = current_min;
                if (current_max > max) max = current_max;
            }
            normalize_mins[parameter_name] = min;
            normalize_maxs[parameter_name] = max;
            avg = numerator_average / total_values;
            double numerator_std_dev = 0.0;
            //get the Bessel-corrected (n-1 denominator) combined standard deviation
            for (int j = 0; j < sentence_series.size(); j++) {
                int n_values = sentence_series[j]->get_number_chars();
                double avg_diff = sentence_series[j]->get_average(parameter_name) - avg;
                numerator_std_dev += ((n_values - 1) * sentence_series[j]->get_variance(parameter_name)) + (n_values * avg_diff * avg_diff);
            }
            std_dev = numerator_std_dev / (total_values - 1);
            normalize_avgs[parameter_name] = avg;
            normalize_std_devs[parameter_name] = std_dev;
            Log::info("calculated bounds for     ");
        }
        
        double norm_min = (min - avg) / std_dev;
        double norm_max = (max - avg) / std_dev;
        norm_max = fmax(norm_min, norm_max);
        Log::info_no_header("%30s, min: %22.10lf, max: %22.10lf, norm_max; %22.10lf, combined average: %22.10lf, combined std_dev: %22.10lf\n", parameter_name.c_str(), min, max, avg, norm_max, std_dev);
        //for each series, subtract min, divide by (max - min)
        for (int j = 0; j < sentence_series.size(); j++) {
            sentence_series[j]->normalize_avg_std_dev(parameter_name, avg, std_dev, norm_max);
        }
    }
    normalize_type = "avg_std_dev";
}

void Corpus::normalize_avg_std_dev(const map<string,double> &_normalize_avgs, const map<string,double> &_normalize_std_devs, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs) {
    normalize_avgs = _normalize_avgs;
    normalize_std_devs = _normalize_std_devs;
    normalize_mins = _normalize_mins;
    normalize_maxs = _normalize_maxs;

    for (int32_t i = 0; i < (int32_t)all_parameter_names.size(); i++) {
        string field = all_parameter_names[i];
        if (normalize_avgs.count(field) == 0) {
            //field doesn't exist in the normalize values, report an error
            Log::fatal("ERROR, couldn't find field '%s' in normalize avg values.\n", field.c_str());
            Log::fatal("normalize avg fields/values:\n");
            for (auto iterator = normalize_avgs.begin(); iterator != normalize_avgs.end(); iterator++) {
                Log::fatal("\t%s: %lf\n", iterator->first.c_str(), iterator->second);
            }
            exit(1);
        }
        if (normalize_std_devs.count(field) == 0) {
            //field doesn't exist in the normalize values, report an error
            Log::fatal("ERROR, couldn't find field '%s' in normalize std_dev values.\n", field.c_str());
            Log::fatal("normalize std_dev fields/values:\n");
            for (auto iterator = normalize_std_devs.begin(); iterator != normalize_std_devs.end(); iterator++) {
                Log::fatal("\t%s: %lf\n", iterator->first.c_str(), iterator->second);
            }
            exit(1);
        }
        if (normalize_mins.count(field) == 0) {
            //field doesn't exist in the normalize values, report an error
            Log::fatal("ERROR, couldn't find field '%s' in normalize min values.\n", field.c_str());
            Log::fatal("normalize min fields/values:\n");
            for (auto iterator = normalize_mins.begin(); iterator != normalize_mins.end(); iterator++) {
                Log::fatal("\t%s: %lf\n", iterator->first.c_str(), iterator->second);
            }
            exit(1);
        }
        if (normalize_maxs.count(field) == 0) {
            //field doesn't exist in the normalize values, report an error
            Log::fatal("ERROR, couldn't find field '%s' in normalize max values.\n", field.c_str());
            Log::fatal("normalize max fields/values:\n");
            for (auto iterator = normalize_maxs.begin(); iterator != normalize_maxs.end(); iterator++) {
                Log::fatal("\t%s: %lf\n", iterator->first.c_str(), iterator->second);
            }
            exit(1);
        }

        double min = normalize_mins[field];
        double max = normalize_maxs[field];
        double avg = normalize_avgs[field];
        double std_dev = normalize_std_devs[field];
        double norm_min = (min - avg) / std_dev;
        double norm_max = (max - avg) / std_dev;
        norm_max = fmax(norm_min, norm_max);
        //for each series, subtract avg, divide by std_dev; then divide by normalized_max to make between -1 and 1
        for (int j = 0; j < sentence_series.size(); j++) {
            sentence_series[j]->normalize_avg_std_dev(field, avg, std_dev, norm_max);
        }
    }
    normalize_type = "avg_std_dev";
}

void Corpus::export_sentence_series(const vector<int> &sentence_indexes, int character_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs) {
    inputs.resize(sentence_indexes.size());
    outputs.resize(sentence_indexes.size());
    
    for (int i = 0; i < sentence_indexes.size(); i++) {
        int sentence_index = sentence_indexes[i];
        sentence_series[sentence_index]->export_character_series(inputs[i], -character_offset);
        sentence_series[sentence_index]->export_character_series(outputs[i], character_offset);
    }
}

/**
 * This exports the time series marked as training series by the training_indexes vector.
 * vector input: < sentence < character index < value over the sentence (onehot)> > >
 */
void Corpus::export_training_series(int character_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs) {
    for (int i = 0; i < training_sentence_indexes.size(); i++) {
        Log::error("training sentence indexes: %d\n", training_sentence_indexes[i]);
    }
    if (training_sentence_indexes.size() == 0) {
        Log::fatal("ERROR: attempting to export training time series, however the training_sentence_indexes were not specified.\n");
        exit(1);
    }
    export_sentence_series(training_sentence_indexes, character_offset, inputs, outputs);
}

/**
 * This exports the time series marked as test series by the test_indexes vector.
 * vector input: < sentence < character index < value over the sentence (onehot)> > >
 */
void Corpus::export_test_series(int character_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs) {
    for (int i = 0; i < test_sentence_indexes.size(); i++) {
        Log::error("test setentence indexes: %d\n", test_sentence_indexes[i]);
    }
    if (test_sentence_indexes.size() == 0) {
        Log::fatal("ERROR: attempting to export test time series, however the test_sentence_indexes were not specified.\n");
        exit(1);
    }
    export_sentence_series(test_sentence_indexes, character_offset, inputs, outputs);
}

void Corpus::export_series_by_name(string field_name, vector< vector<double> > &exported_series) {
    exported_series.clear();

    for (int32_t i = 0; i < sentence_series.size(); i++) {
        vector<double> current_series;
        sentence_series[i]->get_series(field_name, current_series);
        exported_series.push_back(current_series);
    }
}

string Corpus::get_normalize_type() const {
    return normalize_type;
}

map<string,double> Corpus::get_normalize_mins() const {
    return normalize_mins;
}

map<string,double> Corpus::get_normalize_maxs() const {
    return normalize_maxs;
}

map<string,double> Corpus::get_normalize_avgs() const {
    return normalize_avgs;
}

map<string,double> Corpus::get_normalize_std_devs() const {
    return normalize_std_devs;
}


vector<string> Corpus::get_input_parameter_names() const {
    return input_parameter_names;
}

vector<string> Corpus::get_output_parameter_names() const {
    return output_parameter_names;
}

int Corpus::get_number_series() const {
    return sentence_series.size();
}

int Corpus::get_number_inputs() const {
    return input_parameter_names.size();
}

int Corpus::get_number_outputs() const {
    return output_parameter_names.size();
}

void Corpus::set_training_file_indexes(const vector<int> &_training_file_indexes) {
    training_file_indexes = _training_file_indexes;
}

void Corpus::set_test_file_indexes(const vector<int> &_test_file_indexes) {
    test_file_indexes = _test_file_indexes;
}

SentenceSeries* Corpus::get_set(int32_t i) {
    return sentence_series.at(i);
}
