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

#include <limits>
using std::numeric_limits;

#include <sstream>
using std::stringstream;

#include <stdexcept>
using std::invalid_argument;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "time_series.hxx"

TimeSeries::TimeSeries(string _name) {
    name = _name;
}

void TimeSeries::add_value(double value) {
    values.push_back(value);
}

double TimeSeries::get_value(int i) {
    return values[i];
}

void TimeSeries::calculate_statistics() {
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

/*
void TimeSeries::print_statistics(ostream &out) {
    out << "\t" << setw(25) << name << " stats";
    out << ", min: " << setw(13) << min;
    out << ", avg: " << setw(13) << average;
    out << ", max: " << setw(13) << max;
    out << ", min_change: " << setw(13) << min_change;
    out << ", max_change: " << setw(13) << max_change;
    out << ", std_dev: " << setw(13) << std_dev;
    out << ", variance: " << setw(13) << variance << endl;
}
*/

int TimeSeries::get_number_values() const {
    return values.size();
}

double TimeSeries::get_min() const {
    return min;
}

double TimeSeries::get_average() const {
    return average;
}

double TimeSeries::get_max() const {
    return max;
}

double TimeSeries::get_std_dev() const {
    return std_dev;
}

double TimeSeries::get_variance() const {
    return variance;
}

double TimeSeries::get_min_change() const {
    return min_change;
}

double TimeSeries::get_max_change() const {
    return max_change;
}

void TimeSeries::normalize_min_max(double min, double max) {
    Log::debug("normalizing time series '%s' with min: %lf and %lf, series min: %lf, series max: %lf\n", name.c_str(), min, max, this->min, this->max);

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

void TimeSeries::cut(int32_t start, int32_t stop) {
    auto first = values.begin() + start;
    auto last = values.begin() + stop;
    values = vector<double>(first, last);

    //update the statistics after the cut
    calculate_statistics();
}

TimeSeries::TimeSeries() {
}

TimeSeries* TimeSeries::copy() {
    TimeSeries *ts = new TimeSeries();

    ts->name = name;
    ts->min = min;
    ts->average = average;
    ts->max = max;
    ts->std_dev = std_dev;
    ts->variance = variance;
    ts->min_change = min_change;
    ts->max_change = max_change;

    ts->values = values;

    return ts;
}

void TimeSeries::copy_values(vector<double> &series) {
    series = values;
}


void string_split(const string &s, char delim, vector<string> &result) {
    stringstream ss;
    ss.str(s);

    string item;
    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
}

void TimeSeriesSet::add_time_series(string name) {
    if (time_series.count(name) == 0) {
        time_series[name] = new TimeSeries(name);
    } else {
        Log::error("ERROR! Trying to add a time series to a time series set with name '%s' which already exists in the set!\n", name.c_str());
    }
}

TimeSeriesSet::TimeSeriesSet(string _filename, const vector<string> &_fields) {
    filename = _filename;
    fields = _fields;

    ifstream ts_file(filename);

    string line;

    if (!getline(ts_file, line)) {
        Log::error("ERROR! Could not get headers from the CSV file. File potentially empty!\n");
        exit(1);
    }

    vector<string> file_fields;
    string_split(line, ',', file_fields);

    //check to see that all the specified fields are in the file
    for (int32_t i = 0; i < (int32_t)fields.size(); i++) {
        if (find(file_fields.begin(), file_fields.end(), fields[i]) == file_fields.end()) {
            //one of the given fields didn't exist in the time series file
            Log::fatal("ERROR: could not find specified field '%s' in time series file: '%s'\n", fields[i].c_str(), _filename.c_str());
            Log::fatal("file's fields:\n");
            for (int32_t j = 0; j < (int32_t)file_fields.size(); j++) {
                Log::fatal("'%s'\n", file_fields[j].c_str());
            }
            exit(1);
        }
     }

    Log::debug("fields.size(): %d, file_fields.size(): %d\n", fields.size(), file_fields.size());

    //specify which of the file fields (columns) are used
    vector<bool> file_fields_used(true, file_fields.size());
    for (int32_t i = 0; i < (int32_t)file_fields.size(); i++) {

        Log::debug("\tchecking to see if '%s' was in specified fields, file_fields_used[%d]: %d", file_fields[i].c_str(), i, file_fields_used[i]);

        if (find(fields.begin(), fields.end(), file_fields[i]) == fields.end()) {
            //the ith file field wasn't found in the specified fields
            Log::debug_no_header(" -- field was not found!\n");
            file_fields_used[i] = false;
        } else {
            Log::debug_no_header(" -- Field was found!\n");
            file_fields_used[i] = true;
        }
    }

    Log::debug("number fields: %d\n", fields.size());
    for (uint32_t i = 0; i < fields.size(); i++) {
        Log::debug("\t%s used: %d\n", fields[i].c_str(), file_fields_used[i]);

        add_time_series(fields[i]);
    }

    int row = 1;
    while (getline(ts_file, line)) {
        if (line.size() == 0 || line[0] == '#' || row < 0) {
            row++;
            continue;
        }

        vector<string> parts;
        string_split(line, ',', parts);

        if (parts.size() != file_fields.size()) {
            Log::fatal("ERROR! number of values in row %d was %d, but there were %d fields in the header.\n", row, parts.size(), file_fields.size());
            exit(1);
        }

        for (uint32_t i = 0; i < parts.size(); i++) {
            if (!file_fields_used[i]) continue;

            Log::trace("parts[%d]: %s being added to '%s'\n", i, parts[i].c_str(), file_fields[i].c_str());

            try {
                time_series[ file_fields[i] ]->add_value( stod(parts[i]) );
            } catch (const invalid_argument& ia) {
                Log::error("file: '%s' -- invalid argument: '%s' on row %d and column %d: '%s', value: '%s'\n", filename.c_str(), ia.what(), row, i, file_fields[i].c_str(), parts[i].c_str());
            }
        }

        row++;
    }

    number_rows = time_series.begin()->second->get_number_values();

    for (auto series = time_series.begin(); series != time_series.end(); series++) {
        series->second->calculate_statistics();
        if (series->second->get_min_change() == 0 && series->second->get_max_change() == 0) {
            Log::warning("WARNING: unchanging series: '%s'\n", series->first.c_str());
            //Log::warning("removing unchanging series: '%s'\n", series->first.c_str());
            //series->second->print_statistics(cout);
            //time_series.erase(series);
        } else {
            //series->second->print_statistics(cout);
        }

        int series_rows = series->second->get_number_values();

        if (series_rows != number_rows) {
            Log::error("ERROR! number of rows for field '%s' (%d) doesn't equal number of rows in first field '%s' (%d)\n", series->first.c_str(), series->second->get_number_values(), time_series.begin()->first.c_str(), time_series.begin()->second->get_number_values());
        }
    }

    Log::debug("read time series '%s' with number rows: %d\n", filename.c_str(), number_rows);
}

int TimeSeriesSet::get_number_rows() const {
    return number_rows;
}

int TimeSeriesSet::get_number_columns() const {
    return fields.size();
}

string TimeSeriesSet::get_filename() const {
    return filename;
}

vector<string> TimeSeriesSet::get_fields() const {
    return fields;
}

void TimeSeriesSet::get_series(string field_name, vector<double> &series) {
    time_series[field_name]->copy_values(series);
}

double TimeSeriesSet::get_min(string field) {
    return time_series[field]->get_min();
}

double TimeSeriesSet::get_average(string field) {
    return time_series[field]->get_average();
}

double TimeSeriesSet::get_max(string field) {
    return time_series[field]->get_max();
}

double TimeSeriesSet::get_std_dev(string field) {
    return time_series[field]->get_std_dev();
}

double TimeSeriesSet::get_variance(string field) {
    return time_series[field]->get_variance();
}

double TimeSeriesSet::get_min_change(string field) {
    return time_series[field]->get_min_change();
}

double TimeSeriesSet::get_max_change(string field) {
    return time_series[field]->get_max_change();
}

void TimeSeriesSet::normalize_min_max(string field, double min, double max) {
    time_series[field]->normalize_min_max(min, max);
}



/**
 *  Time offset < 0 generates input data. Do not use the last <time_offset> values
 *  Time offset > 0 generates output data. Do not use the first <time_offset> values
 */
void TimeSeriesSet::export_time_series(vector< vector<double> > &data, const vector<string> &requested_fields, int32_t time_offset) {
    Log::debug("clearing data\n");
    data.clear();
    Log::debug("resizing '%s' to %d by %d\n", filename.c_str(), requested_fields.size(), number_rows - fabs(time_offset));

    data.resize(requested_fields.size(), vector<double>(number_rows - fabs(time_offset), 0.0));

    Log::debug("resized! time_offset = %d\n", time_offset);

    if (time_offset == 0) {
        for (int i = 0; i != requested_fields.size(); i++) {
            for (int j = 0; j < number_rows; j++) {
                data[i][j] = time_series[ requested_fields[i] ]->get_value(j);
            }
        }

    } else if (time_offset < 0) {
        //input data, ignore the last N values
        for (int i = 0; i != requested_fields.size(); i++) {
            for (int j = 0; j < number_rows + time_offset; j++) {
                data[i][j] = time_series[ requested_fields[i] ]->get_value(j);
            }
        }

    } else if (time_offset > 0) {
        //output data, ignore the first N values
        for (int i = 0; i != requested_fields.size(); i++) {
            for (int j = time_offset; j < number_rows; j++) {
                data[i][j - time_offset] = time_series[ requested_fields[i] ]->get_value(j);
            }
        }

    }
}

void TimeSeriesSet::export_time_series(vector< vector<double> > &data, const vector<string> &requested_fields) {
    export_time_series(data, requested_fields, 0);
}

void TimeSeriesSet::export_time_series(vector< vector<double> > &data) {
    export_time_series(data, fields, 0);
}

TimeSeriesSet::TimeSeriesSet() {
}

TimeSeriesSet* TimeSeriesSet::copy() {
    TimeSeriesSet *tss = new TimeSeriesSet();

    tss->number_rows = number_rows;
    tss->filename = filename;
    tss->fields = fields;
    
    for (auto series = time_series.begin(); series != time_series.end(); series++) {
        tss->time_series[series->first] = series->second->copy();
    }

    return tss;
}

void TimeSeriesSet::cut(int32_t start, int32_t stop) {
    for (auto series = time_series.begin(); series != time_series.end(); series++) {
        series->second->cut(start, stop);
    }
    number_rows = stop - start;
}

void TimeSeriesSet::split(int slices, vector<TimeSeriesSet*> &sub_series) {
    sub_series.clear();

    double start = 0;
    double slice_size = (double)number_rows / (double)slices;
    double stop = slice_size;

    for (int32_t i = 0; i < slices; i++) {
        if (i == (slices - 1)) {
            stop = number_rows;
        }

        TimeSeriesSet *slice = this->copy();
        slice->filename = filename + "_split_" + to_string(i);

        slice->cut((int32_t)start, (int32_t)stop);
        sub_series.push_back( this->copy() );

        Log::info("split series from time %d to %d\n",  (int32_t)start, (int32_t)stop);

        start += slice_size;
        stop += slice_size;
    }
}

void TimeSeriesSet::select_parameters(const vector<string> &parameter_names) {
    for (auto series = time_series.begin(); series != time_series.end(); series++) {
        if (std::find(parameter_names.begin(), parameter_names.end(), series->first) == parameter_names.end()) {
            Log::info("removing series: '%s'\n", series->first.c_str());
            time_series.erase(series->first);
        }
    }
}

void TimeSeriesSet::select_parameters(const vector<string> &input_parameter_names, const vector<string> &output_parameter_names) {
    vector<string> combined_parameters = input_parameter_names;
    combined_parameters.insert(combined_parameters.end(), output_parameter_names.begin(), output_parameter_names.end());

    select_parameters(combined_parameters);
}

void TimeSeriesSets::help_message() {
    Log::info("TimeSeriesSets initialization options from arguments:\n");
    Log::info("\tFile input:\n");
    Log::info("\t\t\t--filenames <filenames>* : list of input CSV files\n");
    Log::info("\t\tWith the following are optional unless you want to split the data into training/testing sets:\n");
    Log::info("\t\t\t--training_indexes : array of ints (starting at 0) specifying which files are training files\n");
    Log::info("\t\t\t--test_indexes : array of ints (starting at 0) specifying which files are test files\n");
    Log::info("\tOR:\n");
    Log::info("\t\t\t--training_filenames : list of input CSV files for training time series\n");
    Log::info("\t\t\t--test_filenames : list of input CSV files for test time series\n");

    Log::info("\tSpecifying parameters:\n");
    Log::info("\t\t\t--input_parameter_names <name>*: parameters to be used as inputs\n");
    Log::info("\t\t\t--output_parameter_names <name>*: parameters to be used as outputs\n");
    Log::info("\t\tOR:\n");
    Log::info("\t\t\t --parameters <name setting [min_bound max_bound]>* : list of parameters, with a settings string and potentially bounds\n");
    Log::info("\t\t\t\tThe settings string should consist of only the characters 'i', 'o', and 'b'.\n");
    Log::info("\t\t\t\t'i' denotes the parameter as an input parameter.\n");
    Log::info("\t\t\t\t'o' denoting the parameter as an output parameter.\n");
    Log::info("\t\t\t\t'b' denoting the parameter as having user specified bounds, if this is specified the following two values should be the min and max bounds for the parameter.\n");
    Log::info("\t\t\t\tThe settings string requires at one of 'i' or 'o'.\n");

    Log::info("\tNormalization:\n");
    Log::info("\t\t--normalize : normalize the data. data will be normalized between user specified bounds if given, otherwise the min and max values for a parameter will be calculated over all input files.\n");
}

TimeSeriesSets::TimeSeriesSets() : normalized(false) {
}

void merge_parameter_names(const vector<string> &input_parameter_names, const vector<string> &output_parameter_names, vector<string> &all_parameter_names) {
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

void TimeSeriesSets::parse_parameters_string(const vector<string> &p) {
    for (auto i = p.begin(); i != p.end();) {
        string parameter = *i;
        i++;
        string settings = *i;
        i++;

        if (settings.find_first_not_of("iob")) {
            Log::fatal("Settings string for parameter '%s' was invalid, should consist only of characters 'i', 'o', or 'b'; i : input, o : output, b : bounded.\n", parameter.c_str());
            help_message();
            exit(1);
        }

        bool has_input = false;
        bool has_output = false;
        bool has_bounds = false;
        double min_bound = 0.0, max_bound = 0.0;

        all_parameter_names.push_back(parameter);
        if (settings.find('i') != string::npos) {
            input_parameter_names.push_back(parameter);
        }

        if (settings.find('o') != string::npos) {
            output_parameter_names.push_back(parameter);
        }

        if (settings.find('b') != string::npos) {
            string min_bound_s = *i;
            i++;
            string max_bound_s = *i;
            i++;

            min_bound = stod(min_bound_s);
            max_bound = stod(min_bound_s);
            normalize_mins[parameter] = min_bound;
            normalize_maxs[parameter] = max_bound;
        }

        if (!has_input && !has_output) {
            Log::fatal("Settings string for parameter '%s' was invalid, did not contain an 'i' for input or 'o' for output.\n", parameter.c_str());
            help_message();
            exit(1);
        }

        Log::info("parsed parameter '%s' as ");
        if (has_input) Log::info_no_header("input");
        if (has_output && has_input) Log::info_no_header(", ");
        if (has_output) Log::info_no_header("output");
        if (has_bounds) Log::info_no_header(", min_bound: %lf, max_bound: %lf", min_bound, max_bound);
        Log::info_no_header("\n");
    }
}


void TimeSeriesSets::load_time_series() {
    int32_t rows = 0;
    time_series.clear();
    if (Log::at_level(Log::DEBUG)) {
        Log::debug("loading time series with parameters:");

        for (uint32_t i = 0; i < all_parameter_names.size(); i++) {
            Log::debug("\t'%s'\n", all_parameter_names[i].c_str());
        }
        Log::debug("got time series filenames:\n");
    }

    for (uint32_t i = 0; i < filenames.size(); i++) {
        Log::debug("\t%s", filenames[i].c_str());

        TimeSeriesSet *ts = new TimeSeriesSet(filenames[i], all_parameter_names);
        time_series.push_back( ts );

        rows += ts->get_number_rows();
    }
    Log::debug("number of time series files: %d, total rows: %d\n", filenames.size(), rows);
}


TimeSeriesSets* TimeSeriesSets::generate_from_arguments(const vector<string> &arguments) {
    TimeSeriesSets *tss = new TimeSeriesSets();

    tss->filenames.clear();

    if (argument_exists(arguments, "--filenames")) {
        get_argument_vector(arguments, "--filenames", true, tss->filenames);
        get_argument_vector(arguments, "--training_indexes", false, tss->training_indexes);
        get_argument_vector(arguments, "--test_indexes", false, tss->test_indexes);

    } else if (argument_exists(arguments, "--training_filenames") && argument_exists(arguments, "--test_filenames")) {
        vector<string> training_filenames;
        get_argument_vector(arguments, "--training_filenames", true, training_filenames);

        vector<string> test_filenames;
        get_argument_vector(arguments, "--test_filenames", true, test_filenames);

        int current = 0;
        for (int i = 0; i < training_filenames.size(); i++) {
            tss->filenames.push_back(training_filenames[i]);
            tss->training_indexes.push_back(current);
            current++;
        }

        for (int i = 0; i < test_filenames.size(); i++) {
            tss->filenames.push_back(test_filenames[i]);
            tss->test_indexes.push_back(current);
            current++;
        }

    } else {
        Log::fatal("Could not find the '--filenames' or the '--training_filenames' and '--test_filenames' command line arguments.  Usage instructions:\n");
        help_message();
        exit(1);
    }

    tss->all_parameter_names.clear();
    tss->input_parameter_names.clear();
    tss->output_parameter_names.clear();
    tss->normalize_mins.clear();
    tss->normalize_maxs.clear();

    if (argument_exists(arguments, "--parameters")) {
        vector<string> p;
        get_argument_vector(arguments, "--parameters", true, p);
        tss->parse_parameters_string(p);

    } else if (argument_exists(arguments, "--input_parameter_names") && argument_exists(arguments, "--output_parameter_names")) {
        get_argument_vector(arguments, "--input_parameter_names", true, tss->input_parameter_names);

        get_argument_vector(arguments, "--output_parameter_names", true, tss->output_parameter_names);

        merge_parameter_names(tss->input_parameter_names, tss->output_parameter_names, tss->all_parameter_names);

        if (Log::at_level(Log::DEBUG)) {
            Log::debug("input parameter names:\n");
            for (int i = 0; i < tss->input_parameter_names.size(); i++) {
                Log::debug("\t%s\n", tss->input_parameter_names[i].c_str());
            }

            Log::debug("output parameter names:\n");
            for (int i = 0; i < tss->output_parameter_names.size(); i++) {
                Log::debug("\t%s\n", tss->output_parameter_names[i].c_str());
            }

            Log::debug("all parameter names:\n");
            for (int i = 0; i < tss->all_parameter_names.size(); i++) {
                Log::debug("\t%s\n", tss->all_parameter_names[i].c_str());
            }
        }
    } else {
        Log::fatal("Could not find the '--parameters' or the '--input_parameter_names' and '--output_parameter_names' command line arguments.  Usage instructions:\n");
        help_message();
        exit(1);
    }


    tss->load_time_series();

    bool _normalize = argument_exists(arguments, "--normalize");

    if (_normalize) {
        tss->normalized = true;

        tss->normalize();
        Log::debug("normalized all time series.\n");
    } else {
        tss->normalized = false;
        Log::debug("not normalizing time series.\n");
    }

    return tss;
}

TimeSeriesSets* TimeSeriesSets::generate_test(const vector<string> &_test_filenames, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names) {
    TimeSeriesSets *tss = new TimeSeriesSets();

    tss->filenames = _test_filenames;

    tss->training_indexes.clear();
    tss->test_indexes.clear();
    for (int32_t i = 0; i < (int32_t)tss->filenames.size(); i++) {
        tss->test_indexes.push_back(i);
    }
        
    tss->input_parameter_names = _input_parameter_names;
    tss->output_parameter_names = _output_parameter_names;
    merge_parameter_names(tss->input_parameter_names, tss->output_parameter_names, tss->all_parameter_names);

    tss->normalize_mins.clear();
    tss->normalize_maxs.clear();

    tss->load_time_series();

    return tss;
}

void TimeSeriesSets::normalize() {
    Log::debug("normalizing:\n");

    for (int i = 0; i < all_parameter_names.size(); i++) {
        string parameter_name = all_parameter_names[i];

        double min = numeric_limits<double>::max();
        double max = -numeric_limits<double>::max();

        //get the min of all series of the same name
        //get the max of all series of the same name

        if (normalize_mins.count(parameter_name) > 0) {
            min = normalize_mins[parameter_name];
            max = normalize_maxs[parameter_name];

            Log::debug("user specified bounds for ");

        }  else {
            for (int j = 0; j < time_series.size(); j++) {
                double current_min = time_series[j]->get_min(parameter_name);
                double current_max = time_series[j]->get_max(parameter_name);

                if (current_min < min) min = current_min;
                if (current_max > max) max = current_max;
            }

            normalize_mins[parameter_name] = min;
            normalize_maxs[parameter_name] = max;

            Log::debug("calculated bounds for     ");
        }

        Log::debug_no_header("%30s, min: %5.10lf, max: %5.10lf\n", parameter_name.c_str(), min, max);

        //for each series, subtract min, divide by (max - min)
        for (int j = 0; j < time_series.size(); j++) {
            time_series[j]->normalize_min_max(parameter_name, min, max);
        }
    }
}

void TimeSeriesSets::normalize(const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs) {
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
        for (int j = 0; j < time_series.size(); j++) {
            time_series[j]->normalize_min_max(field, normalize_mins[field], normalize_maxs[field]);
        }
    }

    normalized = true;
}

/**
 * the series argument is a vector of indexes of the time series that was 
 * initially loaded that are to be exported
 */
void TimeSeriesSets::export_time_series(const vector<int> &series_indexes, int time_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs) {
    inputs.resize(series_indexes.size());
    outputs.resize(series_indexes.size());

    for (uint32_t i = 0; i < series_indexes.size(); i++) {
        int series_index = series_indexes[i];

        time_series[series_index]->export_time_series(inputs[i], input_parameter_names, -time_offset);
        time_series[series_index]->export_time_series(outputs[i], output_parameter_names, time_offset);
    }
}

/**
 * This exports the time series marked as training series by the training_indexes vector.
 */
void TimeSeriesSets::export_training_series(int time_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs) {
    if (training_indexes.size() == 0) {
        Log::fatal("ERROR: attempting to export training time series, however the training_indexes were not specified.\n");
        exit(1);
    }

    export_time_series(training_indexes, time_offset, inputs, outputs);
}

/**
 * This exports the time series marked as test series by the test_indexes vector.
 */
void TimeSeriesSets::export_test_series(int time_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs) {
    if (test_indexes.size() == 0) {
        Log::fatal("ERROR: attempting to export test time series, however the test_indexes were not specified.\n");
        exit(1);
    }

    export_time_series(test_indexes, time_offset, inputs, outputs);
}


/**
 * This exports from all the loaded time series a particular column
 */
void TimeSeriesSets::export_series_by_name(string field_name, vector< vector<double> > &exported_series) {
    exported_series.clear();

    for (int32_t i = 0; i < time_series.size(); i++) {
        vector<double> current_series;

        time_series[i]->get_series(field_name, current_series);
        exported_series.push_back(current_series);
    }
}


void TimeSeriesSets::write_time_series_sets(string base_filename) {
    for (uint32_t i = 0; i < time_series.size(); i++) {
        ofstream outfile(base_filename + to_string(i) + ".csv");

        vector< vector<double> > data;
        time_series[i]->export_time_series(data);

        for (int j = 0; j < all_parameter_names.size(); j++) {
            if (j > 0) {
                outfile << ",";
            }
            outfile << all_parameter_names[j];
        }
        outfile << endl;

        for (int j = 0; j < data[0].size(); j++) {
            for (int k = 0; k < data.size(); k++) {
                if (k > 0) {
                    outfile << ",";
                }

                outfile << data[k][j];
            }
            outfile << endl;
        }
    }
}

void TimeSeriesSets::split_series(int series, int number_slices) {
    TimeSeriesSet *ts = time_series[series];

    vector<TimeSeriesSet*> sub_series;
    ts->split(number_slices, sub_series);

    time_series.erase(time_series.begin() + series);
    time_series.insert(time_series.begin() + series, sub_series.begin(), sub_series.end());
}

void TimeSeriesSets::split_all(int number_slices) {
    for (int i = time_series.size() - 1; i >= 0; i++) {
        split_series(i, number_slices);
    }
}

map<string,double> TimeSeriesSets::get_normalize_mins() const {
    return normalize_mins;
}

map<string,double> TimeSeriesSets::get_normalize_maxs() const {
    return normalize_maxs;
}

vector<string> TimeSeriesSets::get_input_parameter_names() const {
    return input_parameter_names;
}

vector<string> TimeSeriesSets::get_output_parameter_names() const {
    return output_parameter_names;
}

int TimeSeriesSets::get_number_series() const {
    return time_series.size();
}

int TimeSeriesSets::get_number_inputs() const {
    return input_parameter_names.size();
}

int TimeSeriesSets::get_number_outputs() const {
    return output_parameter_names.size();
}

void TimeSeriesSets::set_training_indexes(const vector<int> &_training_indexes) {
    training_indexes = _training_indexes;
}

void TimeSeriesSets::set_test_indexes(const vector<int> &_test_indexes) {
    test_indexes = _test_indexes;
}
