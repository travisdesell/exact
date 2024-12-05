#include <algorithm>

#include <cmath>
using std::find;


#include <fstream>
using std::ifstream;

#include <regex>
using std::regex;
using std::regex_match;

#include <sstream>
using std::stringstream;

#include <stdexcept>
using std::invalid_argument;

#include <string>
using std::string;
using std::getline;


#include <vector>
using std::vector;

#include "common/log.hxx"
#include "time_series_new.hxx"


void string_split_new(const string& s, char delim, vector<string>& result) {
    stringstream ss;
    ss.str(s);

    string item;
    while (getline(ss, item, delim)) {
        // get rid of carriage returns (sometimes windows messes this up)
        item.erase(std::remove(item.begin(), item.end(), '\r'), item.end());

        result.push_back(item);
    }
}


/**
 * Initialize the input and output parameter names and then call the default constructor
 * which only takes a filename
 */
TimeSeriesNew::TimeSeriesNew(string _filename, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names) : TimeSeriesNew(_filename) {

    //set the input/output parameter names to those specified by the user
    input_parameter_names = _input_parameter_names;
    output_parameter_names = _output_parameter_names;

    vector<string> regex_input_parameter_names;
    vector<string> regex_output_parameter_names;

    // check to see that all the specified input and output parameter names are in the file
    for (int32_t i = 0; i < (int32_t) input_parameter_names.size(); i++) {
        regex e(input_parameter_names[i]);

        bool found = false;
        for (string parameter_name : parameter_names) {
            Log::trace("checking if '%s' matches regex '%s'\n", parameter_name.c_str(), input_parameter_names[i].c_str());

            if (regex_match(parameter_name, e)) {
                found = true;

                Log::trace("'%s' matched regex '%s'\n", parameter_name.c_str(), input_parameter_names[i].c_str());

                //add this parameter name to the input parameter names if we haven't already
                if (find(input_parameter_names.begin(), input_parameter_names.end(), parameter_name) == input_parameter_names.end()) {
                    Log::info("\tadded '%s' to input parameter names\n", parameter_name.c_str());
                    regex_input_parameter_names.push_back(parameter_name);
                }
            }
        }

        if (!found) {
            // one of the given parameter_names didn't match to any column header in the time series file
            Log::fatal("ERROR: could not find specified input parameter name '%s' in time series file: '%s'\n", input_parameter_names[i].c_str(), filename.c_str() );

            Log::fatal("file's parameter_names:\n");
            for (int32_t j = 0; j < (int32_t) parameter_names.size(); j++) {
                Log::fatal("\t'%s'\n", parameter_names[j].c_str());
            }
            exit(1);
        }
    }

    for (int32_t i = 0; i < (int32_t) output_parameter_names.size(); i++) {
        regex e(output_parameter_names[i]);

        bool found = false;
        for (string parameter_name : parameter_names) {
            Log::trace("checking if '%s' matches regex '%s'\n", parameter_name.c_str(), output_parameter_names[i].c_str());

            if (regex_match(parameter_name, e)) {
                found = true;

                Log::trace("'%s' matched regex '%s'\n", parameter_name.c_str(), output_parameter_names[i].c_str());

                //add this parameter name to the output parameter names if we haven't already
                if (find(output_parameter_names.begin(), output_parameter_names.end(), parameter_name) == output_parameter_names.end()) {
                    Log::info("\tadded '%s' to output parameter names\n", parameter_name.c_str());
                    regex_output_parameter_names.push_back(parameter_name);
                }
            }
        }

        if (!found) {
            // one of the given parameter_names didn't exist in the time series file
            Log::fatal("ERROR: could not find specified output parameter name '%s' in time series file: '%s'\n", output_parameter_names[i].c_str(), filename.c_str() );

            Log::fatal("file's parameter_names:\n");
            for (int32_t j = 0; j < (int32_t) parameter_names.size(); j++) {
                Log::fatal("\t'%s'\n", parameter_names[j].c_str());
            }
            exit(1);
        }
    }

    input_parameter_names = regex_input_parameter_names;
    output_parameter_names = regex_output_parameter_names;
}

TimeSeriesNew::TimeSeriesNew(string _filename) {
    filename = _filename;
    ifstream ts_file(filename);

    string line;
    if (!getline(ts_file, line)) {
        Log::error("ERROR! Could not get headers from the CSV file (first line). File potentially empty!\n");
        exit(1);
    }

    // if the first line is the comment character '#', remove that character from the header
    if (line[0] == '#') {
        line = line.substr(1);
    }

    string_split_new(line, ',', parameter_names);
    Log::info("parameter_names (%d):\n", parameter_names.size());
    for (int32_t i = 0; i < (int32_t)parameter_names.size(); i++) {
        Log::info("\t%s\n", parameter_names[i].c_str());

        time_series[parameter_names[i]] = new vector<double>();
    }

    int32_t row = 1;
    while (getline(ts_file, line)) {
        if (line.size() == 0 || line[0] == '#' || row < 0) {
            row++;
            continue;
        }

        vector<string> parts;
        string_split_new(line, ',', parts);

        if (parts.size() != parameter_names.size()) {
            Log::fatal(
                    "ERROR! number of values in row %d was %d, but there were %d parameter_names in the header.\n", row,
                    parts.size(), parameter_names.size()
                    );
            exit(1);
        }

        for (int32_t i = 0; i < (int32_t) parts.size(); i++) {
            Log::trace("parts[%d]: %s being added to '%s'\n", i, parts[i].c_str(), parameter_names[i].c_str());

            try {
                time_series[parameter_names[i]]->push_back(stod(parts[i]));
            } catch (const invalid_argument& ia) {
                Log::fatal(
                        "file: '%s' -- invalid argument: '%s' on row %d and column %d: '%s', value: '%s'\n",
                        filename.c_str(), ia.what(), row, i, parameter_names[i].c_str(), parts[i].c_str()
                        );
                exit(1);
            }
        }

        row++;
    }
    Log::info("read %d rows.\n", row);

    number_rows = 0;
    number_columns = parameter_names.size();
    int32_t prev_number_rows = 0;
    int32_t column = 0;
    for (auto kv = time_series.begin(); kv != time_series.end(); kv++) {
        number_rows = kv->second->size();
        if (number_rows <= 0) {
            Log::fatal("ERROR, number rows: %d <= 0\n", number_rows);
            exit(1);
        }

        //make sure all columns are the same length
        if (column != 0 && number_rows != prev_number_rows) {
            Log::fatal("ERROR, number rows on column %d (%d) was != number of rows on previous column (%d), all columns must be the same length.", column, number_rows, prev_number_rows);
            exit(1);
        }
    }

    Log::info("read time series file '%s' with %d columns and %d rows.\n", filename.c_str(), number_columns, number_rows);


    // If the input or output parameters are not specified, all are
    // assumed to be used by default.
    input_parameter_names = parameter_names;
    output_parameter_names = parameter_names;
}

TimeSeriesNew::~TimeSeriesNew() {
    // the time series vectors are stored as the same pointers across the time_series,
    // input_time_series, and output_time_series maps
    for (auto it = time_series.begin(); it != time_series.end(); it = time_series.begin()) {
        vector<double>* series = it->second;
        time_series.erase(it);
        delete series;
    }
}

void TimeSeriesNew::export_vectors(vector<vector<double> > &inputs, vector<vector<double> > &outputs, int32_t output_time_offset) {

    inputs.resize(number_rows - output_time_offset, vector<double>(get_number_input_columns()));
    outputs.resize(number_rows - output_time_offset, vector<double>(get_number_output_columns()));

    for (int32_t column = 0; column < input_parameter_names.size(); column++) {
        vector<double> *values = time_series[input_parameter_names[column]];

        for (int32_t row = 0; row < number_rows - output_time_offset; row++) {
            inputs[row][column] = values->at(row);
        }
    }

    for (int32_t column = 0; column < output_parameter_names.size(); column++) {
        vector<double> *values = time_series[output_parameter_names[column]];

        for (int32_t row = 0; row < number_rows - output_time_offset; row++) {
            outputs[row][column] = values->at(row + output_time_offset);
        }
    }
}

void TimeSeriesNew::get_inputs_at(int32_t time_step, map<string, double> &values) {
    values.clear();
    for (string input_parameter_name : input_parameter_names) {
        values[input_parameter_name] = time_series[input_parameter_name]->at(time_step);
    }
}

void TimeSeriesNew::get_outputs_at(int32_t time_step, map<string, double> &values) {
    values.clear();
    for (string output_parameter_name : output_parameter_names) {
        values[output_parameter_name] = time_series[output_parameter_name]->at(time_step);
    }
}



string TimeSeriesNew::get_filename() const {
    return filename;
}

int32_t TimeSeriesNew::get_number_rows() const {
    return number_rows;
}

int32_t TimeSeriesNew::get_number_columns() const {
    return number_columns;
}

int32_t TimeSeriesNew::get_number_input_columns() const {
    return input_parameter_names.size();
}

int32_t TimeSeriesNew::get_number_output_columns() const {
    return output_parameter_names.size();
}

vector<string> TimeSeriesNew::get_parameter_names() const {
    return parameter_names;
}

vector<string> TimeSeriesNew::get_input_parameter_names() const {
    return input_parameter_names;
}

vector<string> TimeSeriesNew::get_output_parameter_names() const {
    return output_parameter_names;
}


