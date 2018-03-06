#include <cmath>

#include <fstream>
using std::ifstream;

#include <iomanip>
using std::setw;

#include <iostream>
using std::cout;
using std::cerr;
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
    for (int i = 0; i < values.size(); i++) {
        values[i] = (values[i] - min) / (max - min);
    }
}

void split(const string &s, char delim, vector<string> &result) {
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
        cerr << "ERROR! Trying to add a time series to a time series set with name '" << name << "' which already exists in the set!" << endl;
    }
}

TimeSeriesSet::TimeSeriesSet(string filename, double _expected_class) {
    expected_class = _expected_class;

    ifstream ts_file(filename);

    string line;

    if (!getline(ts_file, line)) {
        cerr << "ERROR! Could not get headers from the CSV file. File potentially empty!" << endl;
        exit(1);
    }
    split(line, ',', fields);
    
    //cout << "number fields: " << fields.size() << endl;

    for (uint32_t i = 0; i < fields.size(); i++) {
        //cout << "\t" << fields[i] << endl;

        add_time_series(fields[i]);
    }

    int row = 1;
    //cout << "values:" << endl;
    while (getline(ts_file, line)) {
        if (line.size() == 0 || line[0] == '#' || row < 100) {
            row++;
            continue;
        }

        vector<string> parts;
        split(line, ',', parts);

        if (parts.size() != fields.size()) {
            cerr << "ERROR! number of values in row " << row << " was " << parts.size() << ", but there were " << fields.size() << " fields in the header." << endl;
            exit(1);
        }

        for (uint32_t i = 0; i < parts.size(); i++) {
            try {
                time_series[ fields[i] ]->add_value( stod(parts[i]) );
            } catch (const invalid_argument& ia) {
                cerr << "file: '" << filename << "' -- invalid argument: '" << ia.what() << "' on row " << row << " and column " << i << ": '" << fields[i] << "', value: '" << parts[i] << "'" << endl;
            }
        }

        row++;
    }

    number_rows = time_series.begin()->second->get_number_values();

    for (auto series = time_series.begin(); series != time_series.end(); series++) {
        series->second->calculate_statistics();
        if (series->second->get_min_change() == 0 && series->second->get_max_change() == 0) {
            //cerr << "removing unchanging series: '" << series->first << "'" << endl;
            cerr << "WARNING: unchanging series: '" << series->first << "'" << endl;
            //series->second->print_statistics(cout);
            //time_series.erase(series);
        } else {
            //series->second->print_statistics(cout);
        }

        int series_rows = series->second->get_number_values();

        if (series_rows != number_rows) {
            cerr << "ERROR! number of rows for field '" << series->first << "' (" << series->second->get_number_values() << ") doesn't equal number of rows in first field '" << time_series.begin()->first << " (" << time_series.begin()->second->get_number_values() << ")" << endl;
        }
    }

}

int TimeSeriesSet::get_number_rows() const {
    return number_rows;
}

int TimeSeriesSet::get_number_columns() const {
    return fields.size();
}


vector<string> TimeSeriesSet::get_fields() const {
    return fields;
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


void normalize_time_series_sets(vector<TimeSeriesSet*> time_series) {
    vector<string> fields = time_series[0]->get_fields();

    for (int i = 1; i < time_series.size(); i++) {
        vector<string> other_fields = time_series[i]->get_fields();

        if (other_fields != fields) {
            cerr << "ERROR! cannot normalize time series sets with different fields!" << endl;
            cerr << "first time series fields:" << endl;
            for (int j = 0; j < other_fields.size(); j++) {
                cerr << "\t" << other_fields[j] << endl;
            }
            cerr << "second time series fields:" << endl;
            for (int j = 0; j < fields.size(); j++) {
                cerr << "\t" << fields[j] << endl;
            }
            exit(1);
        }
    }

    for (int i = 0; i < fields.size(); i++) {
        double min = numeric_limits<double>::max();
        double max = -numeric_limits<double>::max();

        //get the min of all series of the same name
        //get the max of all series of the same name
        for (int j = 0; j < time_series.size(); j++) {
            double current_min = time_series[j]->get_min(fields[i]);
            double current_max = time_series[j]->get_max(fields[i]);

            if (current_min < min) min = current_min;
            if (current_max > max) max = current_max;
        }
        //cout << setw(25) << fields[i] << ", overall min: " << setw(12) << min << ", overall max: " << setw(12) << max << endl;

        //for each series, subtract min, divide by (max - min)
        for (int j = 0; j < time_series.size(); j++) {
            time_series[j]->normalize_min_max(fields[i], min, max);
        }
    }
}

/**
 *  Time offset < 0 generates input data. Do not use the last <time_offset> values
 *  Time offset > 0 generates output data. Do not use the first <time_offset> values
 */
void TimeSeriesSet::export_time_series(vector< vector<double> > &data, const vector<string> &requested_fields, int32_t time_offset) {
    cout << "clearing data" << endl;
    data.clear();
    cout << "resizing to " << requested_fields.size() << " by " << number_rows - fabs(time_offset) << endl;

    data.resize(requested_fields.size(), vector<double>(number_rows - fabs(time_offset), 0.0));

    cout << "resized! time_offset = " << time_offset << endl;

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


double TimeSeriesSet::get_expected_class() const {
    return expected_class;
}

#ifdef TIME_SERIES_TEST

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    vector<string> before_filenames;
    get_argument_vector(arguments, "--before_filenames", true, before_filenames);

    vector<string> after_filenames;
    get_argument_vector(arguments, "--after_filenames", true, after_filenames);

    vector<TimeSeriesSet*> all_time_series;

    int before_rows = 0;
    vector<TimeSeriesSet*> before_time_series;
    cout << "got before time series filenames:" << endl;
    for (uint32_t i = 0; i < before_filenames.size(); i++) {
        cout << "\t" << before_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(before_filenames[i], 1.0);
        before_time_series.push_back( ts );
        all_time_series.push_back( ts );

        cout << "\t\trows: " << ts->get_number_rows() << endl;
        before_rows += ts->get_number_rows();
    }
    cout << "total rows for before flights: " << before_rows << endl;

    int after_rows = 0;
    vector<TimeSeriesSet*> after_time_series;
    cout << "got after time series filenames:" << endl;
    for (uint32_t i = 0; i < after_filenames.size(); i++) {
        cout << "\t" << after_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(after_filenames[i], 0.0);
        after_time_series.push_back( ts );
        all_time_series.push_back( ts );

        cout << "\t\trows: " << ts->get_number_rows() << endl;
        after_rows += ts->get_number_rows();
    }
    cout << "total rows for after flights: " << after_rows << endl;

    normalize_time_series_sets(all_time_series);
}


#endif
