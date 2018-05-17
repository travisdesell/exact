#ifndef EXALT_TIME_SERIES_HXX
#define EXALT_TIME_SERIES_HXX

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <map>
using std::map;

#include <vector>
using std::vector;

class TimeSeries {
    private:
        string name;

        double min;
        double average;
        double max;
        double std_dev;
        double variance;
        double min_change;
        double max_change;

        vector<double> values;

    public:
        TimeSeries(string _name);

        void add_value(double value);
        double get_value(int i);

        void calculate_statistics();
        void print_statistics(ostream &out);

        int get_number_values() const;

        double get_min() const;
        double get_average() const;
        double get_max() const;
        double get_std_dev() const;
        double get_variance() const;
        double get_min_change() const;
        double get_max_change() const;

        void normalize_min_max(double min, double max);
};

class TimeSeriesSet {
    private:
        int number_rows;
        double expected_class;
        string filename;

        vector<string> fields;

        map<string, TimeSeries*> time_series;

    public:

        TimeSeriesSet(string _filename, double _expected_class);

        void add_time_series(string name);

        int get_number_rows() const;
        int get_number_columns() const;
        string get_filename() const;

        vector<string> get_fields() const;

        double get_min(string field);
        double get_average(string field);
        double get_max(string field);
        double get_std_dev(string field);
        double get_variance(string field);
        double get_min_change(string field);
        double get_max_change(string field);

        void normalize_min_max(string field, double min, double max);

        void export_time_series(vector< vector<double> > &data);
        void export_time_series(vector< vector<double> > &data, const vector<string> &requested_fields);
        void export_time_series(vector< vector<double> > &data, const vector<string> &requested_fields, int32_t time_offset);

        double get_expected_class() const;
};

void normalize_time_series_sets(vector<TimeSeriesSet*> time_series);

void load_time_series(const vector<string> &training_filenames, const vector<string> &testing_filenames, const vector<string> &input_parameter_names, const vector<string> &output_parameter_names, int time_offset, vector< vector< vector<double> > > &training_inputs, vector< vector< vector<double> > > &training_outputs, vector< vector< vector<double> > > &testing_inputs, vector< vector< vector<double> > > &testing_outputs, bool normalize);

#endif
