#ifndef EXAMM_TIME_SERIES_HXX
#define EXAMM_TIME_SERIES_HXX

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

    TimeSeries();

   public:
    TimeSeries(string _name);

    void add_value(double value);
    double get_value(int32_t i);

    void calculate_statistics();
    void print_statistics();

    int32_t get_number_values() const;

    double get_min() const;
    double get_average() const;
    double get_max() const;
    double get_std_dev() const;
    double get_variance() const;
    double get_min_change() const;
    double get_max_change() const;

    void normalize_min_max(double min, double max);
    void normalize_avg_std_dev(double avg, double std_dev, double norm_max);

    void cut(int32_t start, int32_t stop);

    double get_correlation(const TimeSeries* other, int32_t lag) const;

    TimeSeries* copy();

    void copy_values(vector<double>& series);
};

class TimeSeriesSet {
   private:
    int32_t number_rows;
    string filename;

    vector<string> fields;

    map<string, TimeSeries*> time_series;

    TimeSeriesSet();

   public:
    TimeSeriesSet(string _filename, const vector<string>& _fields);
    ~TimeSeriesSet();
    void add_time_series(string name);

    int32_t get_number_rows() const;
    int32_t get_number_columns() const;
    string get_filename() const;

    vector<string> get_fields() const;

    void get_series(string field_name, vector<double>& series);

    /** Return the value of a field at a given row (time step). */
    // double get_value(string field_name, int32_t row) const;

    double get_min(string field);
    double get_average(string field);
    double get_max(string field);
    double get_std_dev(string field);
    double get_variance(string field);
    double get_min_change(string field);
    double get_max_change(string field);

    double get_correlation(string field1, string field2, int32_t lag) const;

    void normalize_min_max(string field, double min, double max);
    void normalize_avg_std_dev(string field, double avg, double std_dev, double norm_max);

    void export_time_series(vector<vector<double> >& data);
    void export_time_series(vector<vector<double> >& data, const vector<string>& requested_fields);
    void export_time_series(
        vector<vector<double> >& data, const vector<string>& requested_fields, const vector<string>& shift_fields,
        int32_t time_offset
    );

    TimeSeriesSet* copy();

    void cut(int32_t start, int32_t stop);
    void split(int32_t slices, vector<TimeSeriesSet*>& sub_series);

    void select_parameters(const vector<string>& parameter_names);
    void select_parameters(const vector<string>& input_parameter_names, const vector<string>& output_parameter_names);

    // /** Add a synthetic column (e.g. one-hot class) with given values. Used for in-program one-hot expansion. */
    // void add_synthetic_column(const string& name, const vector<double>& values);
};

class TimeSeriesSets {
   private:
    string normalize_type;

    // /** When true, each CSV row is one independent sample (no time dimension). Export produces one sequence per row, length 1. */
    // bool row_per_sample;

    vector<string> filenames;

    vector<int> training_indexes;
    vector<int> test_indexes;

    vector<string> input_parameter_names;
    vector<string> output_parameter_names;
    vector<string> shift_parameter_names;
    vector<string> all_parameter_names;

    vector<TimeSeriesSet*> time_series;

    map<string, double> normalize_mins;
    map<string, double> normalize_maxs;

    map<string, double> normalize_avgs;
    map<string, double> normalize_std_devs;

    void parse_parameters_string(const vector<string>& p);
    void load_time_series();

   public:
    static void help_message();

    TimeSeriesSets();
    ~TimeSeriesSets();
    static TimeSeriesSets* generate_from_arguments(const vector<string>& arguments);
    static TimeSeriesSets* generate_test(
        const vector<string>& _validation_filenames, const vector<string>& _input_parameter_names,
        const vector<string>& _output_parameter_names
    );

    void normalize_min_max();
    void normalize_min_max(const map<string, double>& _normalize_mins, const map<string, double>& _normalize_maxs);

    void normalize_avg_std_dev();
    void normalize_avg_std_dev(
        const map<string, double>& _normalize_avgs, const map<string, double>& _normalize_std_devs,
        const map<string, double>& _normalize_mins, const map<string, double>& _normalize_maxs
    );

    void split_series(int32_t series, int32_t number_slices);
    void split_all(int32_t number_slices);

    void write_time_series_sets(string base_filename);

    void export_time_series(
        const vector<int>& series_indexes, int32_t time_offset, vector<vector<vector<double> > >& inputs,
        vector<vector<vector<double> > >& outputs
    );

    void export_training_series(
        int32_t time_offset, vector<vector<vector<double> > >& inputs, vector<vector<vector<double> > >& outputs
    );

    void export_test_series(
        int32_t time_offset, vector<vector<vector<double> > >& inputs, vector<vector<vector<double> > >& outputs
    );

    // /**
    //  * Load one file where each row is one time series set (one series).
    //  * Columns: first num_inputs are input params, next num_outputs are output params.
    //  * Output layout matches export_time_series: [series][param_index][timestep]; here timestep length is 1 per row.
    //  */
    // static void load_single_file_row_per_series(
    //     const string& filename, int32_t num_inputs, int32_t num_outputs,
    //     vector<vector<vector<double> > >& inputs, vector<vector<vector<double> > >& outputs, bool skip_header = true
    // );

    void export_series_by_name(string field_name, vector<vector<double> >& exported_series);

    double denormalize(string field_name, double value);

    string get_normalize_type() const;
    map<string, double> get_normalize_mins() const;
    map<string, double> get_normalize_maxs() const;
    map<string, double> get_normalize_avgs() const;
    map<string, double> get_normalize_std_devs() const;

    vector<string> get_input_parameter_names() const;
    vector<string> get_output_parameter_names() const;

    int32_t get_number_series() const;

    int32_t get_number_inputs() const;
    int32_t get_number_outputs() const;

    void set_training_indexes(const vector<int>& _training_indexes);
    void set_test_indexes(const vector<int>& _test_indexes);

    // bool get_row_per_sample() const;

    TimeSeriesSet* get_set(int32_t i);

    /**
     * When prediction_type is categorical and there is exactly one output column, expand it to one-hot
     * (class0, class1, ...) in memory. Pass num_classes (e.g. 2 for binary) or use 0 to infer from data.
     */
    // void expand_single_output_to_onehot(int32_t num_classes);
};

#endif
