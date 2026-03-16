#ifndef EXAMM_TIME_SERIES_NEW_HXX
#define EXAMM_TIME_SERIES_NEW_HXX

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <map>
using std::map;

#include <vector>
using std::vector;

class TimeSeriesNew {
    private:
        int32_t number_rows;
        int32_t number_columns;
        string filename;

        vector<string> parameter_names;
        vector<string> input_parameter_names;
        vector<string> output_parameter_names;

        map<string, vector<double>* > time_series;

    public:
        /**
         * Initializes a (typically multivariate) time series from a file, with user specified
         * input and output parameter names.
         *
         * \param filename is the filename (CSV) to load the time series from. the first line
         *      are the column headers (parameter names), followed by rows of data.
         * \param input_parameter_names are the columns to be used as input data for whatever
         *      model or process is being used.
         * \param output_parameter_names are the columns to be used as output data for whatever
         *      model or process is being used.
         */
        TimeSeriesNew(string filename, const vector<string> &input_parameter_names, const vector<string> &output_parameter_names);

        /**
         * Initializes a (typically multivariate) time series from a file, defaulting to having
         * all columns be used as input and output parameters.
         *
         * \param filename is the filename (CSV) to load the time series from. the first line
         *      are the column headers (parameter names), followed by rows of data.
         */
        TimeSeriesNew(string filename);

        /**
         * Destructor for TimeSeriesNew.
         */
        ~TimeSeriesNew();

        /**
         * Export the input columns into a 2-d vectors (rows x columns) where each
         * row is a step in the time series for the input and output columns. The
         * number of rows will be (number_rows - time_step) so the number of input
         * rows and output rows is the same.
         *
         * \param inputs is the vector to fill in
         * \param outputs the vector to fill in
         * \param output_time_offset is how many time steps in the future the outputs
         *      will be to their corresponding inputs, e.g., an output_time_offset of
         *      1 will set the outputs shifted 1 into the future from the inputs.
         */
        void export_vectors(vector<vector<double> > &inputs, vector<vector<double> > &outputs, int32_t output_time_offset);

        /**
         * Get the input for each input parameter at a given time step.
         *
         * \param time_step is the time (row number) for the values
         * \param inputs is a map of strings (parameter names) to values at 
         *      the given time step.
         */
        void get_inputs_at(int32_t time_step, map<string, double> &inputs);

        /**
         * Get the output for each output parameter at a given time step.
         *
         * \param time_step is the time (row number) for the values
         * \param outputs is a map of strings (parameter names) to values at 
         *      the given time step.
         */
        void get_outputs_at(int32_t time_step, map<string, double> &outputs);


        /**
         * \return the filename used to create the time series.
         */
        string get_filename() const;

        /**
         * \return the number of rows (not counting the headers) in the time series.
         */
        int32_t get_number_rows() const;

        /**
         * \return the total number of columns (parameters) in the time series.
         */ 
        int32_t get_number_columns() const;

        /**
         * \return the number of columns used as input data.
         */
        int32_t get_number_input_columns() const;

        /**
         * \return the number of columns used as output data.
         */
        int32_t get_number_output_columns() const;

        /**
         * \return a vector of all the column headers (parameter names).
         */
        vector<string> get_parameter_names() const;

        /**
         * \return a vector of all the column headers used as input data.
         */
        vector<string> get_input_parameter_names() const;

        /**
         * \return a vector of all the column headers used as output data.
         */
        vector<string> get_output_parameter_names() const;
};

#endif
