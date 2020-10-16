#ifndef EXAMM_WORD_SERIES_HXX
#define EXAMM_WORD_SERIES_HXX

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <map>
using std::map;

#include <vector>
using std::vector;


 class WordSeries{
 	private:
 		string name;
 		int vocab_size;

 		double min;
        double average;
        double max;
        double std_dev;
        double variance;
        double min_change;
        double max_change;

 		vector<double> values;
 		WordSeries(); 

 	public:
 		WordSeries(string _name);

        void add_value(double value);
        double get_value(int i);

        void calculate_statistics();
        void print_statistics();

        int get_number_values() const;

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

        double get_correlation(const WordSeries *other, int32_t lag) const;

        WordSeries* copy();

        void copy_values(vector<double> &series);

 };

 class SentenceSeries {
	private:
        int number_rows;
        string filename;
      	map<string,int> vocab;
      	vector<string> word_index;
        map <string,WordSeries*> word_series;


        SentenceSeries();
    public:


SentenceSeries(const string _filename,const vector<string> & _word_index , const map<string,int> &_vocab);        ~SentenceSeries();
        void add_word_series(string name);

        int get_number_rows() const;
        int get_number_columns() const;

        string get_filename() const;
        
        vector<string> get_word_index() const;
		

        void get_series(string word_name, vector<double> &series);

        double get_min(string word);
        double get_average(string word);
        double get_max(string word);
        double get_std_dev(string word);
        double get_variance(string word);
        double get_min_change(string word);
        double get_max_change(string word);

        double get_correlation(string word1, string word2, int32_t lag) const;

        void normalize_min_max(string word, double min, double max);
        void normalize_avg_std_dev(string word, double avg, double std_dev, double norm_max);

        void export_word_series(vector< vector<double> > &data , int word_offset);
        void export_word_series(vector< vector<double> > &data );
       
        SentenceSeries* copy();

        void select_parameters(const vector<string> &input_parameter_names, const vector<string> &output_parameter_names);
        void select_parameters(const vector<string> &parameter_names);

};


class Corpus {

	private:

		string normalize_type;

        vector<string> filenames;

        vector<int> training_indexes;
        vector<int> test_indexes;

        vector<string> input_parameter_names;
        vector<string> output_parameter_names;
        vector<string> all_parameter_names;

        vector<SentenceSeries*> sent_series;

        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;

        map<string,double> normalize_avgs;
        map<string,double> normalize_std_devs;

		vector<string> word_index;
		map<string,int> vocab;

		void load_word_library();


	public:

		Corpus();
		~Corpus();

		static Corpus* generate_from_arguments(const vector<string> &arguments);
		static Corpus* generate_test(const vector<string> &_test_filenames, const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names);

        void normalize_min_max();
        void normalize_min_max(const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs);

        void normalize_avg_std_dev();
        void normalize_avg_std_dev(const map<string,double> &_normalize_avgs, const map<string,double> &_normalize_std_devs, const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs);

        void write_sentence_series_sets(string base_filename);

        void export_sent_series(const vector<int> &series_indexes, int word_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs);

        void export_training_series(int word_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs);

        void export_test_series(int word_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs);

        void export_series_by_name(string field_name, vector< vector<double> > &exported_series);

        double denormalize(string field_name, double value);

        string get_normalize_type() const;
        map<string,double> get_normalize_mins() const;
        map<string,double> get_normalize_maxs() const;
        map<string,double> get_normalize_avgs() const;
        map<string,double> get_normalize_std_devs() const;

        vector<string> get_input_parameter_names() const;
        vector<string> get_output_parameter_names() const;

        int get_number_series() const;

        int get_number_inputs() const;
        int get_number_outputs() const;

        void set_training_indexes(const vector<int> &_training_indexes);
        void set_test_indexes(const vector<int> &_test_indexes);

        SentenceSeries *get_set(int32_t i);
};

#endif