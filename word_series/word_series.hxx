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

        /**
         * Specifies the string name in the data.
         * All the unique strings in the datasets will have its each word series.
         */
 		string name;
 		/**
         * Specifies the number of the unique strings in the dataset.
         */
        int vocab_size;

        /**
         *  Specifies the minimum of the word series in a particular string.
         *  This could be 0 or 1
         */
 		double min;
        /**
         *  Specifies the average of the word series in a particular string.
         *  This tells about the words which are used the most.
         */
        double average;
        /**
         *  Specifies the maximum of the word series in a particular string.
         *  This could be 0 or 1
         */
        double max;
        /**
         *  Specifies the std. deviation  of the word series in a particular string.
         */
        double std_dev;
        /**
         *  Specifies the variance of the word series in a particular string.
         *  This tells about the closeness of a word in the dataset. 
         */
        double variance;
        double min_change;
        double max_change;

        /**
         *  Storers the values in the word series respective of the time in a particular string.
         */
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

        /**
         *  Specifies the number of the rows in the dataset.
         *  This tells about the number of the unique words in the dataset. 
         */
        int number_rows;
        /**
         *  Specifices the filename for the training or testing data. 
         */
        string filename;
        /**
         *  Maps the word string to index of the word series in a particular string. 
         */
      	map<string,int> vocab;
        /**
         *  Maps the index to word string of the word series in a particular string.
         */
      	vector<string> word_index;
        /**
         *  Maps the word string to the word series in a particular string.
         */
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

        /**
         *  Specifices the batches in which the dataset is to be divided into chunks.
         */
        uint32_t batch_size;
		string normalize_type;

        /**
         *  Stores all the filenames including training and testing 
         */
        vector<string> filenames;

        /**
         *  stores the training index of the filenames. 
         */
        vector<int> training_indexes;
        
        /**
         *  stores the testing index of the filenames. 
         */        
        vector<int> test_indexes;

        /**
         *  stores the input parameter names of the filenames.
         *  stores the unique words used in the filenames. 
         */          
        vector<string> input_parameter_names;
        /**
         *  stores the input parameter names of the filenames.
         *  stores the unique words used in the filenames. 
         */
        vector<string> output_parameter_names;
        /**
         *  stores the input parameter names of the filenames.
         *  stores the unique words used in the filenames. 
         */
        vector<string> all_parameter_names;

        /**
         *  stores the Sentence series used in the filenames.
         * Each file is divided into sentences and word series is made for each word in the sentence. 
         */
        vector<SentenceSeries*> sent_series;

        map<string,double> normalize_mins;
        map<string,double> normalize_maxs;

        map<string,double> normalize_avgs;
        map<string,double> normalize_std_devs;


        /**
         *  Maps the index to word string of the word series in a particular filename.
         */
		vector<string> word_index;
		
        /**
         *  Maps the word string to the word series in a particular filename.
         */
        map<string,int> vocab;

        /**
        * Loads the filenames, word index and the vocabulary of the files.
        */
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


        /**
         *  Generates the csv data file. 
         * \param base_filename is the filename to be used.
         *
         */
        void write_sentence_series_sets(string base_filename);

        void export_sent_series(const vector<int> &series_indexes, int word_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs);

        /**
         * Exports Corpus Dataset to be used for training in  RNN genome.
         *
         * \param series_indexes are the indexes of the input parameters to be used in the dataset
         * \param word_offset is used to predict the after how many words, the predictions to be done in the future. 
         * \param inputs is the data used for training in the RNN genome
         * \param outpus is the data used for training in the RNN genome
         *
         */
        void export_training_series(int word_offset, vector< vector< vector<double> > > &inputs, vector< vector< vector<double> > > &outputs);

        /**
         * Exports Corpus Dataset to be used for testing in RNN genome.
         *
         * \param series_indexes are the indexes of the input parameters to be used in the dataset
         * \param word_offset is used to predict the after how many words, the predictions to be done in the future. 
         * \param inputs is the data used for testing in the RNN genome
         * \param outpus is the data used for testing in the RNN genome
         *
         */
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