#include <iostream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <tinyneat.hpp>
#include <tinyann.hpp>
#include <math.h>

#include <vector>
using std::vector;

using namespace std ;

#include "common/arguments.hxx"
#include "time_series/time_series.hxx"

vector<string> arguments;
vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;


// returns the fitness.
vector<double> evaluate(ann::neuralnet& n, vector< vector<double> > training_inputs,
	 						vector< vector<double> > training_outputs){
	std::vector<double> input;
	std::vector<double> output (1, 0.0);
	double mse = 0.0;
	double mae = 0.0;
	double answer;

	for (int j=0; j<training_inputs[0].size(); j++) {
		answer = training_outputs[0][j] ;
		for (int i=0; i<training_inputs.size(); i++) {
			input.push_back( training_inputs[i][j] ) ;
		}
		n.evaluate(input, output);
		input.clear() ;
		mse += (answer - output[0]) * (answer - output[0]) ;
		mae +=  abs( answer - output[0] );
		// getchar();
		// break;
	}
	return vector<double> {mse/training_inputs[0].size(), mae/training_inputs[0].size()};
}


void test_output(vector< vector<double> > training_inputs, vector< vector<double> > training_outputs){
	ann::neuralnet n;
	n.import_fromfile("fit = 200");
	evaluate(n, training_inputs, training_outputs);
}

int main(int argc, char** argv){
	double Best_mse = 9999999.0;
	double Best_mae = 9999999.0;
	vector<double> results;
	bool write_output  = false ;
    for ( int i=0; i<argc; i++) {
        if ( string( argv[i] )=="--write_output" ) {
            write_output = true;
            cout << "Priting Outputs\n" ;
            break;
        }
    }
	TimeSeriesSets *time_series_sets = NULL;
	arguments = vector<string>(argv, argv + argc);
	time_series_sets = TimeSeriesSets::generate_from_arguments(arguments, false);

	int32_t time_offset = 1;
	get_argument(arguments, "--time_offset", true, time_offset);

	string file_name = "";
    get_argument(arguments, "--output_parameter_names", true, file_name);

	string experiment = "";
    get_argument(arguments, "--experiment", false, experiment);

	string work_dir = "./";
    get_argument(arguments, "--work_dir", false, work_dir);

    time_series_sets->export_training_series(time_offset, training_inputs, training_outputs);
	neat::pool p(12, 1, 0, true);
	// p.import_fromfile("xor_test.res");
	srand(time(NULL));
    int count = 0 ;
	while (count < 100000){
		double current_fitness = 0.0;
		for (auto s = p.species.begin(); s != p.species.end(); s++)
			for (size_t i=0; i<(*s).genomes.size(); i++){
				ann::neuralnet n;
				neat::genome& g = (*s).genomes[i];
				n.from_genome(g);
				results = evaluate(n, training_inputs[0], training_outputs[0]);
				if (write_output)cout << count << ": " << "Local MSE: " << results[0] << " - Local MAE: " << results[1] << endl;
				current_fitness = results[0];
				if (current_fitness < Best_mse){
					Best_mse = current_fitness;
					string fname = work_dir+"/"+"fit_" + to_string(current_fitness) + ".txt";
					n.export_tofile(fname);
				}
				if (results[1] < Best_mae)
					Best_mae = results[1];

				g.fitness = 1-current_fitness;
				count++ ;
			}

		// cout << "Generation " << p.generation() << " successfuly tested. Global min fitness: " << Best_mse << endl;
		p.new_generation();

		cout << count << ": "  << " Best MSE fitness: " << Best_mse << "  -  Best MAE fitness: " << Best_mae << endl;
		// getchar();
	}
	// test_output();
	p.export_tofile(file_name+"_"+experiment+".res");

	string filename = work_dir+"/"+file_name+"_"+experiment+".best" ;
	std::ofstream output;
	output.open(filename);
	if (!output.is_open()){
		std::cerr << "cannot open file '" << filename << "' !";
		return 0;
	}

	// current state
	output << count << ": "  << " Best MSE fitness: " << Best_mse << "  -  Best MAE fitness: " << Best_mae << endl;
	output.close();



	return 0;
}
