#ifndef EXAMM_RANDOM_DAG_NODE_HXX
#define EXAMM_RANDOM_DAG_NODE_HXX

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include <utility>
using std::pair;
using std::make_pair;


#include "common/random.hxx"

#include "rnn_node_interface.hxx"

class RANDOM_DAG_Node : public RNN_Node_Interface{
	private:
		
		vector<int> node_output;
		// starting node 0
		double rw;
		double zw;
		
		// weights for other nodes
		vector<double> weights;


		// gradients of starting node 0
		vector<double> d_zw;
		vector<double> d_rw;

		// gradients of other nodes 
		vector<vector<double>> d_weights;
		
		// gradient of prev output
		vector<double> d_h_prev;

		// output of edge between node with weight wj from node with weight wi 	
		vector<vector<double>> Nodes;
		// derivative of edge between node with weight wj from node with weight wi 	
		vector<vector<double>> l_Nodes;

		
	public:

		RANDOM_DAG_Node(int _innovation_number, int _type, double _depth);
		~RANDOM_DAG_Node();


        void initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma);
        void initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng1_1, double range);
        void initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range);
        void initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng);


        double get_gradient(string gradient_name);
        void print_gradient(string gradient_name);

        double activation(double value, int act_operator);
        double activation_derivative(double value, double input, int act_operator);

        void input_fired(int time, double incoming_output);

        void try_update_deltas(int time);
        void error_fired(int time, double error);
        void output_fired(int time, double delta);

        uint32_t get_number_weights() const;

        void get_weights(vector<double> &parameters) const;
        void set_weights(const vector<double> &parameters);

        void get_weights(uint32_t &offset, vector<double> &parameters) const;
        void set_weights(uint32_t &offset, const vector<double> &parameters);

        void get_gradients(vector<double> &gradients);

        void reset(int _series_length);

        void write_to_stream(ostream &out);

        RNN_Node_Interface* copy() const;

        friend class RNN_Edge;
	
};



#endif
