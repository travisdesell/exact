#ifndef EXAMM_ENAS_DAG_NODE_HXX
#define EXAMM_ENAS_DAG_NODE_HXX

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

class ENAS_DAG_Node : public RNN_Node_Interface{
	private:
		
		/**
         * Weight of the start node in the memory cell.
         */
		double rw;
		/**
         * Weight of the start node in the memory cell.
         */
		double zw;
		

		/**
         * Weights of the subseqeuent nodes in topological order the memory cell.
         */
		vector<double> weights;


		/**
         * Gradient of the weight of the start node in the memory cell.
         */
		vector<double> d_zw;
		/**
         * Gradient of the weight of the start node in the memory cell.
         */
		vector<double> d_rw;

		/**
         * Gradient of the  weights of the other nodes in the memory cell
         */ 
		vector<vector<double>> d_weights;
		
		/**
         * Gradient of the previous output i.e. t-1 in the memory cell.
         */
		vector<double> d_h_prev;

		/**
         * Outputs of the nodes in the memory cell connecting with weight wj from node with weight wi.
         */
		vector<vector<double>> Nodes;
		
		/**
         * Derivative of the nodes in the memory cell connecting with weight wj from node with weight wi.
         */ 	
		vector<vector<double>> l_Nodes;

		
	public:

		ENAS_DAG_Node(int _innovation_number, int _type, double _depth);
		~ENAS_DAG_Node();


        void initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma);
        void initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng1_1, double range);
        void initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range);
        void initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng);


        /**
        *   Gives the gradients in essence the deriviative  of the weights in the memory cell in  the network.    
        *   
        *   \param gradient_name in the weight name.
        *   
        *   \return gradient. 
        */
        double get_gradient(string gradient_name);
        
        /**
        *   Gives the gradients in essence the deriviative  of the weights in the memory cell in  the network.    
        *   
        *   \param gradient_name in the weight name.
        *   
        * 	prints the gradient. 
        */
        void print_gradient(string gradient_name);


        /**
        *   Gives the activations of the node in the memory cell in the network.    
        *   
        *   \param value is  the output of the node after multiplying weights
        *   \param act_operator is the actiation type : sigmoid, tanh, swish, identity
        *   
        *   \return output of the node after applying activation
        */
        double activation(double value, int act_operator);
        
        /**
        *   Gives the activations derviative of the node in the memory cell in the network.    
        *   
        *   \param input is the input of the node before multiplying weights
        *   \param value is  the output of the node after multiplying weights
        *   \param act_operator is the actiation type : sigmoid, tanh, swish, identity
        *   
        *   \return output derivative of the node after applying activation
        */
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
