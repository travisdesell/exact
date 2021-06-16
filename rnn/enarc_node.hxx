#ifndef EXAMM_ENARC_NODE_HXX
#define EXAMM_ENARC_NODE_HXX

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"

#include "rnn_node_interface.hxx"

class ENARC_Node : public RNN_Node_Interface{
	private:
		
		double rw;
		double zw;

		double w1;
		double w2;
		double w3;
		double w6;
		double w4;
		double w5;
		double w7;
		double w8;

		vector<double> d_zw;
		vector<double> d_rw;

		vector<double> d_w1;

		vector<double> d_w2; 
		vector<double> d_w3; 
		vector<double> d_w6; 

		vector<double> d_w4; 
		vector<double> d_w5; 
		vector<double> d_w7;
		vector<double> d_w8; 
	
		vector<double> d_h_prev;

		vector<double> z;
		vector<double> l_d_z;

		vector<double> w1_z;
		vector<double> l_w1_z;

		vector<double> w2_w1;
		vector<double> l_w2_w1;

		vector<double> w3_w1;
		vector<double> l_w3_w1;

		vector<double> w6_w1;
		vector<double> l_w6_w1;

		vector<double> w4_w2;
		vector<double> l_w4_w2;

		vector<double> w5_w3;
		vector<double> l_w5_w3;

		vector<double> w7_w3;
		vector<double> l_w7_w3;

		vector<double> w8_w3;
		vector<double> l_w8_w3;


	public:
		ENARC_Node(int _innovation_number, int _type, double _depth);
		~ENARC_Node();

        void initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma);
        void initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng1_1, double range);
        void initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range);
        void initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng);


        double get_gradient(string gradient_name);
        void print_gradient(string gradient_name);

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
