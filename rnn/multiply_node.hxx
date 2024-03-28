#ifndef EXAMM_MULTIPLY_NODE_HXX
#define EXAMM_MULTIPLY_NODE_HXX

#include <vector>
using std::vector;

#include "rnn_node_interface.hxx"

class MULTIPLY_Node : public RNN_Node_Interface {
   protected:
    double bias;
    double d_bias;

    vector<vector<double>> ordered_input;

   public:
    // constructor for hidden nodes
    MULTIPLY_Node(int32_t _innovation_number, int32_t _layer_type, double _depth);

    ~MULTIPLY_Node();

    void initialize_lamarckian(
        minstd_rand0& generator, NormalDistribution& normal_distribution, double mu, double sigma
    );
    void initialize_xavier(minstd_rand0& generator, uniform_real_distribution<double>& rng1_1, double range);
    void initialize_kaiming(minstd_rand0& generator, NormalDistribution& normal_distribution, double range);
    void initialize_uniform_random(minstd_rand0& generator, uniform_real_distribution<double>& rng);

    void input_fired(int32_t time, double incoming_output);
    virtual void try_update_deltas(int32_t time);
    void output_fired(int32_t time, double delta);
    void error_fired(int32_t time, double error);

    int32_t get_number_weights() const;
    void get_weights(vector<double>& parameters) const;
    void set_weights(const vector<double>& parameters);
    void get_weights(int32_t& offset, vector<double>& parameters) const;
    void set_weights(int32_t& offset, const vector<double>& parameters);

    void reset(int32_t _series_length);

    void get_gradients(vector<double>& gradients);

    RNN_Node_Interface* copy() const;

    void write_to_stream(ostream& out);

    friend class RNN_Edge;
    friend class RNN_Recurrent_Edge;
};

#endif
