#include <cmath>

#include <fstream>
using std::ostream;

#include <iomanip>
using std::setw;

#include <string>
using std::string;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;

#include "common/random.hxx"
#include "common/log.hxx"

#include "rnn_node_interface.hxx"
#include "mse.hxx"
#include "enarc_node.hxx"

#define NUMBER_ENARC_WEIGHTS 10

ENARC_Node::ENARC_Node(int _innovation_number, int _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth) {
  node_type = ENARC_NODE;
}

ENARC_Node::~ENARC_Node() {

}

void ENARC_Node::initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {

    zw = bound(normal_distribution.random(generator, mu, sigma));
    rw = bound(normal_distribution.random(generator, mu, sigma));

    w1 = bound(normal_distribution.random(generator, mu, sigma));

    w2 = bound(normal_distribution.random(generator, mu, sigma));
    w3 = bound(normal_distribution.random(generator, mu, sigma));
    w6 = bound(normal_distribution.random(generator, mu, sigma));

    w4 = bound(normal_distribution.random(generator, mu, sigma));
    w5 = bound(normal_distribution.random(generator, mu, sigma));
    w7 = bound(normal_distribution.random(generator, mu, sigma));
    w8 = bound(normal_distribution.random(generator, mu, sigma));
}

void ENARC_Node::initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) {

    zw = range * (rng_1_1(generator));
    rw = range * (rng_1_1(generator));

    w1 = range * (rng_1_1(generator));

    w2 = range * (rng_1_1(generator));
    w3 = range * (rng_1_1(generator));
    w6 = range * (rng_1_1(generator));

    w4 = range * (rng_1_1(generator));
    w5 = range * (rng_1_1(generator));
    w7 = range * (rng_1_1(generator));
    w8 = range * (rng_1_1(generator));
}

void ENARC_Node::initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range) {
    zw = range * normal_distribution.random(generator, 0, 1);
    rw = range * normal_distribution.random(generator, 0, 1);

    w1 = range * normal_distribution.random(generator, 0, 1);

    w2 = range * normal_distribution.random(generator, 0, 1);
    w3 = range * normal_distribution.random(generator, 0, 1);
    w6 = range * normal_distribution.random(generator, 0, 1);

    w4 = range * normal_distribution.random(generator, 0, 1);
    w5 = range * normal_distribution.random(generator, 0, 1);
    w7 = range * normal_distribution.random(generator, 0, 1);
    w8 = range * normal_distribution.random(generator, 0, 1);
}

void ENARC_Node::initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) {
    zw = rng(generator);
    rw = rng(generator);

    w1 = rng(generator);

    w2 = rng(generator);
    w3 = rng(generator);
    w6 = rng(generator);

    w4 = rng(generator);
    w5 = rng(generator);
    w7 = rng(generator);
    w8 = rng(generator);
}




double ENARC_Node::get_gradient(string gradient_name) {
    double gradient_sum = 0.0;
    for (uint32_t i = 0; i < series_length; i++ ) {
        if (gradient_name == "zw") {
            gradient_sum += d_zw[i];
        } else if (gradient_name == "rw") {
            gradient_sum += d_rw[i];
        } else if (gradient_name == "w1") {
            gradient_sum += d_w1[i];

        } else if (gradient_name == "w2") {
            gradient_sum += d_w2[i];
        } else if (gradient_name == "w3") {
            gradient_sum += d_w3[i];
        } else if (gradient_name == "w6") {
            gradient_sum += d_w6[i];

        } else if (gradient_name == "w4") {
            gradient_sum += d_w4[i];



        } else if (gradient_name == "w5") {
            gradient_sum += d_w5[i];
        } else if (gradient_name == "w7") {
            gradient_sum += d_w7[i];
        } else if (gradient_name == "w8") {
            gradient_sum += d_w8[i];
        } else {
            Log::fatal("ERROR: tried to get unknown gradient: '%s'\n", gradient_name.c_str()); 
            exit(1);
        }
    }
    return gradient_sum;
}

void ENARC_Node::print_gradient(string gradient_name) {
    Log::info("\tgradient['%s']: %lf\n", gradient_name.c_str(), get_gradient(gradient_name));
}

void ENARC_Node::input_fired(int time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) return;
    else if (inputs_fired[time] > total_inputs) {
        Log::fatal("ERROR: inputs_fired on ENARC_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time, inputs_fired[time], total_inputs);
        exit(1);
    }

    //update the reset gate bias so its centered around 1
    //r_bias += 1;

    double x = input_values[time];

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];

    double xzw = x*zw;
    double hrw = h_prev*rw;
    double z_sum = hrw+xzw;

     z[time] = tanh(z_sum);
     l_d_z[time] = tanh_derivative(z[time]);

  /**************1st layer ***************/    

     double w1_z_mul = w1 * z[time];
     w1_z[time] = tanh(w1_z_mul);
     l_w1_z[time] = tanh_derivative(w1_z[time]);

 //     /**************2nd layer ***************/

    double w6_w1_mul = w6*w1_z[time];
    w6_w1[time] = swish(w6_w1_mul);
    l_w6_w1[time] = swish_derivative(w6_w1_mul,w6_w1[time]);

     double w2_w1_mul = w2*w1_z[time];
     w2_w1[time] = tanh(w2_w1_mul);
     l_w2_w1[time] = tanh_derivative(w2_w1[time]);

     double w3_w1_mul = w3*w1_z[time];
     w3_w1[time] = leakyReLU(w3_w1_mul);
     l_w3_w1[time] = leakyReLU_derivative(w3_w1[time]);


  // *************3rd layer **************    

     double w4_w2_mul = w4*w2_w1[time];
     w4_w2[time] = leakyReLU(w4_w2_mul);
     l_w4_w2[time] = leakyReLU_derivative(w4_w2[time]);

     double w5_w3_mul = w5*w3_w1[time];
     w5_w3[time] = swish(w5_w3_mul);
     l_w5_w3[time] = swish_derivative(w5_w3_mul,w5_w3[time]);

    double w7_w3_mul = w7*w3_w1[time];
     w7_w3[time] = tanh(w7_w3_mul);
     l_w7_w3[time] = tanh_derivative(w7_w3[time]);

     double w8_w3_mul = w8*w3_w1[time];
     w8_w3[time] = swish(w8_w3_mul);
     l_w8_w3[time] = swish_derivative(w8_w3_mul,w8_w3[time]);    

  /**************4rd layer ***************/    

     output_values[time] = w6_w1[time] + w4_w2[time] + w5_w3[time] + w7_w3[time] + w8_w3[time];
    




}

void ENARC_Node::try_update_deltas(int time){
  if (outputs_fired[time] < total_outputs) return;
    else if (outputs_fired[time] > total_outputs) {
        Log::fatal("ERROR: outputs_fired on ENARC_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired[time], total_outputs);
        exit(1);
    }

    double error = error_values[time];
    double x = input_values[time];

    double h_prev = 0.0;
    if (time > 0) h_prev = output_values[time - 1];

    double d_h = error;
    if (time < (series_length - 1)) d_h += d_h_prev[time + 1];

    //d_h *= 0.2;

    d_w6[time] = d_h*l_w6_w1[time]*w1_z[time];

    d_w8[time] = d_h*l_w8_w3[time]*w3_w1[time];
    d_w7[time] = d_h*l_w7_w3[time]*w3_w1[time];
    d_w5[time] = d_h*l_w5_w3[time]*w3_w1[time];

    d_w4[time] = d_h*l_w4_w2[time]*w2_w1[time];

    double d_h_tanh2 = d_h*l_w4_w2[time]*w4;
    double d_h_leaky2 = d_h*l_w8_w3[time]*w8 +d_h*l_w7_w3[time]*w7+ d_h*l_w5_w3[time]*w5;
    
    
    d_w2[time] = d_h_tanh2*l_w2_w1[time]*w1_z[time];
    d_w3[time] = d_h_leaky2*l_w3_w1[time]*w1_z[time];
    
    double d_h_tanh1 =  d_h*l_w6_w1[time]*w6 + d_h_tanh2*l_w2_w1[time]*w2 + d_h_leaky2*l_w3_w1[time]*w3;

    d_w1[time] = d_h_tanh1*l_w1_z[time]*z[time];

    double d_h_tanh = d_h_tanh1*l_w1_z[time]*w1;

    d_h_prev[time] += d_h_tanh*l_d_z[time]*rw;
    d_rw[time] = d_h_tanh*l_d_z[time]*h_prev;

    d_input[time] += d_h_tanh*l_d_z[time]*zw;
    d_zw[time] = d_h_tanh*l_d_z[time]*x;
}

void ENARC_Node::error_fired(int time, double error) {
    outputs_fired[time]++;

    error_values[time] *= error;

    try_update_deltas(time);
}

void ENARC_Node::output_fired(int time, double delta) {
    outputs_fired[time]++;

    error_values[time] += delta;

    try_update_deltas(time);
}


uint32_t ENARC_Node::get_number_weights() const {
    return NUMBER_ENARC_WEIGHTS;
}

void ENARC_Node::get_weights(vector<double> &parameters) const {
    parameters.resize(get_number_weights());
    uint32_t offset = 0;
    get_weights(offset, parameters);
}

void ENARC_Node::set_weights(const vector<double> &parameters) {
    uint32_t offset = 0;
    set_weights(offset, parameters);
}

void ENARC_Node::set_weights(uint32_t &offset, const vector<double> &parameters) {
    //uint32_t start_offset = offset;

  zw = bound(parameters[offset++]);
  rw = bound(parameters[offset++]);

  w1 = bound(parameters[offset++]);

  w2 = bound(parameters[offset++]);
  w3 = bound(parameters[offset++]);
  w6 = bound(parameters[offset++]);

  w4 = bound(parameters[offset++]);
  w5 = bound(parameters[offset++]);
  w7 = bound(parameters[offset++]);
  w8 = bound(parameters[offset++]);


    //uint32_t end_offset = offset;
    //Log::trace("set weights from offset %d to %d on ENARC_Node %d\n", start_offset, end_offset, innovation_number);
}

void ENARC_Node::get_weights(uint32_t &offset, vector<double> &parameters) const {
    //uint32_t start_offset = offset;



    parameters[offset++] = zw;
    parameters[offset++] = rw;

    parameters[offset++] = w1; 

    parameters[offset++] = w2;
    parameters[offset++] = w3;
    parameters[offset++] = w6;

    parameters[offset++] = w4;
    parameters[offset++] = w5;
    parameters[offset++] = w7;
    parameters[offset++] = w8;


    //uint32_t end_offset = offset;
    //Log::trace("got weights from offset %d to %d on ENARC_Node %d\n", start_offset, end_offset, innovation_number);
}

void ENARC_Node::get_gradients(vector<double> &gradients) {
    gradients.assign(NUMBER_ENARC_WEIGHTS, 0.0);

    for (uint32_t i = 0; i < NUMBER_ENARC_WEIGHTS; i++) {
        gradients[i] = 0.0;
    }

    for (uint32_t i = 0; i < series_length; i++) {
        gradients[0] += d_zw[i];
        gradients[1] += d_rw[i];

        gradients[2] += d_w1[i];

        gradients[3] += d_w2[i];
        gradients[4] += d_w3[i];
        gradients[5] += d_w6[i];

        gradients[6] += d_w4[i];
        gradients[7] += d_w5[i];
        gradients[8] += d_w7[i];
        gradients[9] += d_w8[i];

    }
}

void ENARC_Node::reset(int _series_length) {
    series_length = _series_length;

    d_zw.assign(series_length, 0.0);
    d_rw.assign(series_length, 0.0);

    d_w1.assign(series_length, 0.0);

    d_w2.assign(series_length, 0.0); 
    d_w3.assign(series_length, 0.0); 
    d_w6.assign(series_length, 0.0); 

    d_w4.assign(series_length, 0.0); 
    d_w5.assign(series_length, 0.0); 
    d_w7.assign(series_length, 0.0);
    d_w8.assign(series_length, 0.0); 
  
    d_h_prev.assign(series_length, 0.0);

    z.assign(series_length, 0.0);
    l_d_z.assign(series_length, 0.0);

    w1_z.assign(series_length, 0.0);
    l_w1_z.assign(series_length, 0.0);

    w2_w1.assign(series_length, 0.0);
    l_w2_w1.assign(series_length, 0.0);

    w3_w1.assign(series_length, 0.0);
    l_w3_w1.assign(series_length, 0.0);

    w6_w1.assign(series_length, 0.0);
    l_w6_w1.assign(series_length, 0.0);

    w4_w2.assign(series_length, 0.0);
    l_w4_w2.assign(series_length, 0.0);

    w5_w3.assign(series_length, 0.0);
    l_w5_w3.assign(series_length, 0.0);

    w7_w3.assign(series_length, 0.0);
    l_w7_w3.assign(series_length, 0.0);

    w8_w3.assign(series_length, 0.0);
    l_w8_w3.assign(series_length, 0.0);



    //reset values from rnn_node_interface
    d_input.assign(series_length, 0.0);
    error_values.assign(series_length, 0.0);

    input_values.assign(series_length, 0.0);
    output_values.assign(series_length, 0.0);

    inputs_fired.assign(series_length, 0);
    outputs_fired.assign(series_length, 0);
}

RNN_Node_Interface* ENARC_Node::copy() const {
    ENARC_Node* n = new ENARC_Node(innovation_number, layer_type, depth);

    //copy ENARC_Node values
    n->rw = rw;
    n->zw = zw;
    n->w1 = w1;

    n->w2 = w2;
    n->w3 = w3;
    n->w6 = w6;

    n->w4 = w4;
    n->w5 = w5;
    n->w7 = w7;
    n->w8 = w8;

    n->d_zw = d_zw;
    n->d_rw = d_rw;

    n->d_w1 = d_w1;

    n->d_w2 = d_w2;
    n->d_w3 = d_w3;
    n->d_w6 = d_w6;

    n->d_w4 = d_w4;
    n->d_w5 = d_w5;
    
    n->d_w7 = d_w7;
    n->d_w8 = d_w8;
    n->d_h_prev = d_h_prev;

    n->z = z;
    n->l_d_z = l_d_z; 
    
    n->w1_z = w1_z;
    n->l_w1_z = l_w1_z; 
    
    n->w2_w1 = w2_w1;
    n->l_w2_w1 = l_w2_w1; 
    
    n->w3_w1 = w3_w1;
    n->l_w3_w1 = l_w3_w1; 
    
    n->w6_w1 = w6_w1;
    n->l_w6_w1 = l_w6_w1;
    
    n->w4_w2 = w4_w2;
    n->l_w4_w2 = l_w4_w2;
    
    n->w5_w3 = w5_w3;
    n->l_w5_w3 = l_w5_w3;
    
    n->w7_w3 = w7_w3;
    n->l_w7_w3 = l_w7_w3;
    
    n->w8_w3 = w8_w3; 
    n->l_w8_w3 = l_w8_w3; 


    //copy RNN_Node_Interface values
    n->series_length = series_length;
    n->input_values = input_values;
    n->output_values = output_values;
    n->error_values = error_values;
    n->d_input = d_input;

    n->inputs_fired = inputs_fired;
    n->total_inputs = total_inputs;
    n->outputs_fired = outputs_fired;
    n->total_outputs = total_outputs;
    n->enabled = enabled;
    n->forward_reachable = forward_reachable;
    n->backward_reachable = backward_reachable;

    return n;
}

void ENARC_Node::write_to_stream(ostream &out) {
    RNN_Node_Interface::write_to_stream(out);
}

