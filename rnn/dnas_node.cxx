#include <algorithm>
#include <cassert>
#include <cmath>
using std::max;

#include <string>

#include "common/log.hxx"
#include "dnas_node.hxx"

DNASNode::DNASNode(
    vector<RNN_Node_Interface*>&& _nodes, int32_t _innovation_number, int32_t _type, double _depth, int32_t counter
)
    : RNN_Node_Interface(_innovation_number, _type, _depth),
      nodes(_nodes),
      pi(vector<double>(nodes.size(), 1.0)),
      z(vector<double>(nodes.size())),
      x(vector<double>(nodes.size())),
      g(vector<double>(nodes.size())),
      d_pi(vector<double>(nodes.size())),
      noise(vector<double>(nodes.size())),
      counter(counter) {
    node_type = DNAS_NODE;
    generator.seed(std::random_device()());
    for (auto node : nodes) {
        node->total_inputs = 1;
        node->total_outputs = 1;
    }
    sample_gumbel_softmax(generator);
}

DNASNode::DNASNode(const DNASNode& src) : RNN_Node_Interface(src.innovation_number, src.layer_type, src.depth) {
    generator.seed(std::random_device()());

    for (auto node : src.nodes) {
        nodes.push_back(node->copy());
    }

    node_type = DNAS_NODE;

    pi = src.pi;
    d_pi = src.d_pi;
    z = src.z;
    g = src.g;
    x = src.x;
    xtotal = src.xtotal;
    tao = src.tao;
    stochastic = src.stochastic;
    counter = src.counter;
    maxi = src.maxi;

    series_length = src.series_length;
    input_values = src.input_values;
    output_values = src.output_values;
    error_values = src.error_values;
    d_input = src.d_input;

    inputs_fired = src.inputs_fired;
    total_inputs = src.total_inputs;
    outputs_fired = src.outputs_fired;
    total_outputs = src.total_outputs;
    enabled = src.enabled;
    forward_reachable = src.forward_reachable;
    backward_reachable = src.backward_reachable;
}

DNASNode::~DNASNode() {
    for (auto node : nodes) {
        delete node;
    }
}

template <uniform_random_bit_generator Rng>
void DNASNode::gumbel_noise(Rng& rng, vector<double>& output) {
    for (int i = 0; i < output.size(); i++) {
        output[i] = -log(-log(uniform_real_distribution<double>(0.0, 1.0)(rng)));
    }
}

template <uniform_random_bit_generator Rng>
void DNASNode::sample_gumbel_softmax(Rng& rng) {
    z.assign(pi.size(), 0.0);
    x.assign(pi.size(), 0.0);

    gumbel_noise(rng, g);

    calculate_z();
}

void DNASNode::calculate_z() {
    tao = max(1.0 / 3.0, 1.0 / (1.0 + (double) counter * 0.05));

    xtotal = 0.0;
    double emax = -10000000;
    for (int i = 0; i < z.size(); i++) {
        x[i] = g[i] + log(pi[i]);
        x[i] /= tao;
        emax = max(emax, x[i]);
    }
    for (int i = 0; i < z.size(); i++) {
        x[i] = exp(emax - x[i]);
        xtotal += x[i];
    }
    for (int i = 0; i < z.size(); i++) {
        z[i] = x[i] / xtotal;
    }
}

void DNASNode::reset(int32_t series_length) {
    d_pi = vector<double>(pi.size(), 0.0);
    d_input = vector<double>(series_length, 0.0);
    node_outputs = vector<vector<double>>(series_length, vector<double>(pi.size(), 0.0));
    output_values = vector<double>(series_length, 0.0);
    error_values = vector<double>(series_length, 0.0);
    inputs_fired = vector<int>(series_length, 0);
    outputs_fired = vector<int>(series_length, 0);
    input_values = vector<double>(series_length, 0.0);

    if (counter >= CRYSTALLIZATION_THRESHOLD) {
        nodes[maxi]->reset(series_length);
    } else {
        if (stochastic) {
            sample_gumbel_softmax(generator);
        }

        for (auto node : nodes) {
            node->reset(series_length);
        }
    }
}

void DNASNode::input_fired(int32_t time, double incoming_output) {
    inputs_fired[time]++;

    input_values[time] += incoming_output;

    if (inputs_fired[time] < total_inputs) {
        return;
    } else if (inputs_fired[time] > total_inputs) {
        Log::fatal(
            "ERROR: inputs_fired on DNASNode %d at time %d is %d and total_inputs is %d\n", innovation_number, time,
            inputs_fired[time], total_inputs
        );
        exit(1);
    }

    if (counter >= CRYSTALLIZATION_THRESHOLD) {
        assert(maxi >= 0);

        nodes[maxi]->input_fired(time, input_values[time]);
        node_outputs[time][maxi] = nodes[maxi]->output_values[time];
        output_values[time] = nodes[maxi]->output_values[time];
    } else {
        for (int i = 0; i < nodes.size(); i++) {
            auto node = nodes[i];
            node->input_fired(time, input_values[time]);
            node_outputs[time][i] = node->output_values[time];
            output_values[time] += z[i] * node->output_values[time];
        }
    }
}

void DNASNode::error_fired(int32_t time, double error) {
    Log::fatal("This method actually gets used ???\n");
    exit(-1);
}

void DNASNode::output_fired(int32_t time, double delta) {
    outputs_fired[time]++;

    error_values[time] += delta;

    try_update_deltas(time);
}

void DNASNode::try_update_deltas(int32_t time) {
    if (outputs_fired[time] < total_outputs) {
        return;
    } else if (outputs_fired[time] > total_outputs) {
        Log::fatal(
            "ERROR: outputs_fired on DNMASNode %d at time %d is %d and total_outputs is %d\n", innovation_number, time,
            outputs_fired[time], total_outputs
        );
        exit(1);
    }

    double delta = error_values[time];
    if (counter >= CRYSTALLIZATION_THRESHOLD) {
        nodes[maxi]->output_fired(time, delta);
        d_input[time] += nodes[maxi]->d_input[time];

    } else {
        for (int i = 0; i < z.size(); i++) {
            nodes[i]->output_fired(time, delta * z[i]);
            double p = (x[i] / pi[i]);
            p *= ((delta * node_outputs[time][i]) / xtotal);
            p *= (1 - (x[i] / xtotal));
            p *= 1 / tao;
            d_pi[i] += p;
        }
        d_input[time] = 0.0;
        for (auto node : nodes) {
            d_input[time] += node->d_input[time];
        }
    }
}

void DNASNode::initialize_lamarckian(
    minstd_rand0& generator, NormalDistribution& normal_distribution, double mu, double sigma
) {
    for (auto node : nodes) {
        node->initialize_lamarckian(generator, normal_distribution, mu, sigma);
    }
}

void DNASNode::initialize_xavier(minstd_rand0& generator, uniform_real_distribution<double>& rng_1_1, double range) {
    for (auto node : nodes) {
        node->initialize_xavier(generator, rng_1_1, range);
    }
}

void DNASNode::initialize_kaiming(minstd_rand0& generator, NormalDistribution& normal_distribution, double range) {
    for (auto node : nodes) {
        node->initialize_kaiming(generator, normal_distribution, range);
    }
}

void DNASNode::initialize_uniform_random(minstd_rand0& generator, uniform_real_distribution<double>& rng) {
    for (auto node : nodes) {
        node->initialize_uniform_random(generator, rng);
    }
}

int32_t DNASNode::get_number_weights() const {
    int n_weights = pi.size();

    for (auto node : nodes) {
        n_weights += node->get_number_weights();
    }

    return n_weights;
}

void DNASNode::get_weights(vector<double>& parameters) const {
    parameters.resize(get_number_weights());
    int32_t offset = 0;
    get_weights(offset, parameters);
}

void DNASNode::set_weights(const vector<double>& parameters) {
    int32_t offset = 0;
    set_weights(offset, parameters);
}

void DNASNode::get_weights(int32_t& offset, vector<double>& parameters) const {
    // Log::info("pi start %d; ", offset);
    for (int i = 0; i < pi.size(); i++) {
        parameters[offset++] = pi[i];
    }
    // Log::info_no_header("pi end %d \n", offset);
    for (auto node : nodes) {
        node->get_weights(offset, parameters);
    }
}

void DNASNode::set_weights(int32_t& offset, const vector<double>& parameters) {
    // int start = offset;
    for (int i = 0; i < pi.size(); i++) {
        pi[i] = parameters[offset++];
    }
    // Log::info("Pi indices: %d-%d\n", start, offset);
    for (auto node : nodes) {
        node->set_weights(offset, parameters);
    }
    Log::info("Just set weights\n");
    calculate_z();
    string s = "Pi = { ";
    for (auto p : pi) {
        s += std::to_string(p) + ", ";
    }
    Log::info("%s }\n", s.c_str());
}

void DNASNode::set_pi(const vector<double>& new_pi) {
    for (int i = 0; i < pi.size(); i++) {
        pi[i] = new_pi[i];
    }
    calculate_maxi();
}

void DNASNode::calculate_maxi() {
    if (counter >= CRYSTALLIZATION_THRESHOLD && maxi < 0) {
        maxi = 0;
        double max_pi = pi[0];

        for (int i = 1; i < nodes.size(); i++) {
            if (pi[i] > max_pi) {
                max_pi = pi[i];
                maxi = i;
            }
        }
    }
}

void DNASNode::get_gradients(vector<double>& gradients) {
    gradients.assign(get_number_weights(), 0.0);

    // We want this to count weight updates, fetching gradients implies an impending weight update
    counter += 1;
    calculate_maxi();

    vector<double> temp;
    int offset = 0;

    if (counter >= CRYSTALLIZATION_THRESHOLD) {
        offset += pi.size();
        for (int i = 0; i < nodes.size(); i++) {
            RNN_Node_Interface* node = nodes[i];
            if (i == maxi) {
                node->get_gradients(temp);
                for (int j = 0; j < temp.size(); j++) {
                    gradients[offset++] = temp[j];
                }
            } else {
                offset += node->get_number_weights();
            }
        }
    } else {
        gradients.assign(get_number_weights(), 0.0);
        int offset = 0;
        for (int i = 0; i < pi.size(); i++) {
            gradients[offset++] = d_pi[i];
        }

        for (auto node : nodes) {
            node->get_gradients(temp);
            for (int i = 0; i < temp.size(); i++) {
                gradients[offset++] = temp[i];
            }
        }
    }
}

void DNASNode::write_to_stream(ostream& out) {
    RNN_Node_Interface::write_to_stream(out);

    int32_t n = nodes.size();
    out.write((char*) &n, sizeof(int32_t));
    out.write((char*) &counter, sizeof(int32_t));
    out.write((char*) &pi[0], sizeof(double) * n);
    for (auto node : nodes) {
        node->write_to_stream(out);
    }
}

RNN_Node_Interface* DNASNode::copy() const {
    return new DNASNode(*this);
}

void DNASNode::set_stochastic(bool stochastic) {
    this->stochastic = stochastic;
}
