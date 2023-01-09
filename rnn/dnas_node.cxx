#include <cmath>

#include <algorithm>
using std::max;

#include <string>

#include "dnas_node.hxx"
#include "common/log.hxx"

DNASNode::DNASNode(vector<RNN_Node_Interface *> &&nodes, int32_t _innovation_number, int32_t _type, double _depth) : RNN_Node_Interface(_innovation_number, _type, _depth), nodes(nodes) {
  generator.seed(std::random_device()());

  for (auto node : nodes) {
    node->total_inputs = 1;
    node->total_outputs = 1;
  }
}

DNASNode::DNASNode(const DNASNode &src) : RNN_Node_Interface(src.innovation_number, src.node_type, src.depth) {
  generator.seed(std::random_device()());

  for (auto node : src.nodes)
    nodes.push_back(node->copy());

  pi = src.pi;
  d_pi = src.d_pi;
  z = src.z;
  x = src.x;
  xtotal = src.xtotal;
  tao = src.tao;

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
  for (auto node : nodes)
    delete node;
}

template <uniform_random_bit_generator Rng>
void DNASNode::gumbel_noise(Rng &rng, vector<double> &output) {
  Log::info("Drawing %d samples of noise from the gumbel distribution\n", output.size());

  for (int i = 0; i < output.size(); i++)
    output[i] = -log(-log(generate_canonical<double, 64>(rng)));
}

template <uniform_random_bit_generator Rng>
void DNASNode::sample_gumbel_softmax(Rng &rng) {
  z.assign(pi.size(), 0.0);
  x.assign(pi.size(), 0.0);

  gumbel_noise(rng, x);

  xtotal = 0.0;
  double emax = -10000000;
  for (int i = 0; i < z.size(); i++) {
    x[i] += log(pi[i]);
    x[i] /= tao;
    emax = max(emax, x[i]);
  }

  for (int i = 0; i < z.size(); i++) {
    x[i] = exp(x[i] - emax);
    xtotal += x[i];
  }

  for (int i = 0; i < z.size(); i++)
    z[i] = x[i] / xtotal;
}

void DNASNode::reset(int32_t series_length) {
  d_pi = vector<double>(pi.size(), 0.0);
  node_outputs = vector<vector<double>>(series_length, vector<double>(pi.size(), 0.0));
  xtotal = 0;
  x.assign(pi.size(), 0.0);
  for (int i = 0; i < output_values.size(); i++) output_values[i] = 0.0;

  for (auto node : nodes) node->reset(series_length);

  sample_gumbel_softmax(generator);
}

void DNASNode::input_fired(int32_t time, double incoming_output) {
  inputs_fired[time]++;

  input_values[time] += incoming_output;

  if (inputs_fired[time] < total_inputs) {
    return;
  } else if (inputs_fired[time] > total_inputs) {
      Log::fatal("ERROR: inputs_fired on Delta_Node %d at time %d is %d and total_inputs is %d\n", innovation_number, time, inputs_fired[time], total_inputs);
      exit(1);
  }

  for (int i = 0; i < nodes.size(); i++) {
      auto node = nodes[i];
      node->input_fired(time, input_values[time]);
      node_outputs[time][i] = node->output_values[time];
      output_values[time] += z[i] * node->output_values[time];
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
      Log::fatal("ERROR: outputs_fired on Delta_Node %d at time %d is %d and total_outputs is %d\n", innovation_number, time, outputs_fired[time], total_outputs);
      exit(1);
  }

  double delta = error_values[time];
  for (int i = 0; i < z.size(); i++) {
    nodes[i]->output_fired(time, delta * z[i]);
  }

  d_input[time] = 0.0;
  for (auto node : nodes) {
    d_input[time] += node->d_input[time];
  }
}
void DNASNode::initialize_lamarckian(minstd_rand0 &generator, NormalDistribution &normal_distribution, double mu, double sigma) {
  for (auto node : nodes)
    node->initialize_lamarckian(generator, normal_distribution, mu, sigma);
}

void DNASNode::initialize_xavier(minstd_rand0 &generator, uniform_real_distribution<double> &rng_1_1, double range) {
  for (auto node : nodes)
    node->initialize_xavier(generator, rng_1_1, range);
}

void DNASNode::initialize_kaiming(minstd_rand0 &generator, NormalDistribution &normal_distribution, double range) {
  for (auto node : nodes)
    node->initialize_kaiming(generator, normal_distribution, range);
}

void DNASNode::initialize_uniform_random(minstd_rand0 &generator, uniform_real_distribution<double> &rng) {
  for (auto node : nodes)
    node->initialize_uniform_random(generator, rng);
}

int32_t DNASNode::get_number_weights() const {
  int n_weights = pi.size();

  for (auto node : nodes)
    n_weights += node->get_number_weights();

  return n_weights;
}

void DNASNode::get_weights(vector<double> &parameters) const {
  parameters.resize(get_number_weights());
  int32_t offset = 0;
  get_weights(offset, parameters);
}

void DNASNode::set_weights(const vector<double> &parameters) {
  int32_t offset = 0;
  set_weights(offset, parameters);
}

void DNASNode::get_weights(int32_t &offset, vector<double> &parameters) const {
  for (int i = 0; i < pi.size(); i++) parameters[offset++] = pi[i];
  for (auto node : nodes) node->get_weights(offset, parameters);
}

void DNASNode::set_weights(int32_t &offset, const vector<double> &parameters) {
  for (int i = 0; i < pi.size(); i++) pi[i] = parameters[offset++];
  for (auto node : nodes) node->set_weights(offset, parameters);
}

void DNASNode::set_pi(const vector<double> &new_pi) {
  for (int i = 0; i < pi.size(); i++)
    pi[i] = new_pi[i];
}

void DNASNode::get_gradients(vector<double> &gradients) {
  gradients.assign(get_number_weights(), 0.0);
  int offset = 0;
  for (int i = 0; i < pi.size(); i++)
    gradients[offset++] = d_pi[i];

  vector<double> temp;
  for (auto node : nodes) {
    node->get_gradients(temp);
    for (int i = 0; i < temp.size(); i++)
      gradients[offset++] = temp[i];
  }
}

void DNASNode::write_to_stream(ostream &out) {
  RNN_Node_Interface::write_to_stream(out);

  int32_t n = nodes.size();
  out.write((char *) &n, sizeof(int32_t));
  out.write((char *) &pi[0], sizeof(double) * n);
  for (auto node : nodes)
    node->write_to_stream(out);
}

RNN_Node_Interface *DNASNode::copy() const {
  return new DNASNode(*this);
}
