#ifndef EXAMM_RNN_EDGE_HXX
#define EXAMM_RNN_EDGE_HXX

#include <algorithm>
#include <vector>

#include "inon.hxx"
#include "rnn_node_interface.hxx"
using std::vector;

#include <unordered_map>
using std::unordered_map;

class RNN_Edge;
typedef inon_type<RNN_Edge> edge_inon;

class RNN_Edge {
 private:
  vector<double> outputs;
  vector<double> deltas;
  vector<bool> dropped_out;

  double weight;
  double d_weight;

  bool enabled;
  bool forward_reachable;
  bool backward_reachable;

  RNN_Node_Interface *input_node;
  RNN_Node_Interface *output_node;

 public:
  const edge_inon inon;
  const node_inon input_inon;
  const node_inon output_inon;

  RNN_Edge(edge_inon inon, RNN_Node_Interface *_input_node, RNN_Node_Interface *_output_node);

  RNN_Edge(edge_inon inon, node_inon _input_inon, node_inon _output_inon, const vector<RNN_Node_Interface *> &nodes);

  RNN_Edge *copy(const vector<RNN_Node_Interface *> new_nodes);
  RNN_Edge *copy(unordered_map<node_inon, RNN_Node_Interface *> new_nodes);

  void reset(uint32_t nseries_length);

  void propagate_forward(int32_t time);
  void propagate_backward(int32_t time);

  void propagate_forward(int32_t time, bool training, double dropout_probability);
  void propagate_backward(int32_t time, bool training, double dropout_probability);

  double get_gradient() const;
  edge_inon get_inon() const;
  node_inon get_input_inon() const;
  node_inon get_output_inon() const;

  const RNN_Node_Interface *get_input_node() const;
  const RNN_Node_Interface *get_output_node() const;

  bool is_enabled() const;
  bool is_reachable() const;

  bool equals(RNN_Edge *other) const;

  void write_to_stream(ostream &out);

  friend class RNN_Genome;
  friend class RNN;
  friend class GenomeOperators;
  friend class EXAMM;
};

struct sort_RNN_Edges_by_depth {
  bool operator()(RNN_Edge *n1, RNN_Edge *n2) {
    if (n1->get_input_node()->get_depth() < n2->get_input_node()->get_depth()) {
      return true;

    } else if (n1->get_input_node()->get_depth() == n2->get_input_node()->get_depth()) {
      // make sure the order of the edges is *always* the same
      // going through the edges in different orders may effect the output
      // of backpropagation
      if (n1->inon < n2->inon) {
        return true;
      } else {
        return false;
      }

    } else {
      return false;
    }
  }
};

struct sort_RNN_Edges_by_output_depth {
  bool operator()(RNN_Edge *n1, RNN_Edge *n2) {
    if (n1->get_output_node()->get_depth() < n2->get_output_node()->get_depth()) {
      return true;

    } else if (n1->get_output_node()->get_depth() == n2->get_output_node()->get_depth()) {
      // make sure the order of the edges is *always* the same
      // going through the edges in different orders may effect the output
      // of backpropagation
      if (n1->inon < n2->inon) {
        return true;
      } else {
        return false;
      }

    } else {
      return false;
    }
  }
};

struct sort_RNN_Edges_by_inon {
  bool operator()(RNN_Edge *n1, RNN_Edge *n2) { return n1->inon < n2->inon; }
};

void insert_edge_by_depth(vector<RNN_Edge *> &edges, RNN_Edge *edge);
void insert_edge_by_innovation(vector<RNN_Edge *> &edges, RNN_Edge *edge);

#endif
