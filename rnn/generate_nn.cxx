#include <string>
using std::string;

#include <vector>
using std::vector;

#include <cstring>

#include "generate_nn.hxx"
#include "rnn/delta_node.hxx"
#include "rnn/enarc_node.hxx"
#include "rnn/enas_dag_node.hxx"
#include "rnn/gru_node.hxx"
#include "rnn/lstm_node.hxx"
#include "rnn/mgu_node.hxx"
#include "rnn/random_dag_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "rnn/ugrnn_node.hxx"

#include "common/log.hxx"

RNN_Genome *create_ff(const vector<string> &input_parameter_names,
                      uint32_t number_hidden_layers,
                      uint32_t number_hidden_nodes,
                      const vector<string> &output_parameter_names,
                      uint32_t max_recurrent_depth, TrainingParameters tp,
                      WeightType weight_initialize,
                      WeightType weight_inheritance,
                      WeightType mutated_component_weight) {
  Log::debug("creating feed forward network with inputs: %d, hidden: %dx%d, "
             "outputs: %d, max recurrent depth: %d\n",
             input_parameter_names.size(), number_hidden_layers,
             number_hidden_nodes, output_parameter_names.size(),
             max_recurrent_depth);
  vector<RNN_Node_Interface *> rnn_nodes;
  vector<vector<RNN_Node_Interface *>> layer_nodes(2 + number_hidden_layers);
  vector<RNN_Edge *> rnn_edges;
  vector<RNN_Recurrent_Edge *> recurrent_edges;

  int node_innovation_count = 0;
  int edge_innovation_count = 0;
  int current_layer = 0;

  for (uint32_t i = 0; i < input_parameter_names.size(); i++) {
    RNN_Node *node =
        new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer,
                     SIMPLE_NODE, input_parameter_names[i]);
    rnn_nodes.push_back(node);
    layer_nodes[current_layer].push_back(node);
  }
  current_layer++;

  for (uint32_t i = 0; i < number_hidden_layers; i++) {
    for (uint32_t j = 0; j < number_hidden_nodes; j++) {
      RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER,
                                    current_layer, SIMPLE_NODE);
      rnn_nodes.push_back(node);
      layer_nodes[current_layer].push_back(node);

      for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
        rnn_edges.push_back(new RNN_Edge(
            ++edge_innovation_count, layer_nodes[current_layer - 1][k], node));

        for (uint32_t d = 1; d <= max_recurrent_depth; d++) {
          recurrent_edges.push_back(
              new RNN_Recurrent_Edge(++edge_innovation_count, d,
                                     layer_nodes[current_layer - 1][k], node));
        }
      }
    }
    current_layer++;
  }

  for (uint32_t i = 0; i < output_parameter_names.size(); i++) {
    RNN_Node *output_node =
        new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer,
                     SIMPLE_NODE, output_parameter_names[i]);
    rnn_nodes.push_back(output_node);

    for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
      rnn_edges.push_back(new RNN_Edge(++edge_innovation_count,
                                       layer_nodes[current_layer - 1][k],
                                       output_node));

      for (uint32_t d = 1; d <= max_recurrent_depth; d++) {
        recurrent_edges.push_back(new RNN_Recurrent_Edge(
            ++edge_innovation_count, d, layer_nodes[current_layer - 1][k],
            output_node));
      }
    }
  }

  RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, tp,
                                      weight_initialize, weight_inheritance,
                                      mutated_component_weight);
  genome->set_parameter_names(input_parameter_names, output_parameter_names);
  return genome;
}

RNN_Genome *create_jordan(const vector<string> &input_parameter_names,
                          uint32_t number_hidden_layers,
                          uint32_t number_hidden_nodes,
                          const vector<string> &output_parameter_names,
                          uint32_t max_recurrent_depth, TrainingParameters tp,
                          WeightType weight_initialize) {
  Log::debug("creating jordan neural network with inputs: %d, hidden: %dx%d, "
             "outputs: %d, max recurrent depth: %d\n",
             input_parameter_names.size(), number_hidden_layers,
             number_hidden_nodes, output_parameter_names.size(),
             max_recurrent_depth);
  vector<RNN_Node_Interface *> rnn_nodes;
  vector<RNN_Node_Interface *> output_layer;
  vector<vector<RNN_Node_Interface *>> layer_nodes(2 + number_hidden_layers);
  vector<RNN_Edge *> rnn_edges;
  vector<RNN_Recurrent_Edge *> recurrent_edges;

  int node_innovation_count = 0;
  int edge_innovation_count = 0;
  int current_layer = 0;

  for (uint32_t i = 0; i < input_parameter_names.size(); i++) {
    RNN_Node *node =
        new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer,
                     SIMPLE_NODE, input_parameter_names[i]);
    rnn_nodes.push_back(node);
    layer_nodes[current_layer].push_back(node);
  }
  current_layer++;

  for (uint32_t i = 0; i < number_hidden_layers; i++) {
    for (uint32_t j = 0; j < number_hidden_nodes; j++) {
      RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER,
                                    current_layer, JORDAN_NODE);
      rnn_nodes.push_back(node);
      layer_nodes[current_layer].push_back(node);

      for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
        rnn_edges.push_back(new RNN_Edge(
            ++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
      }
    }
    current_layer++;
  }

  for (uint32_t i = 0; i < output_parameter_names.size(); i++) {
    RNN_Node *output_node =
        new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer,
                     SIMPLE_NODE, output_parameter_names[i]);
    output_layer.push_back(output_node);

    rnn_nodes.push_back(output_node);

    for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
      rnn_edges.push_back(new RNN_Edge(++edge_innovation_count,
                                       layer_nodes[current_layer - 1][k],
                                       output_node));
    }
  }

  // connect the output node with recurrent edges to each hidden node
  for (uint32_t k = 0; k < output_layer.size(); k++) {
    for (uint32_t i = 0; i < number_hidden_layers; i++) {
      for (uint32_t j = 0; j < number_hidden_nodes; j++) {
        for (uint32_t d = 1; d <= max_recurrent_depth; d++) {
          recurrent_edges.push_back(
              new RNN_Recurrent_Edge(++edge_innovation_count, d,
                                     output_layer[k], layer_nodes[i + 1][j]));
        }
      }
    }
  }

  RNN_Genome *genome =
      new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, tp,
                     weight_initialize, WeightType::NONE, WeightType::NONE);
  genome->set_parameter_names(input_parameter_names, output_parameter_names);
  return genome;
}

RNN_Genome *create_elman(const vector<string> &input_parameter_names,
                         uint32_t number_hidden_layers,
                         uint32_t number_hidden_nodes,
                         const vector<string> &output_parameter_names,
                         uint32_t max_recurrent_depth, TrainingParameters tp,
                         WeightType weight_initialize) {
  Log::debug("creating elman network with inputs: %d, hidden: %dx%d, outputs: "
             "%d, max recurrent depth: %d\n",
             input_parameter_names.size(), number_hidden_layers,
             number_hidden_nodes, output_parameter_names.size(),
             max_recurrent_depth);
  vector<RNN_Node_Interface *> rnn_nodes;
  vector<RNN_Node_Interface *> output_layer;
  vector<vector<RNN_Node_Interface *>> layer_nodes(2 + number_hidden_layers);
  vector<RNN_Edge *> rnn_edges;
  vector<RNN_Recurrent_Edge *> recurrent_edges;

  int node_innovation_count = 0;
  int edge_innovation_count = 0;
  int current_layer = 0;

  for (uint32_t i = 0; i < input_parameter_names.size(); i++) {
    RNN_Node *node =
        new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer,
                     SIMPLE_NODE, input_parameter_names[i]);
    rnn_nodes.push_back(node);
    layer_nodes[current_layer].push_back(node);
  }
  current_layer++;

  for (uint32_t i = 0; i < number_hidden_layers; i++) {
    for (uint32_t j = 0; j < number_hidden_nodes; j++) {
      RNN_Node *node = new RNN_Node(++node_innovation_count, HIDDEN_LAYER,
                                    current_layer, ELMAN_NODE);
      rnn_nodes.push_back(node);
      layer_nodes[current_layer].push_back(node);

      for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
        rnn_edges.push_back(new RNN_Edge(
            ++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
      }
    }
    current_layer++;
  }

  for (uint32_t i = 0; i < output_parameter_names.size(); i++) {
    RNN_Node *output_node =
        new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer,
                     SIMPLE_NODE, output_parameter_names[i]);
    output_layer.push_back(output_node);

    rnn_nodes.push_back(output_node);

    for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
      rnn_edges.push_back(new RNN_Edge(++edge_innovation_count,
                                       layer_nodes[current_layer - 1][k],
                                       output_node));
    }
  }

  // recurrently connect every hidden node to every other hidden node
  for (uint32_t i = 0; i < number_hidden_layers; i++) {
    for (uint32_t j = 0; j < number_hidden_nodes; j++) {

      for (uint32_t k = 0; k < number_hidden_layers; k++) {
        for (uint32_t l = 0; l < number_hidden_nodes; l++) {

          for (uint32_t d = 1; d <= max_recurrent_depth; d++) {
            recurrent_edges.push_back(new RNN_Recurrent_Edge(
                ++edge_innovation_count, d, layer_nodes[i + 1][j],
                layer_nodes[k + 1][l]));
          }
        }
      }
    }
  }

  RNN_Genome *genome =
      new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, tp,
                     weight_initialize, WeightType::NONE, WeightType::NONE);
  genome->set_parameter_names(input_parameter_names, output_parameter_names);
  return genome;
}

template <class MemoryCell>
RNN_Genome *create_memory_cell_nn(const vector<string> &input_parameter_names,
                                  uint32_t number_hidden_layers,
                                  uint32_t number_hidden_nodes,
                                  const vector<string> &output_parameter_names,
                                  uint32_t max_recurrent_depth,
                                  TrainingParameters tp,
                                  WeightType weight_initialize) {
  static_assert(std::is_base_of<RNN_Node_Interface, MemoryCell>::value,
                "Supplied node type is not derived from RNN_Node_Interface");
  Log::debug("creating LSTM network with inputs: %d, hidden: %dx%d, outputs: "
             "%d, max recurrent depth: %d\n",
             input_parameter_names.size(), number_hidden_layers,
             number_hidden_nodes, output_parameter_names.size(),
             max_recurrent_depth);
  vector<RNN_Node_Interface *> rnn_nodes;
  vector<vector<RNN_Node_Interface *>> layer_nodes(2 + number_hidden_layers);
  vector<RNN_Edge *> rnn_edges;
  vector<RNN_Recurrent_Edge *> recurrent_edges;

  int node_innovation_count = 0;
  int edge_innovation_count = 0;
  int current_layer = 0;

  for (uint32_t i = 0; i < input_parameter_names.size(); i++) {
    RNN_Node *node =
        new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer,
                     SIMPLE_NODE, input_parameter_names[i]);
    rnn_nodes.push_back(node);
    layer_nodes[current_layer].push_back(node);
  }
  current_layer++;

  for (uint32_t i = 0; i < number_hidden_layers; i++) {
    for (uint32_t j = 0; j < number_hidden_nodes; j++) {
      RNN_Node_Interface *node =
          new MemoryCell(++node_innovation_count, HIDDEN_LAYER, current_layer);
      rnn_nodes.push_back(node);
      layer_nodes[current_layer].push_back(node);

      for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
        rnn_edges.push_back(new RNN_Edge(
            ++edge_innovation_count, layer_nodes[current_layer - 1][k], node));
      }
    }
    current_layer++;
  }

  for (uint32_t i = 0; i < output_parameter_names.size(); i++) {
    RNN_Node *output_node =
        new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer,
                     SIMPLE_NODE, output_parameter_names[i]);
    rnn_nodes.push_back(output_node);

    for (uint32_t k = 0; k < layer_nodes[current_layer - 1].size(); k++) {
      rnn_edges.push_back(new RNN_Edge(++edge_innovation_count,
                                       layer_nodes[current_layer - 1][k],
                                       output_node));
    }
  }

  RNN_Genome *genome =
      new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, tp,
                     weight_initialize, WeightType::NONE, WeightType::NONE);
  genome->set_parameter_names(input_parameter_names, output_parameter_names);
  return genome;
}

inst_create_memory_cell(LSTM_Node);
inst_create_memory_cell(GRU_Node);
inst_create_memory_cell(Delta_Node);
inst_create_memory_cell(MGU_Node);
inst_create_memory_cell(UGRNN_Node);
