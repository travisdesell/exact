#include "som_vector_space.hxx"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "common/log.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node_interface.hxx"
#include "rnn/rnn_recurrent_edge.hxx"

SomVectorSpace::SomVectorSpace() : node_vector_size(0), edge_vector_size(0) {
}

size_t SomVectorSpace::get_node_vector_size() const {
    return node_vector_size;
}

size_t SomVectorSpace::get_edge_vector_size() const {
    return edge_vector_size;
}

const vector<int32_t>& SomVectorSpace::get_node_innovations() const {
    return node_innovations;
}

const vector<int32_t>& SomVectorSpace::get_edge_innovations() const {
    return edge_innovations;
}

void SomVectorSpace::recompute_sizes() {
    node_vector_size = 0;
    for (int32_t count : node_weight_counts) {
        node_vector_size += (size_t) count;
    }
    edge_vector_size = edge_innovations.size();  // one weight per edge innovation
}

size_t SomVectorSpace::node_offset(int32_t catalog_index) const {
    size_t offset = 0;
    for (int32_t i = 0; i < catalog_index; i++) {
        offset += (size_t) node_weight_counts[i];
    }
    return offset;
}

size_t SomVectorSpace::edge_offset(int32_t catalog_index) const {
    return (size_t) catalog_index;
}

bool SomVectorSpace::expand_from_genome(RNN_Genome* genome, vector<SomUnit>& units) {
    bool grew = false;

    for (RNN_Node_Interface* node : genome->nodes) {
        int32_t innov = node->get_innovation_number();
        if (node_index.find(innov) != node_index.end()) {
            continue;
        }

        int32_t weight_count = node->get_number_weights();
        node_index[innov] = (int32_t) node_innovations.size();
        node_innovations.push_back(innov);
        node_weight_counts.push_back(weight_count);
        node_vector_size += (size_t) weight_count;

        for (SomUnit& unit : units) {
            unit.append_node_zeros(weight_count);
        }
        grew = true;
    }

    auto register_edge = [&](int32_t innov) {
        if (edge_index.find(innov) != edge_index.end()) {
            return;
        }
        edge_index[innov] = (int32_t) edge_innovations.size();
        edge_innovations.push_back(innov);
        edge_vector_size += 1;

        for (SomUnit& unit : units) {
            unit.append_edge_zeros(1);
        }
        grew = true;
    };

    for (RNN_Edge* edge : genome->edges) {
        register_edge(edge->get_innovation_number());
    }
    for (RNN_Recurrent_Edge* edge : genome->recurrent_edges) {
        register_edge(edge->get_innovation_number());
    }

    return grew;
}

void SomVectorSpace::encode_genome(
    RNN_Genome* genome, vector<double>& node_vector, vector<double>& edge_vector
) const {
    node_vector.assign(node_vector_size, 0.0);
    edge_vector.assign(edge_vector_size, 0.0);

    map<int32_t, RNN_Node_Interface*> node_by_innov;
    for (RNN_Node_Interface* node : genome->nodes) {
        node_by_innov[node->get_innovation_number()] = node;
    }

    size_t offset = 0;
    for (size_t i = 0; i < node_innovations.size(); i++) {
        int32_t innov = node_innovations[i];
        int32_t weight_count = node_weight_counts[i];

        auto it = node_by_innov.find(innov);
        if (it != node_by_innov.end() && it->second->is_enabled()) {
            vector<double> weights;
            it->second->get_weights(weights);
            if ((int32_t) weights.size() != weight_count) {
                Log::error(
                    "SomVectorSpace::encode_genome: node innovation %d has %d weights, catalog expects %d\n", innov,
                    (int32_t) weights.size(), weight_count
                );
            }
            size_t copy_count = (size_t) std::min(weight_count, (int32_t) weights.size());
            for (size_t w = 0; w < copy_count; w++) {
                node_vector[offset + w] = weights[w];
            }
        }
        // else: leave zeros (missing or disabled)
        offset += (size_t) weight_count;
    }

    map<int32_t, double> edge_weight_by_innov;
    unordered_set<int32_t> enabled_edges;

    for (RNN_Edge* edge : genome->edges) {
        int32_t innov = edge->get_innovation_number();
        edge_weight_by_innov[innov] = edge->weight;
        if (edge->is_enabled()) {
            enabled_edges.insert(innov);
        }
    }
    for (RNN_Recurrent_Edge* edge : genome->recurrent_edges) {
        int32_t innov = edge->get_innovation_number();
        edge_weight_by_innov[innov] = edge->weight;
        if (edge->is_enabled()) {
            enabled_edges.insert(innov);
        }
    }

    for (size_t i = 0; i < edge_innovations.size(); i++) {
        int32_t innov = edge_innovations[i];
        if (enabled_edges.find(innov) != enabled_edges.end()) {
            edge_vector[i] = edge_weight_by_innov[innov];
        }
        // else: leave 0 (missing or disabled)
    }
}

bool SomVectorSpace::compress(const vector<RNN_Genome*>& genomes, vector<SomUnit>& units) {
    if (node_innovations.empty() && edge_innovations.empty()) {
        return false;
    }

    unordered_set<int32_t> active_nodes;
    unordered_set<int32_t> active_edges;

    for (RNN_Genome* genome : genomes) {
        if (genome == nullptr) {
            continue;
        }
        for (RNN_Node_Interface* node : genome->nodes) {
            if (node->is_enabled()) {
                active_nodes.insert(node->get_innovation_number());
            }
        }
        for (RNN_Edge* edge : genome->edges) {
            if (edge->is_enabled()) {
                active_edges.insert(edge->get_innovation_number());
            }
        }
        for (RNN_Recurrent_Edge* edge : genome->recurrent_edges) {
            if (edge->is_enabled()) {
                active_edges.insert(edge->get_innovation_number());
            }
        }
    }

    bool removed = false;

    // Remove inactive node innovations from the back forward so offsets stay valid.
    for (int32_t i = (int32_t) node_innovations.size() - 1; i >= 0; i--) {
        int32_t innov = node_innovations[i];
        if (active_nodes.find(innov) != active_nodes.end()) {
            continue;
        }

        size_t offset = node_offset(i);
        size_t count = (size_t) node_weight_counts[i];
        for (SomUnit& unit : units) {
            unit.erase_node_slice(offset, count);
        }

        node_innovations.erase(node_innovations.begin() + i);
        node_weight_counts.erase(node_weight_counts.begin() + i);
        removed = true;
    }

    for (int32_t i = (int32_t) edge_innovations.size() - 1; i >= 0; i--) {
        int32_t innov = edge_innovations[i];
        if (active_edges.find(innov) != active_edges.end()) {
            continue;
        }

        size_t offset = edge_offset(i);
        for (SomUnit& unit : units) {
            unit.erase_edge_slice(offset, 1);
        }

        edge_innovations.erase(edge_innovations.begin() + i);
        removed = true;
    }

    if (removed) {
        node_index.clear();
        for (size_t i = 0; i < node_innovations.size(); i++) {
            node_index[node_innovations[i]] = (int32_t) i;
        }
        edge_index.clear();
        for (size_t i = 0; i < edge_innovations.size(); i++) {
            edge_index[edge_innovations[i]] = (int32_t) i;
        }
        recompute_sizes();
    }

    return removed;
}

double SomVectorSpace::euclidean(const vector<double>& a, const vector<double>& b) {
    size_t n = std::min(a.size(), b.size());
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    // Treat length mismatch as distance against zeros for the extra tail.
    if (a.size() > b.size()) {
        for (size_t i = n; i < a.size(); i++) {
            sum += a[i] * a[i];
        }
    } else if (b.size() > a.size()) {
        for (size_t i = n; i < b.size(); i++) {
            sum += b[i] * b[i];
        }
    }
    return std::sqrt(sum);
}

double SomVectorSpace::distance(
    const vector<double>& node_a, const vector<double>& edge_a, const vector<double>& node_b,
    const vector<double>& edge_b
) {
    return euclidean(node_a, node_b) + euclidean(edge_a, edge_b);
}

double SomVectorSpace::distance(
    const SomUnit& unit, const vector<double>& genome_node, const vector<double>& genome_edge
) {
    return distance(unit.node_vector, unit.edge_vector, genome_node, genome_edge);
}
