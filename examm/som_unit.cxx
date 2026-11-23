#include "som_unit.hxx"

SomUnit::SomUnit(size_t node_size, size_t edge_size) : node_vector(node_size, 0.0), edge_vector(edge_size, 0.0) {
}

void SomUnit::resize(size_t node_size, size_t edge_size) {
    node_vector.assign(node_size, 0.0);
    edge_vector.assign(edge_size, 0.0);
}

void SomUnit::append_node_zeros(int32_t count) {
    node_vector.insert(node_vector.end(), (size_t) count, 0.0);
}

void SomUnit::append_edge_zeros(int32_t count) {
    edge_vector.insert(edge_vector.end(), (size_t) count, 0.0);
}

void SomUnit::erase_node_slice(size_t offset, size_t count) {
    if (count == 0) {
        return;
    }
    node_vector.erase(node_vector.begin() + (long) offset, node_vector.begin() + (long) (offset + count));
}

void SomUnit::erase_edge_slice(size_t offset, size_t count) {
    if (count == 0) {
        return;
    }
    edge_vector.erase(edge_vector.begin() + (long) offset, edge_vector.begin() + (long) (offset + count));
}
