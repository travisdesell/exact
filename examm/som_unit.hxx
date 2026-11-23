#ifndef EXAMM_SOM_UNIT_HXX
#define EXAMM_SOM_UNIT_HXX

#include <cstdint>
#include <vector>
using std::vector;

/**
 * One SOM map unit. Holds separate innovation-aligned weight vectors for
 * nodes and edges. Vector length grows/shrinks with the SomVectorSpace catalog.
 */
class SomUnit {
   public:
    vector<double> node_vector;
    vector<double> edge_vector;

    SomUnit() = default;
    SomUnit(size_t node_size, size_t edge_size);

    void resize(size_t node_size, size_t edge_size);
    void append_node_zeros(int32_t count);
    void append_edge_zeros(int32_t count);
    void erase_node_slice(size_t offset, size_t count);
    void erase_edge_slice(size_t offset, size_t count);
};

#endif
