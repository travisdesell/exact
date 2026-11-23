#ifndef EXAMM_SOM_VECTOR_SPACE_HXX
#define EXAMM_SOM_VECTOR_SPACE_HXX

#include <cstdint>
#include <map>
using std::map;

#include <unordered_set>
using std::unordered_set;

#include <vector>
using std::vector;

#include "som_unit.hxx"

class RNN_Genome;

/**
 * Shared innovation catalog for SOM encoding.
 *
 * Node and edge innovations are appended in first-seen order. Each node
 * innovation records how many weights that node type contributes so missing
 * or disabled genes can be zero-padded to a fixed layout.
 *
 * On every insert, call expand_from_genome() so new innovations grow all
 * SomUnit vectors with trailing zeros. Periodically call compress() to drop
 * innovations that are absent or disabled across every living genome.
 */
class SomVectorSpace {
   private:
    // Ordered by first appearance; new innovations are always appended.
    vector<int32_t> node_innovations;
    vector<int32_t> node_weight_counts;  // parallel to node_innovations
    vector<int32_t> edge_innovations;   // feedforward + recurrent

    map<int32_t, int32_t> node_index;  // innovation -> index in node_innovations
    map<int32_t, int32_t> edge_index;  // innovation -> index in edge_innovations

    size_t node_vector_size;
    size_t edge_vector_size;

    void recompute_sizes();
    size_t node_offset(int32_t catalog_index) const;
    size_t edge_offset(int32_t catalog_index) const;

   public:
    SomVectorSpace();

    size_t get_node_vector_size() const;
    size_t get_edge_vector_size() const;
    const vector<int32_t>& get_node_innovations() const;
    const vector<int32_t>& get_edge_innovations() const;

    /**
     * Register any innovations present in the genome that are not yet in the
     * catalog. Appends matching zero weights to every SomUnit in units.
     * Returns true if the catalog grew.
     */
    bool expand_from_genome(RNN_Genome* genome, vector<SomUnit>& units);

    /**
     * Build innovation-aligned node/edge vectors for a genome. Missing or
     * disabled genes contribute zeros of the catalogued width.
     */
    void encode_genome(RNN_Genome* genome, vector<double>& node_vector, vector<double>& edge_vector) const;

    /**
     * Remove innovations that are missing or disabled in every genome.
     * Splices the same slices out of every SomUnit. Returns true if anything
     * was removed.
     */
    bool compress(const vector<RNN_Genome*>& genomes, vector<SomUnit>& units);

    /**
     * distance(node_a, node_b) + distance(edge_a, edge_b) using Euclidean L2.
     */
    static double distance(
        const vector<double>& node_a, const vector<double>& edge_a, const vector<double>& node_b,
        const vector<double>& edge_b
    );

    static double distance(const SomUnit& unit, const vector<double>& genome_node, const vector<double>& genome_edge);

    static double euclidean(const vector<double>& a, const vector<double>& b);
};

#endif
