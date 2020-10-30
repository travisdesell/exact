#ifndef EXAMM_RECURRENT_DEPTH_HXX
#define EXAMM_RECURRENT_DEPTH_HXX

#include <cstdint>

#include <fstream>
using std::ostream;

#include <random>
using std::minstd_rand0;
using std::uniform_int_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#define NODE_RECURRENT_DEPTH 0
#define EDGE_RECURRENT_DEPTH 1

class Recurrent_Depth {
    private:
        int32_t min_recurrent_depth;
        int32_t max_recurrent_depth;
        int32_t type;
        bool various_recurrent_depth;
        minstd_rand0 generator;
        uniform_int_distribution<int32_t> dist;

    public:

        Recurrent_Depth(int32_t _type, int32_t _min_recurrent_depth, int32_t _max_recurrent_depth, bool _various_recurrent_depth);
        ~Recurrent_Depth();
        int32_t get_recurrent_depth();
        int32_t get_node_recurrent_depth();
        int32_t get_edge_recurrent_depth();
        int32_t get_recurrent_depth_type();
        bool is_various_recurrent_depth();

        Recurrent_Depth* copy();
        void write_to_stream(ostream &out);

    friend class RNN_Genome;
    friend class EXAMM;

};

#endif
