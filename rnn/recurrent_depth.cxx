#include "recurrent_depth.hxx"
#include "common/log.hxx"
#include "rnn_genome.hxx"

Recurrent_Depth::Recurrent_Depth(int32_t _type, int32_t _min_recurrent_depth, int32_t _max_recurrent_depth, bool _various_recurrent_depth) : 
                                    type(_type), min_recurrent_depth(_min_recurrent_depth), max_recurrent_depth(_max_recurrent_depth) {

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    dist = uniform_int_distribution <int32_t> (min_recurrent_depth, max_recurrent_depth);
    if (type == EDGE_RECURRENT_DEPTH) {
        various_recurrent_depth = false;
    } else if (type == NODE_RECURRENT_DEPTH) {
        various_recurrent_depth = _various_recurrent_depth;
    } else {
        Log::fatal("wrong recurrent depth type %d, this should never happen! \n", type);
        exit(1);
    }

}

Recurrent_Depth::~Recurrent_Depth() {

}

int32_t Recurrent_Depth::get_recurrent_depth() {
    if (type == NODE_RECURRENT_DEPTH) {
        return get_node_recurrent_depth();
    } else if(type == EDGE_RECURRENT_DEPTH) {
        return get_edge_recurrent_depth();
    } else {
        Log::fatal("Unknown recurrent depth type %d, this should never happen!\n", type);
        exit(1);
    } 
}

int32_t Recurrent_Depth::get_edge_recurrent_depth() {
    if(type == EDGE_RECURRENT_DEPTH) {
        // return dist(generator);
    } else {
        Log::fatal("The recurrent depth type %d is not edge recurrent depth!\n", type);
        exit(1);
    }
    return dist(generator);
}

int32_t Recurrent_Depth::get_node_recurrent_depth() {
    int32_t node_recurrent_depth = 0;
    if (type == NODE_RECURRENT_DEPTH) {
        if (various_recurrent_depth) {
            node_recurrent_depth = dist.max();
            Log::debug("various node depth: %d \n", node_recurrent_depth);
        } else {
            node_recurrent_depth = dist(generator);
            Log::debug("recurrent depth %d \n", node_recurrent_depth);
        }
    } else {
        Log::fatal("The recurrent depth type %d is not node recurrent depth!\n", type);
        exit(1);
    }

    return node_recurrent_depth;
}

Recurrent_Depth* Recurrent_Depth::copy() {
    Recurrent_Depth* copy = new Recurrent_Depth(type, min_recurrent_depth, max_recurrent_depth, various_recurrent_depth);

    return copy;
}

int32_t Recurrent_Depth::get_recurrent_depth_type() {
    return type;
}

bool Recurrent_Depth::is_various_recurrent_depth() {
    return various_recurrent_depth;
}

bool Recurrent_Depth::has_recurrent_depth() {
    if (min_recurrent_depth <= 0 && max_recurrent_depth <=0) {
        return false;
    } else return true;
}

void Recurrent_Depth::write_to_stream(ostream &out_stream) {
    out_stream.write((char*)&min_recurrent_depth, sizeof(int32_t));
    out_stream.write((char*)&max_recurrent_depth, sizeof(int32_t));
    out_stream.write((char*)&type, sizeof(int32_t));
    out_stream.write((char*)&various_recurrent_depth, sizeof(bool));
}