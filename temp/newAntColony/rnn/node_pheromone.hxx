#ifndef NODE_PHEROMONES_HXX
#define NODE_PHEROMONES_HXX

#include <fstream>
using std::istream;
using std::ifstream;
using std::ostream;
using std::ofstream;

#include <vector>
using std::vector;

#include "edge_pheromone.hxx"


class NODE_Pheromones {
    private:
      int32_t node_innovation_number;

      double* type_pheromones;
      vector<EDGE_Pheromone*> *pheromone_lines;

      int layer_type;
      int current_layer;


    public:
      NODE_Pheromones(double *_node_pheromones, vector<EDGE_Pheromone*> *_pheromone_lines, int _layer_type, int _current_layer);

      int32_t get_layer_type();
      int32_t get_current_layer();
      vector<EDGE_Pheromone*>* get_pheromone_lines();


      ~NODE_Pheromones();

      friend class ACNNTO;
};


// double* get_pheromones();

#endif
