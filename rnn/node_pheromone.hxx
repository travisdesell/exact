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

      vector<EDGE_Pheromone*> &pheromone_lines;

      double* type_pheromones;

    public:
      NODE_Pheromones(double *_node_pheromones, vector<EDGE_Pheromone*> &_pheromone_lines);
      // NODE_Pheromones(double* _type_pheromones);

      ~NODE_Pheromones();

      friend class ANT_COLONY;
      friend class TT;
};


// double* get_pheromones();

#endif
