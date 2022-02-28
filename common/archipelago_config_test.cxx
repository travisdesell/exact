#include <iostream>
#include <fstream>
#include <sstream>
#include "archipelago_config.hxx"

int main(int argc, char **argv) {
  vector<string> arguments = vector<string>(argv, argv + argc);

  Log::initialize(arguments);
  Log::set_id("main");

  std::ifstream t(argv[1]);
  std::stringstream buffer;
  buffer << t.rdbuf();

  string s = buffer.str();

  int nn = 32;
  map<string, node_index_type> m;
  m["n_islands"] = 15;
  ArchipelagoConfig config = ArchipelagoConfig::from_string(s, nn, m);

  for (int i = 0; i < nn; i++) {
    Log::info("");
    for (int j = 0; j < nn; j++) {
      Log::info_no_header("%s ", config.connections[i][j] ? "X" : "~");
    }
    Log::info_no_header("\n");
  }

  return 0;
}
