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

  Tokenizer tokenizer(s);

  optional<Token> token;

  while ((token = tokenizer.next_token()) != std::nullopt) {
    std::cout << token->debug() << "\n";
  }

  return 0;
}
