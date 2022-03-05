#ifndef ENV_ARGS_HXX
#define ENV_ARGS_HXX

#include <map>
using std::map;

#include <string>
using std::string;

#include <optional>
using std::optional;

#include <variant>
using std::variant;
using std::get;

#include <initializer_list>
using std::initializer_list;

#include <utility>
using std::pair;

#include <vector>
using std::vector;

void init_args(int argc, char **argv);

struct Argument {
  typedef variant<int, string, vector<int>, vector<string>, bool> data;
  string name;
  string flag;
  string description;
  bool required;
  data default_value;
  enum arg_type { INT = 0, STRING = 1, INT_LIST = 2, STRING_LIST = 3, BOOL = 4 } ty;

  int get_int();
  string get_string();
  vector<int> get_int_list();
  vector<string> get_string_list();
  bool get_bool();

  Argument(string name, string flag, string description, bool required, data default_value);

 private:
  data value;
  bool parsed;

  void parsed_check();
};

class ArgumentSet {
  map<string, Argument> args;

  static vector<string> argv;
 public:
  ArgumentSet(initializer_list<pair<string, Argument>>);
  ArgumentSet(map<string, Argument> args);
  static void set_argv(vector<string> argv);

  // Make sure there is no overlap between the two.
  ArgumentSet operator+(const ArgumentSet &other);

  void parse();
};

#endif
