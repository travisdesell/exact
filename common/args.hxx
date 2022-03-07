#ifndef ENV_ARGS_HXX
#define ENV_ARGS_HXX

#include <functional>
using std::function;

#include <memory>
using std::unique_ptr;

#include <map>
using std::map;

#include <string>
using std::string;

#include <optional>
using std::optional;

#include <variant>
using std::get;
using std::variant;

#include <initializer_list>
using std::initializer_list;

#include <utility>
using std::pair;

#include <vector>
using std::vector;

string to_lower(const string &s);
void init_args(int argc, char **argv);

class ArgumentSet;

struct Argument {
  typedef variant<int, string, vector<int>, vector<string>, bool, double, vector<double>> data;
  string name;
  string flag;
  string description;
  bool required;
  data default_value;
  enum arg_type { INT = 0, STRING = 1, INT_LIST = 2, STRING_LIST = 3, BOOL = 4, DOUBLE = 5, DOUBLE_LIST = 6 } ty;

  int get_int();
  string get_string();
  vector<int> get_int_list();
  vector<string> get_string_list();
  bool get_bool();
  double get_double();
  vector<double> get_double_list();
  int get_bitflags();

  bool was_parsed();

  Argument(string name, string flag, string description, bool required = false, arg_type ty = BOOL,
           data default_value = false);
  Argument(Argument &) = default;
  virtual ~Argument();

 protected:
  data value;
  bool parsed;

  virtual Argument *clone();
  void parsed_check();
  virtual void accept(vector<string> values);
  static int try_parse_int(const string &);
  static double try_parse_double(const string &s);

  friend class ArgumentSet;
};

struct EnumArgument : public Argument {
  // Maps string to enum.
  map<string, int> data_map;
  bool case_sensitive;

  EnumArgument(string name, string flag, string description, bool required, data default_value,
               map<string, int> data_map, bool case_sensitive = false);
  EnumArgument(EnumArgument &) = default;
  virtual ~EnumArgument();

 private:
  virtual void accept(vector<string> values);
  string to_string();
  int find(const string &s);

  virtual EnumArgument *clone();
};

struct ConstrainedArgument : public Argument {
  function<bool(Argument::data &)> constraint;

  ConstrainedArgument(string name, string flag, string description, function<bool(Argument::data &)> constraint,
                      bool required = false, arg_type ty = BOOL, data default_value = false);
  ConstrainedArgument(ConstrainedArgument &) = default;
  virtual ~ConstrainedArgument();

 private:
  virtual void accept(vector<string> values);
  virtual ConstrainedArgument *clone();
};

class ArgumentSet {
  const string name;
  function<bool (ArgumentSet &)> validate;

  static inline vector<string> argv;

 public:
  map<string, unique_ptr<Argument>> args;
  ArgumentSet(string name, initializer_list<Argument *>, function<bool (ArgumentSet &)> validate=[](ArgumentSet &){ return true; } );
  ArgumentSet(string name, map<string, unique_ptr<Argument>> args, function<bool (ArgumentSet &)> validate);
  static void set_argv(vector<string> argv);

  // Make sure there is no overlap between the two.
  ArgumentSet operator+(const ArgumentSet &other);

  void parse();
};

#endif
