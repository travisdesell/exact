#include "args.hxx"
#include "log.hxx"

void Argument::parsed_check() {
  if (!parsed) {
    Log::fatal("ERROR: You must parse arguments before accessing them. See ArgumentSet::parse\n");
    exit(1);
  }
}

int Argument::get_int() {
  parsed_check();
  if (ty == INT) {
    return get<int>(value);
  } else {
    Log::fatal("ERROR: Argument %s has type %d but found type %d\n", name.c_str(), ty, value.index());
    exit(1);
  }
}

string Argument::get_string() {
  parsed_check();
  if (ty == STRING) {
    return get<string>(value);
  } else {
    Log::fatal("ERROR: Argument %s has type %d but found type %d\n", name.c_str(), ty, value.index());
    exit(1);
  }
}

vector<int> Argument::get_int_list() {
  parsed_check();
  if (ty == INT_LIST) {
    return get<vector<int>>(value);
  } else {
    Log::fatal("ERROR: Argument %s has type %d but found type %d\n", name.c_str(), ty, value.index());
    exit(1);
  }
}

vector<string> Argument::get_string_list() {
  parsed_check();
  if (ty == STRING_LIST) {
    return get<vector<string>>(value);
  } else {
    Log::fatal("ERROR: Argument %s has type %d but found type %d\n", name.c_str(), ty, value.index());
    exit(1);
  }
}

bool Argument::get_bool() {
  parsed_check();
  if (ty == BOOL) {
    return get<bool>(value);
  } else {
    Log::fatal("ERROR: Argument %s has type %d but found type %d\n", name.c_str(), ty, value.index());
    exit(1);
  }
}

Argument::Argument(string name, string flag, string description, bool required, data default_value) : name(name), flag(flag), description(description), required(required), default_value(default_value), value(default_value), parsed(false) {}

ArgumentSet::ArgumentSet(initializer_list<pair<string, Argument>> initializers) {
  for (auto it = initializers.begin(); it != initializers.end(); it++) {
    auto[_, inserted] = args.insert(*it);
    if (!inserted) {
      Log::fatal("ERROR: found two definitions for the argument %s\n", it->first.c_str());
      exit(1);
    }
  }
}

ArgumentSet::ArgumentSet(map<string, Argument> args) : args(move(args)) {}

ArgumentSet ArgumentSet::operator+(const ArgumentSet &other) {
  map<string, Argument> new_args = args;
  for (auto it = other.args.begin(); it != other.args.end(); it++) {
    auto[_, inserted] = new_args.insert(*it);
    if (!inserted) {
      Log::fatal("ERROR: failed to combine two argument sets \n");
    }
  }
  return ArgumentSet(move(new_args));
}

void ArgumentSet::set_argv(vector<string> argv) {
  ArgumentSet::argv = argv;
}

void ArgumentSet::parse() {
  exit(1);
}
