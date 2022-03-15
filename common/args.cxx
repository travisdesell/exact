#include "args.hxx"

#include <algorithm>
#include <cctype>

#include "log.hxx"

string to_lower(const string &s) {
  static auto tolower = [](unsigned char c) { return std::tolower(c); };
  string lower = s;
  std::transform(lower.begin(), lower.end(), lower.begin(), tolower);
  return lower;
}

bool Argument::was_parsed() { return parsed; }

void Argument::parsed_check() {
  if (!parsed && required) {
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

double Argument::get_double() {
  parsed_check();
  if (ty == DOUBLE) {
    return get<double>(value);
  } else {
    Log::fatal("ERROR: Argument %s has type %d but found type %d\n", name.c_str(), ty, value.index());
    exit(1);
  }
}

vector<double> Argument::get_double_list() {
  parsed_check();
  if (ty == DOUBLE_LIST) {
    return get<vector<double>>(value);
  } else {
    Log::fatal("ERROR: Argument %s has type %d but found type %d\n", name.c_str(), ty, value.index());
    exit(1);
  }
}

Argument::Argument(string name, string flag, string description, bool required, arg_type ty, data default_value)
    : name(name),
      flag(flag),
      description(description),
      required(required),
      default_value(default_value),
      ty(ty),
      value(default_value),
      parsed(false) {}

Argument::~Argument() {}

Argument *Argument::clone() { return new Argument(*this); }

int Argument::try_parse_int(const string &s) {
  int x;
  try {
    x = std::stoi(s);
  } catch (std::invalid_argument const &ex) {
    Log::fatal("Invalid integer \"%s\"", s.c_str());
    exit(1);
  } catch (std::out_of_range const &ex) {
    Log::fatal("Invalid integer \"%s\", too large to fit in an int.\n", s.c_str());
    exit(1);
  }

  return x;
}

double Argument::try_parse_double(const string &s) {
  double x;
  try {
    x = std::stod(s);
  } catch (std::invalid_argument const &ex) {
    Log::fatal("Invalid double \"%s\"", s.c_str());
    exit(1);
  } catch (std::out_of_range const &ex) {
    Log::fatal("Invalid double \"%s\", too large to fit in a double.\n", s.c_str());
    exit(1);
  }

  return x;
}

void Argument::accept(vector<string> values) {
  parsed = true;
  switch (ty) {
    case INT:
      if (values.size() != 1) {
        Log::fatal("Argument %s expected a single number, but got %d values. Aborting. \n", flag.c_str(),
                   values.size());
        exit(1);
      }
      value = try_parse_int(values[0]);
      break;
    case STRING:
      if (values.size() != 1) {
        Log::fatal("Argument %s expected a single string, but got %d values. Aborting. \n", flag.c_str(),
                   values.size());
        exit(1);
      }
      value = values[0];
      break;
    case INT_LIST: {
      vector<int> ns;
      for (int i = 0; i < (int) values.size(); i++) ns.push_back(try_parse_int(values[i]));
      value = move(ns);
    } break;
    case STRING_LIST:
      value = move(values);
      break;
    case BOOL:
      if (values.size() != 0) {
        Log::fatal("Argument %s is a boolean, but you supplied %d value(s) to it. Aborting.\n", flag.c_str(),
                   values.size());
        exit(1);
      }
      value = true;
      break;
    case DOUBLE:
      if (values.size() != 1) {
        Log::fatal("Argument %s expected a single number, but got %d values. Aborting. \n", flag.c_str(),
                   values.size());
        exit(1);
      }

      value = try_parse_double(values[0]);
      break;
    case DOUBLE_LIST: {
      vector<double> ns;
      for (int i = 0; i < (int) values.size(); i++) ns.push_back(try_parse_double(values[i]));
      value = move(ns);
    } break;
  }
}

EnumArgument::EnumArgument(string name, string flag, string description, bool required, Argument::data default_value,
                           map<string, int> data_map, bool case_sensitive)
    : Argument(name, flag, description, required, (arg_type) default_value.index(), default_value),
      case_sensitive(case_sensitive) {
  if (ty != INT && ty != INT_LIST && ty != STRING && ty != STRING_LIST) {
    Log::fatal("EnumArgument %s must have a default value of int or int list. Aborting. \n", flag.c_str());
    exit(1);
  }

  if (case_sensitive)
    this->data_map = move(data_map);
  else {
    for (auto it = data_map.begin(); it != data_map.end(); it++) {
      this->data_map.insert({to_lower(it->first), it->second});
    }
  }
  switch (default_value.index()) {
    case STRING:
      ty = INT;
      accept({get<string>(default_value)});
      break;
    case STRING_LIST:
      ty = INT_LIST;
      accept(get<vector<string>>(default_value));
    default:;
  }

  parsed = false;
}

EnumArgument::~EnumArgument() {}

EnumArgument *EnumArgument::clone() { return new EnumArgument(*this); }

int Argument::get_bitflags() {
  if (ty == Argument::INT) return get_int();

  vector<int> &values = get<vector<int>>(value);
  int carry = 0;
  for (auto x : values) carry |= x;
  return carry;
}

void EnumArgument::accept(vector<string> values) {
  parsed = true;
  if (!case_sensitive) {
    for (int i = 0; i < (int) values.size(); i++) values[i] = to_lower(values[i]);
  }

  if (ty == INT) {
    if (values.size() != 1) {
      Log::fatal("Argument %s expects a single number, but got %d values. Aborting. \n", flag.c_str(), values.size());
      exit(1);
    }

    value = find(values[0]);
  } else if (ty == INT_LIST) {
    vector<int> enum_values;
    if (values.size() == 0) Log::warning("Found zero arguments for flag %s\n", flag.c_str());

    for (auto it = values.begin(); it != values.end(); it++) enum_values.push_back(find(*it));

    value = enum_values;
  } else {
    Log::fatal(
        "EnumArgument %s should have type INT or INT_LIST. Change the default argument to a int or vector<int>.\n",
        flag.c_str());
  }

  parsed = true;
}

string EnumArgument::to_string() {
  string s;
  for (auto it = data_map.begin(); it != data_map.end(); it++) { s += it->first + ", "; }
  s.pop_back();
  s.pop_back();
  return s;
}

int EnumArgument::find(const string &s) {
  auto it = data_map.find(s);
  if (it == data_map.end()) {
    Log::fatal("Invalid enum value %s supplied for flag %s. Valid values are %s. Aborting. \n", s.c_str(), flag.c_str(),
               to_string().c_str());
    exit(1);
  }
  return it->second;
}

ConstrainedArgument::ConstrainedArgument(string name, string flag, string description,
                                         function<bool(Argument::data &)> constraint, bool required, arg_type ty,
                                         data default_value)
    : Argument(name, flag, description, required, ty, default_value), constraint(constraint) {}

ConstrainedArgument::~ConstrainedArgument() {}

ConstrainedArgument *ConstrainedArgument::clone() { return new ConstrainedArgument(*this); }

void ConstrainedArgument::accept(vector<string> values) {
  Argument::accept(move(values));

  if (!constraint(this->value)) {
    Log::fatal("Argument %s failed to meet its constraints. Aborting.\n", flag.c_str());
    exit(1);
  }
}

ArgumentSet::ArgumentSet(string name, initializer_list<Argument *> initializers, function<bool(ArgumentSet &)> validate)
    : name(name), validate(validate) {
  for (auto it = initializers.begin(); it != initializers.end(); it++) {
    auto [_, inserted] = args.insert({(*it)->name, unique_ptr<Argument>(*it)});
    if (!inserted) {
      Log::fatal("ERROR: found two definitions for the argument %s\n", (*it)->name.c_str());
      exit(1);
    }
  }
}

ArgumentSet::ArgumentSet(string name, map<string, unique_ptr<Argument>> args, function<bool(ArgumentSet &)> validate)
    : name(name), args(move(args)), validate(validate) {}

ArgumentSet ArgumentSet::operator+(const ArgumentSet &other) {
  map<string, unique_ptr<Argument>> new_args;

  for (const auto &it : this->args) new_args.emplace(it.first, unique_ptr<Argument>(it.second->clone()));

  for (const auto &it : other.args) {
    auto [_, inserted] = new_args.emplace(it.first, unique_ptr<Argument>(it.second->clone()));
    if (!inserted) { Log::fatal("ERROR: failed to combine two argument sets \n"); }
  }

  string new_name = name + "+" + other.name;

  auto v_copy = validate;
  auto other_v_copy = other.validate;
  auto new_validate = [=](ArgumentSet &as) { return v_copy(as) && other_v_copy(as); };

  return ArgumentSet(new_name, move(new_args), new_validate);
}

void ArgumentSet::set_argv(vector<string> argv) { ArgumentSet::argv = argv; }

void ArgumentSet::parse() {
  map<string, int> flag_indices;
  for (int i = 0; i < (int) argv.size(); i++) {
    if (argv[i].rfind("--", 0) == 0) { flag_indices[argv[i]] = i; }
  }

  for (auto it = args.begin(); it != args.end(); it++) {
    auto flag_index_it = flag_indices.find(it->first);
    if (flag_index_it == flag_indices.end()) {
      if (it->second->required) {
        Log::fatal("Failed to find required argument %s. Aborting.\n", it->second->flag.c_str());
        exit(1);
      }
      continue;
    }

    int i = flag_index_it->second;
    vector<string> x;
    while (++i < (int) argv.size()) {
      if (argv[i].rfind("--", 0) == 0) break;
      x.push_back(argv[i]);
    }

    it->second->accept(move(x));
  }

  if (!validate(*this)) { Log::fatal("Failed to validate argument set %s. Aborting.\n", name.c_str()); }
}
