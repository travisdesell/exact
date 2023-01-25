#ifndef WEIGHT_RULES_HXX
#define WEIGHT_RULES_HXX

#include <string>
using std::string;
#include <vector>
using std::vector;

enum WeightType { RANDOM = 0, XAVIER = 1, KAIMING = 2, LAMARCKIAN = 3, NONE = -1 };

static string WEIGHT_TYPES_STRING[] = {"random", "xavier", "kaiming", "lamarckian"};
static int32_t NUM_WEIGHT_TYPES = 4;

inline WeightType get_enum_from_string(string input_string) {
    WeightType weight_type;
    for (int i = 0; i < NUM_WEIGHT_TYPES; i++) {
        if (input_string.compare(WEIGHT_TYPES_STRING[i]) == 0) { weight_type = static_cast<WeightType>(i); }
    }
    return weight_type;
}

template <typename Weight_Type>
int32_t enum_to_integer(Weight_Type weight) {
    return static_cast<typename std::underlying_type<Weight_Type>::type>(weight);
}

inline WeightType integer_to_enum(int32_t input_int) {
    WeightType weight_type = static_cast<WeightType>(input_int);
    return weight_type;
}

class WeightRules {
   private:
    WeightType weight_initialize;
    WeightType weight_inheritance;
    WeightType mutated_components_weight;

   public:
    WeightRules();
    explicit WeightRules(const vector<string> &arguments);

    void initialize_from_args(const vector<string> &arguments);
    WeightType get_weight_initialize_method();
    WeightType get_weight_inheritance_method();
    WeightType get_mutated_components_weight_method();

    string get_weight_initialize_method_name();
    string get_weight_inheritance_method_name();
    string get_mutated_components_weight_method_name();

    void set_weight_initialize_method(WeightType _weight_initialize);
    void set_weight_inheritance_method(WeightType _weight_inheritance);
    void set_mutated_components_weight_method(WeightType _mutated_components_weight);
    WeightRules *copy();
};

#endif
