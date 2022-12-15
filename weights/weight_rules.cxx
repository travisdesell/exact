#include "weights/weight_rules.hxx"
#include "common/arguments.hxx"
#include "common/log.hxx"

WeightRules::WeightRules() {
    weight_initialize = XAVIER;
    weight_inheritance = LAMARCKIAN;
    mutated_components_weight = LAMARCKIAN;
}

void WeightRules::generate_weight_initialize_from_arguments(const vector<string> &arguments) {
    string weight_initialize_string = "xavier";
    get_argument(arguments, "--weight_initialize", false, weight_initialize_string);
    weight_initialize = get_enum_from_string(weight_initialize_string);

    string weight_inheritance_string = "lamarckian";
    get_argument(arguments, "--weight_inheritance", false, weight_inheritance_string);
    weight_inheritance = get_enum_from_string(weight_inheritance_string);

    string mutated_component_weight_string = "lamarckian";
    get_argument(arguments, "--mutated_component_weight", false, mutated_component_weight_string);
    mutated_components_weight = get_enum_from_string(mutated_component_weight_string);
}

WeightType WeightRules::get_weight_initialize_method() {
    return weight_initialize;
}

WeightType WeightRules::get_weight_inheritance_method() {
    return weight_inheritance;
}

WeightType WeightRules::get_mutated_components_weight_method() {
    return mutated_components_weight;
}

void WeightRules::set_weight_initialize_method(WeightType _weight_initialize) {
    weight_initialize = _weight_initialize;
}

void WeightRules::set_weight_inheritance_method(WeightType _weight_inheritance) {
    weight_inheritance = _weight_inheritance;
}

void WeightRules::set_mutated_components_weight_method(WeightType _mutated_components_weight) {
    mutated_components_weight = _mutated_components_weight;
}

string WeightRules::get_weight_initialize_method_name() {
    return WEIGHT_TYPES_STRING[weight_initialize];
}
string WeightRules::get_weight_inheritance_method_name() {
    return WEIGHT_TYPES_STRING[weight_inheritance];
}
string WeightRules::get_mutated_components_weight_method_name() {
    return WEIGHT_TYPES_STRING[mutated_components_weight];
}

WeightRules* WeightRules::copy() {
    WeightRules *weight_rule_copy = new WeightRules();
    weight_rule_copy->set_weight_initialize_method(weight_initialize);
    weight_rule_copy->set_weight_inheritance_method(weight_inheritance);
    weight_rule_copy->set_mutated_components_weight_method(mutated_components_weight);
    
    return weight_rule_copy;
}