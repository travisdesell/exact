#ifndef WEIGHT_INITIALIZE_HXX
#define WEIGHT_INITIALIZE_HXX


enum WeightType {
    RANDOM = 0, 
    XAVIER = 1, 
    KAIMING = 2, 
    LAMARCKIAN = 3,
    NONE = -1
};

static string WEIGHT_TYPES_STRING[] = {"random", "xavier", "kaiming", "lamarckian"};
static int32_t NUM_WEIGHT_TYPES = 4;


inline WeightType get_enum_from_string(string input_string) {
    WeightType weight_type;
    for (int i = 0; i < NUM_WEIGHT_TYPES; i++) {
        if (input_string.compare(WEIGHT_TYPES_STRING[i]) == 0) {
            weight_type = static_cast<WeightType>(i);
        }
    }
    return weight_type;
}

template <typename Weight_Type>
int32_t enum_to_integer(Weight_Type weight) {
    return static_cast<typename std::underlying_type<Weight_Type>::type>(weight);
}

inline WeightType integer_to_enum (int32_t input_int) {
    WeightType weight_type = static_cast<WeightType>(input_int);
    return weight_type;
}



#endif
