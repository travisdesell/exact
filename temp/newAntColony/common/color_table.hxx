#pragma once

#include <cstdint>

struct Color {
    int16_t red;
    int16_t green;
    int16_t blue;
};

Color get_colormap(double value);
