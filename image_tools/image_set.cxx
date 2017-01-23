#include <cmath>

#include <fstream>
using std::ifstream;

#include <iomanip>
using std::setw;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::ios;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_set.hxx"

#include "stdint.h"

Image::Image(ifstream &infile, int size, int _rows, int _cols, int _classification) {
    rows = _rows;
    cols = _cols;
    classification = _classification;

    pixels = new double*[cols];
    for (int32_t i = 0; i < cols; i++) {
        pixels[i] = new double[rows];
    }

    char* c_pixels = new char[size];

    infile.read( c_pixels, sizeof(char) * size);

    int current = 0;
    for (int32_t y = 0; y < cols; y++) {
        for (int32_t x = 0; x < rows; x++) {
            pixels[y][x] = (int)(uint8_t)c_pixels[current];
            current++;
        }
    }

    delete [] c_pixels;
}

double Image::get_pixel(int x, int y) const {
    return pixels[y][x];
}

int Image::get_classification() const {
    return classification;
}

int Image::get_rows() const {
    return rows;
}

int Image::get_cols() const {
    return cols;
}

void Image::scale_0_1() {
    for (int32_t y = 0; y < cols; y++) {
        for (int32_t x = 0; x < rows; x++) {
            pixels[y][x] /= 255.0;
        }
    }
}

double Image::get_pixel_avg() const {
    double avg = 0.0;

    for (int32_t y = 0; y < cols; y++) {
        for (int32_t x = 0; x < rows; x++) {
            avg += pixels[y][x];
        }
    }

    avg /= (rows * cols);

    return avg;
}

double Image::get_pixel_variance(double avg) const {
    double variance = 0.0;

    double tmp;
    for (int32_t y = 0; y < cols; y++) {
        for (int32_t x = 0; x < rows; x++) {
            tmp = avg - pixels[y][x];
            variance += tmp * tmp;
        }
    }

    variance /= (rows * cols);

    return variance;
}

void Image::normalize(double avg, double variance) {
    for (int32_t y = 0; y < cols; y++) {
        for (int32_t x = 0; x < rows; x++) {
            pixels[y][x] -= avg;
            pixels[y][x] /= variance;
        }
    }
}


void Image::print(ostream &out) {
    out << "Image Class: " << classification << endl;
    for (int32_t y = 0; y < cols; y++) {
        for (int32_t x = 0; x < rows; x++) {
            out << setw(7) << pixels[y][x];
        }
        out << endl;
    }
}

Images::Images(string binary_filename) {
    ifstream infile(binary_filename.c_str(), ios::in | ios::binary);

    if (!infile.is_open()) {
        cerr << "Could not open '" << binary_filename << "' for reading." << endl;
        return;
    }

    int initial_vals[3];
    infile.read( (char*)&initial_vals, sizeof(initial_vals) );

    number_classes = initial_vals[0];
    int rowscols = initial_vals[1];
    rows = rowscols;
    cols = rowscols;    //square for now

    vals_per_pixel = initial_vals[2];

    cerr << "number_classes: " << number_classes << endl;
    cerr << "rowscols: " << rowscols << endl;
    cerr << "vals_per_pixel: " << vals_per_pixel << endl;

    class_sizes = vector<int>(number_classes, 0);
    infile.read( (char*)&class_sizes[0], sizeof(int) * number_classes );

    int image_size = rowscols * rowscols * vals_per_pixel;

    for (int i = 0; i < number_classes; i++) {
        cerr << "reading image set with " << class_sizes[i] << " images." << endl;

        for (int32_t j = 0; j < class_sizes[i]; j++) {
            images.push_back(Image(infile, image_size, rowscols, rowscols, i));
        }
    }
    number_images = images.size();

    infile.close();

    cerr << "image_size: " << rowscols << "x" << rowscols << " = " << image_size << endl;

    cerr << "read " << images.size() << " images." << endl;
    for (int i = 0; i < (int32_t)class_sizes.size(); i++) {
        cerr << "    class " << setw(4) << i << ": " << class_sizes[i] << endl;
    }

    /*
       for (int i = 0; i < images.size(); i++) {
       images[i].print(cerr);
       }
       */

    cerr << "normalizing images." << endl;
    normalize();
    cerr << "normalized." << endl;
}

int Images::get_class_size(int i) const {
    return class_sizes[i];
}

int Images::get_number_classes() const {
    return number_classes;
}

int Images::get_number_images() const {
    return number_images;
}

int Images::get_image_rows() const {
    return rows;
}

int Images::get_image_cols() const {
    return cols;
}

const Image& Images::get_image(int image) const {
    return images[image];
}

void Images::normalize() {

    double avg = 0.0;

    for (int i = 0; i < number_images; i++) {
        images[i].scale_0_1();
        avg += images[i].get_pixel_avg();
    }

    avg /= number_images;
    
    cerr << "average pixel value: " << avg << endl;

    double variance = 0.0;
    for (int i = 0; i < number_images; i++) {
        variance += images[i].get_pixel_variance(avg);
    }

    variance /= number_images;
    cerr << "pixel variance: " << variance << endl;
    double std_dev = sqrt(variance);
    cerr << "pixel standard deviation: " << std_dev << endl;

    for (int i = 0; i < number_images; i++) {
        images[i].normalize(avg, std_dev);
    }
}
