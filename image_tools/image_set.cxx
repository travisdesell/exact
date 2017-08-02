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

Image::Image(ifstream &infile, int _channels, int _width, int _height, int _padding, int _classification, const Images *_images) {
    channels = _channels;
    width = _width;
    height = _height;
    padding = _padding;
    classification = _classification;
    images = _images;

    pixels = vector< vector< vector<uint8_t> > >(channels, vector< vector<uint8_t> >(height, vector<uint8_t>(width, 0)));

    char* c_pixels = new char[channels * width * height];

    infile.read( c_pixels, sizeof(char) * channels * width * height);

    int current = 0;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                pixels[z][y][x] = (uint8_t)c_pixels[current];
                current++;
            }
        }
    }

    delete [] c_pixels;
}

float Image::get_pixel(int z, int y, int x) const {
    if (y < padding || x < padding) return 0;
    else if (y >= width + padding || x >= height + padding) return 0;
    else {
        return ((pixels[z][y - padding][x - padding] / 255.0) - images->get_channel_avg(z)) / images->get_channel_std_dev(z);
    }
}

int Image::get_classification() const {
    return classification;
}

void Image::get_pixel_avg(vector<float> &channel_avgs) const {
    channel_avgs.clear();
    channel_avgs.assign(channels, 0.0);

    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                channel_avgs[z] += pixels[z][y][x] / 255.0;
            }
        }
        channel_avgs[z] /= (height * width);
    }
}

void Image::get_pixel_variance(const vector<float> &channel_avgs, vector<float> &channel_variances) const {
    channel_variances.clear();
    channel_variances.assign(channels, 0.0);

    float tmp;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                tmp = channel_avgs[z] - (pixels[z][y][x] / 255.0);
                channel_variances[z] += tmp * tmp;
            }
        }

        channel_variances[z] /= (height * width);
    }
}

void Image::print(ostream &out) {
    out << "Image Class: " << classification << endl;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                out << setw(7) << pixels[z][y][x];
            }
            out << endl;
        }
    }
}

string Images::get_filename() const {
    return filename;
}

int Images::read_images(string _filename) {
    filename = _filename;

    ifstream infile(filename.c_str(), ios::in | ios::binary);

    if (!infile.is_open()) {
        cerr << "Could not open '" << filename << "' for reading." << endl;
        return 1;
    }

    int initial_vals[4];
    infile.read( (char*)&initial_vals, sizeof(initial_vals) );

    number_classes = initial_vals[0];
    channels = initial_vals[1];
    width = initial_vals[2];
    height = initial_vals[3];

    cerr << "number_classes: " << number_classes << endl;
    cerr << "channels: " << channels << endl;
    cerr << "width: " << width << endl;
    cerr << "height: " << height << endl;

    class_sizes = vector<int>(number_classes, 0);
    infile.read( (char*)&class_sizes[0], sizeof(int) * number_classes );

    int image_size = channels * width * height;

    for (int i = 0; i < number_classes; i++) {
        cerr << "reading image set with " << class_sizes[i] << " images." << endl;

        for (int32_t j = 0; j < class_sizes[i]; j++) {
            images.push_back(Image(infile, channels, width, height, padding, i, this));
        }
    }
    number_images = images.size();

    infile.close();

    cerr << "image_size: " << channels << "x" << width << "x" << height << " = " << image_size << endl;

    cerr << "read " << images.size() << " images." << endl;
    for (int i = 0; i < (int32_t)class_sizes.size(); i++) {
        cerr << "    class " << setw(4) << i << ": " << class_sizes[i] << endl;
    }

    /*
   for (int i = 0; i < images.size(); i++) {
       images[i].print(cerr);
   }
   */

    return 0;
}



Images::Images(string _filename, int _padding, const vector<float> &_channel_avg, const vector<float> &_channel_std_dev) {
    padding = _padding;

    filename = _filename;
    had_error = read_images(filename);

    channel_avg = _channel_avg;
    channel_std_dev = _channel_std_dev;
}

Images::Images(string _filename, int _padding) {
    padding = _padding;

    filename = _filename;
    had_error = read_images(filename);

    calculate_avg_std_dev();
}

bool Images::loaded_correctly() const {
    return !had_error;
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

int Images::get_image_channels() const {
    return channels;
}

int Images::get_image_height() const {
    return height + (2 * padding);
}

int Images::get_image_width() const {
    return width + (2 * padding);
}


int Images::get_classification(int image) const {
    return images[image].get_classification();
}

float Images::get_pixel(int image, int z, int y, int x) const {
    return images[image].get_pixel(z, y, x);
}


const vector<float>& Images::get_average() const {
    return channel_avg;
}

const vector<float>& Images::get_std_dev() const {
    return channel_std_dev;
}

float Images::get_channel_avg(int channel) const {
    return channel_avg[channel];
}

float Images::get_channel_std_dev(int channel) const {
    return channel_std_dev[channel];
}


void Images::calculate_avg_std_dev() {
    cerr << "calculating averages and standard deviations for images" << endl;
    channel_avg.clear();
    channel_avg.assign(channels, 0.0);

    vector<float> image_avg;
    for (int32_t i = 0; i < number_images; i++) {
        images[i].get_pixel_avg(image_avg);

        for (int32_t j = 0; j < channels; j++) {
            channel_avg[j] += image_avg[j];
        }
    }

    for (int32_t j = 0; j < channels; j++) {
        channel_avg[j] /= number_images;
        cerr << "average pixel value for channel " << j << ": " << channel_avg[j] << endl;
    }

    channel_std_dev.clear();
    channel_std_dev.assign(channels, 0.0);

    vector<float> image_variance;
    for (int i = 0; i < number_images; i++) {
        images[i].get_pixel_variance(channel_avg, image_variance);

        for (int32_t j = 0; j < channels; j++) {
            channel_std_dev[j] += image_variance[j];
        }
    }

    for (int32_t j = 0; j < channels; j++) {
        channel_std_dev[j] /= number_images;
        cerr << "pixel variance for channel " << j << ": " << channel_std_dev[j] << endl;
        channel_std_dev[j] = sqrt(channel_std_dev[j]);
        cerr << "pixel standard deviation for channel " << j << ": " << channel_std_dev[j] << endl;
    }
}
