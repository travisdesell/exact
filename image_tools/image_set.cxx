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

Image::Image(ifstream &infile, int _channels, int _cols, int _rows, int _classification) {
    channels = _channels;
    cols = _cols;
    rows = _rows;
    classification = _classification;

    pixels = vector< vector< vector<double> > >(channels, vector< vector<double> >(cols, vector<double>(rows, 0.0)));

    char* c_pixels = new char[channels * cols * rows];

    infile.read( c_pixels, sizeof(char) * channels * cols * rows);

    int current = 0;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < cols; y++) {
            for (int32_t x = 0; x < rows; x++) {
                pixels[z][y][x] = (int)(uint8_t)c_pixels[current];
                current++;
            }
        }
    }

    delete [] c_pixels;
}

double Image::get_pixel(int z, int y, int x) const {
    return pixels[z][y][x];
}

int Image::get_classification() const {
    return classification;
}

int Image::get_channels() const {
    return channels;
}

int Image::get_rows() const {
    return rows;
}

int Image::get_cols() const {
    return cols;
}

void Image::scale_0_1() {
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < cols; y++) {
            for (int32_t x = 0; x < rows; x++) {
                pixels[z][y][x] /= 255.0;
            }
        }
    }
}

void Image::get_pixel_avg(vector<double> &channel_avgs) const {
    channel_avgs.clear();
    channel_avgs.assign(channels, 0.0);

    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < cols; y++) {
            for (int32_t x = 0; x < rows; x++) {
                channel_avgs[z] += pixels[z][y][x];
            }
        }
        channel_avgs[z] /= (rows * cols);
    }
}

void Image::get_pixel_variance(const vector<double> &channel_avgs, vector<double> &channel_variances) const {
    channel_variances.clear();
    channel_variances.assign(channels, 0.0);

    double tmp;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < cols; y++) {
            for (int32_t x = 0; x < rows; x++) {
                tmp = channel_avgs[z] - pixels[z][y][x];
                channel_variances[z] += tmp * tmp;
            }
        }

        channel_variances[z] /= (rows * cols);
    }
}

void Image::normalize(const vector<double> &channel_avgs, const vector<double> &channel_std_dev) {
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < cols; y++) {
            for (int32_t x = 0; x < rows; x++) {
                pixels[z][y][x] -= channel_avgs[z];
                pixels[z][y][x] /= channel_std_dev[z];
            }
        }
    }
}


void Image::print(ostream &out) {
    out << "Image Class: " << classification << endl;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < cols; y++) {
            for (int32_t x = 0; x < rows; x++) {
                out << setw(7) << pixels[z][y][x];
            }
            out << endl;
        }
    }
}

string Images::get_filename() const {
    return filename;
}

void Images::read_images(string _filename) {
    filename = filename;

    ifstream infile(filename.c_str(), ios::in | ios::binary);

    if (!infile.is_open()) {
        cerr << "Could not open '" << filename << "' for reading." << endl;
        return;
    }

    int initial_vals[4];
    infile.read( (char*)&initial_vals, sizeof(initial_vals) );

    number_classes = initial_vals[0];
    channels = initial_vals[1];
    cols = initial_vals[2];
    rows = initial_vals[3];

    cerr << "number_classes: " << number_classes << endl;
    cerr << "channels: " << channels << endl;
    cerr << "cols: " << cols << endl;
    cerr << "rows: " << rows << endl;

    class_sizes = vector<int>(number_classes, 0);
    infile.read( (char*)&class_sizes[0], sizeof(int) * number_classes );

    int image_size = channels * cols * rows;

    for (int i = 0; i < number_classes; i++) {
        cerr << "reading image set with " << class_sizes[i] << " images." << endl;

        for (int32_t j = 0; j < class_sizes[i]; j++) {
            images.push_back(Image(infile, channels, cols, rows, i));
        }
    }
    number_images = images.size();

    infile.close();

    cerr << "image_size: " << channels << "x" << cols << "x" << rows << " = " << image_size << endl;

    cerr << "read " << images.size() << " images." << endl;
    for (int i = 0; i < (int32_t)class_sizes.size(); i++) {
        cerr << "    class " << setw(4) << i << ": " << class_sizes[i] << endl;
    }

    /*
   for (int i = 0; i < images.size(); i++) {
       images[i].print(cerr);
   }
   */
}



Images::Images(string _filename, const vector<double> &_channel_avg, const vector<double> &_channel_std_dev) {
    filename = _filename;
    read_images(filename);

    cerr << "scaling images." << endl;
    for (int i = 0; i < number_images; i++) {
        images[i].scale_0_1();
    }

    cerr << "normalizing images." << endl;
    channel_avg = _channel_avg;
    channel_std_dev = _channel_std_dev;
    normalize();
    cerr << "normalized." << endl;

}

Images::Images(string _filename) {
    filename = _filename;
    read_images(filename);

    cerr << "scaling images." << endl;
    for (int i = 0; i < number_images; i++) {
        images[i].scale_0_1();
    }

    cerr << "normalizing images." << endl;
    calculate_avg_std_dev();
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

int Images::get_image_channels() const {
    return channels;
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

const vector<double>& Images::get_average() const {
    return channel_avg;
}

const vector<double>& Images::get_std_dev() const {
    return channel_std_dev;
}

void Images::calculate_avg_std_dev() {
    channel_avg.clear();
    channel_avg.assign(channels, 0.0);

    vector<double> image_avg;
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

    vector<double> image_variance;
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

void Images::normalize() {
    for (int32_t j = 0; j < channels; j++) {
        cerr << "Normalizing channel " << j << " with avg: " << channel_avg[j] << ", std_dev: " << channel_std_dev[j] << endl;
    }

    for (int i = 0; i < number_images; i++) {
        images[i].normalize(channel_avg, channel_std_dev);
    }
}
