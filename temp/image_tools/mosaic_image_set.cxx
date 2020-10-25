#include <cmath>

#include <dirent.h>

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


#include "large_image_set.hxx"
#include "mosaic_image_set.hxx"

#include "stdint.h"

#include <sstream>
using std::ostringstream;

#include "tiff.h"
#include "tiffio.h"

Point::Point(int _y, int _x) : y(_y), x(_x) {
}

Line::Line(int _y1, int _x1, int _y2, int _x2) : y1(_y1), x1(_x1), y2(_y2), x2(_x2) {
}

Rectangle::Rectangle(int _y1, int _x1, int _y2, int _x2) : y1(_y1), x1(_x1), y2(_y2), x2(_x2) {
}


void MosaicImages::get_mosaic_pixels(string filename, vector< vector< vector<uint8_t> > > &pixels, uint32_t &height, uint32_t &width) {

    TIFF *tif = TIFFOpen(filename.c_str(), "r");
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);        // uint32 height;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);           // uint32 width;
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels);

    cout << filename << ", height: " << height << ", width: " << width << ", channels: " << channels << endl;

    uint32_t *raster = (uint32_t*)_TIFFmalloc(height * width * sizeof(uint32_t));
    TIFFReadRGBAImage(tif, width, height, raster, 0);

    pixels.resize(channels, vector< vector<uint8_t> >(height, vector<uint8_t>(width, 0)));

    //cout << "pixels: " << endl;
    int current_y = 0;
    int current_x = 0;
    for (uint32_t i = 0; i < height * width; i++) {
        pixels[0][height - 1 - current_y][current_x] = TIFFGetR(raster[i]);
        if (channels > 1) pixels[1][height - 1 - current_y][current_x] = TIFFGetG(raster[i]);
        if (channels > 2) pixels[2][height - 1 - current_y][current_x] = TIFFGetB(raster[i]);
        if (channels > 3) pixels[3][height - 1 - current_y][current_x] = TIFFGetA(raster[i]);

        current_x++;
        if (current_x == width) {
            current_x = 0;
            current_y++;
        }
    }
    _TIFFfree(raster);
    TIFFClose(tif);
}


void MosaicImages::read_mosaic(string filename, const vector<Point> &box_centers, int box_radius, const vector<int> &box_classes) {

    uint32_t height;
    uint32_t width;
    vector< vector< vector<uint8_t> > > pixels;
    get_mosaic_pixels(filename, pixels, height, width);

    for (uint32_t i = 0; i < box_centers.size(); i++) {
        int32_t start_x = box_centers[i].x - box_radius;
        int32_t start_y = box_centers[i].y - box_radius;
        int32_t end_x = box_centers[i].x + box_radius;
        int32_t end_y = box_centers[i].y + box_radius;

        //need to be careful if the box is near the edge of the mosaic
        if (start_x < 0) start_x = 0;
        if (start_y < 0) start_y = 0;
        if (end_x >= width) end_x = width;
        if (end_y >= height) end_y = height;

        int box_width = end_x - start_x;
        int box_height = end_y - start_y;

        cout << "creating box image from x[" << start_x << " - " << end_x << "], y[" << start_y << " - " << end_y << "] with class: " << box_classes[i] << endl;

        int subimages_along_width = (box_width - subimage_width) + 1;
        int subimages_along_height = (box_height - subimage_height) + 1;
        int number_subimages = subimages_along_width * subimages_along_height;

        vector< vector< vector<uint8_t> > > box_pixels(channels, vector< vector<uint8_t> >(box_height, vector<uint8_t>(box_width, 0)));
        for (uint32_t bz = 0; bz < channels; bz++) {
            for (uint32_t by = 0; by < box_height; by++) {
                for (uint32_t bx = 0; bx < box_width; bx++) {
                    box_pixels[bz][by][bx] = pixels[bz][start_y + by][start_x + bx];
                }
            }
        }

        LargeImage mosaic_image(number_subimages, channels, box_width, box_height, padding, box_classes[i], box_pixels);
        images.push_back(mosaic_image);

        /*
        ostringstream oss;
        oss << "box_y" << box_centers[i].y << "_x" << box_centers[i].x << ".tif";
        mosaic_image.draw_image(oss.str());
        */
    }

}

void MosaicImages::read_mosaic(string filename, const vector<Line> &lines, int line_height, const vector<int> &line_classes) {

    uint32_t height;
    uint32_t width;
    vector< vector< vector<uint8_t> > > pixels;
    get_mosaic_pixels(filename, pixels, height, width);

    for (uint32_t i = 0; i < lines.size(); i++) {
        int32_t y1 = lines[i].y1;
        int32_t x1 = lines[i].x1;
        int32_t y2 = lines[i].y2;
        int32_t x2 = lines[i].x2;

        //calculate the center of the line
        float y_center = (y1 + y2) / 2.0;
        float x_center = (x1 + x2) / 2.0;

        float y_temp = y2 - y_center;
        float x_temp = x2 - x_center;

        //calculate the distance from the center to the endpoints
        int32_t half_length = sqrt( (y_temp * y_temp) + (x_temp * x_temp) ) + 1.0;
        //add the height of the line so we make sure we get all the pixels
        half_length += (line_height / 2.0);

        int32_t line_width = 2.0 * half_length;
        int32_t half_height = line_height / 2.0;

        double rotation_angle = atan((double)(y2 - y1) / (double)(x2 - x1));

        int32_t start_x = x_center - half_length;
        int32_t start_y = y_center - half_length;
        int32_t end_x = x_center + half_length;
        int32_t end_y = y_center + half_length;

        cout << "creating line image within x[" << start_x << " - " << end_x << "], y[" << start_y << " - " << end_y << "] with class: " << line_classes[i] << endl;

        cout << "line_width: " << line_width << ", line_height: " << line_height << ", line_angle: " << (rotation_angle * 180 / M_PI) << endl;

//        vector< vector< vector<uint8_t> > > line_pixels(channels, vector< vector<uint8_t> >(line_height, vector<uint8_t>(line_width, 0)));
        vector< vector< vector<uint8_t> > > line_pixels(channels, vector< vector<uint8_t> >(line_height, vector<uint8_t>(line_width, 0)));

        double cos_angle = cos(rotation_angle);
        double sin_angle = sin(rotation_angle);

        //helpful links for rotation by area mapping:
        //http://www.leptonica.com/rotation.html#ROTATION-BY-AREA-MAPPING
        //https://computergraphics.stackexchange.com/questions/2074/rotate-image-around-its-center
        //http://datagenetics.com/blog/august32013/index.html
        int32_t tly, tlx;
        for (uint32_t bz = 0; bz < channels; bz++) {
            for (int32_t ly = -half_height; ly < half_height; ly++) {
                for (int32_t lx = -half_length; lx < half_length; lx++) {
                    tly = ly + half_height;
                    tlx = lx + half_length;

                    double tmx = (lx * cos_angle - ly * sin_angle) + x_center;
                    double tmy = (lx * sin_angle + ly * cos_angle) + y_center;
                    //cout << "tmx: " << tmx << ", tmy: " << tmy << endl;

                    //interpolation for area mapping
                    //N == 1
                    //[(N - x)(N - y)fi,j + x(N - y)fi,j+1 + y(N - x)fi+1,j + xyfi+1,j+1]

                    if (tmx < 0 || tmy < 0 || (int32_t)tmx + 1 >= width || (int32_t)tmy + 1 >= height) continue;

                    double fy = tmy - (int32_t)tmy;
                    double fx = tmx - (int32_t)tmx;
                    line_pixels[bz][tly][tlx] = ((1 - fx) * (1 - fy) * pixels[bz][tmy][tmx])
                        + (fx * (1 - fy) * pixels[bz][tmy][tmx + 1])
                        + ((1 - fx) * fy * pixels[bz][tmy + 1][tmx])
                        + (fx * fy * pixels[bz][tmy + 1][tmx + 1]);
                }
            }
        }

        cout << "completed rotating pixels!" << endl;

        int subimages_along_width = (line_width - subimage_width) + 1;
        int subimages_along_height = (line_height - subimage_height) + 1;
        int number_subimages = subimages_along_width * subimages_along_height;

        cout << "creating a mosaic image with " << number_subimages << " subimages" << endl;

        LargeImage mosaic_image(number_subimages, channels, line_width, line_height, padding, line_classes[i], line_pixels);
        images.push_back(mosaic_image);

        /*
        ostringstream oss;
        oss << "line_y1_" << lines[i].y1 << "_x1_" << lines[i].x1 << "__y2_" << lines[i].y2 << "_x2_" << lines[i].x2 << ".tif";
        mosaic_image.draw_image(oss.str());
        */
    }

}


void MosaicImages::initialize_counts(const vector< vector<int> > &classes) {
    //find the max class value and add one to it for the number of classes
    number_classes = 0;
    for (uint32_t i = 0; i < classes.size(); i++) {
        for (uint32_t j = 0; j < classes[i].size(); j++) {
            if (classes[i][j] > number_classes) number_classes = classes[i][j];
        }
    }
    number_classes++;

    class_sizes.assign(number_classes, 0);

    number_images = 0;
    for (uint32_t i = 0; i < images.size(); i++) {
        int32_t number_subimages = images[i].get_number_subimages();
        number_images += number_subimages;
        class_sizes[images[i].get_classification()] += number_subimages;
    }
}

MosaicImages::MosaicImages(vector<string> _filenames, const vector< vector<Point> > &_box_centers, int _box_radius, const vector< vector<int> > &_box_classes, int _padding, int _subimage_height, int _subimage_width) {

    padding = _padding;
    subimage_height = _subimage_height;
    subimage_width = _subimage_width;

    filenames = _filenames;

    for (uint32_t i = 0; i < filenames.size(); i++) {
        cout << "reading images from file: " << filenames[i] << endl;
        read_mosaic(filenames[i], _box_centers[i], _box_radius, _box_classes[i]);
    }

    initialize_counts(_box_classes);

    calculate_avg_std_dev();
}

MosaicImages::MosaicImages(vector<string> _filenames, const vector< vector<Point> > &_box_centers, int _box_radius, const vector< vector<int> > &_box_classes, int _padding, int _subimage_height, int _subimage_width, const vector<float> &_channel_avg, const vector<float> &_channel_std_dev) {

    padding = _padding;
    subimage_height = _subimage_height;
    subimage_width = _subimage_width;

    filenames = _filenames;

    for (uint32_t i = 0; i < filenames.size(); i++) {
        cout << "reading images from file: " << filenames[i] << endl;
        read_mosaic(filenames[i], _box_centers[i], _box_radius, _box_classes[i]);
    }
    initialize_counts(_box_classes);

    channel_avg = _channel_avg;
    channel_std_dev = _channel_std_dev;

    for (int32_t j = 0; j < channels; j++) {
        cerr << "setting pixel variance for channel " << j << ": " << channel_std_dev[j] << endl;
        cerr << "setting pixel standard deviation for channel " << j << ": " << channel_std_dev[j] << endl;
    }
}

MosaicImages::MosaicImages(vector<string> _filenames, const vector< vector<Line> > &_lines, int _line_height, const vector< vector<int> > &_line_classes, int _padding, int _subimage_height, int _subimage_width) {

    padding = _padding;
    subimage_height = _subimage_height;
    subimage_width = _subimage_width;

    filenames = _filenames;

    for (uint32_t i = 0; i < filenames.size(); i++) {
        cout << "reading images from file: " << filenames[i] << endl;
        read_mosaic(filenames[i], _lines[i], _line_height, _line_classes[i]);
    }
    initialize_counts(_line_classes);

    calculate_avg_std_dev();
}

MosaicImages::MosaicImages(vector<string> _filenames, const vector< vector<Line> > &_lines, int _line_height, const vector< vector<int> > &_line_classes, int _padding, int _subimage_height, int _subimage_width, const vector<float> &_channel_avg, const vector<float> &_channel_std_dev) {

    padding = _padding;
    subimage_height = _subimage_height;
    subimage_width = _subimage_width;

    filenames = _filenames;

    //these will be initialized in read_mosaic
    number_images = 0;
    number_classes = 0;

    for (uint32_t i = 0; i < filenames.size(); i++) {
        cout << "reading images from file: " << filenames[i] << endl;
        read_mosaic(filenames[i], _lines[i], _line_height, _line_classes[i]);
    }
    initialize_counts(_line_classes);

    channel_avg = _channel_avg;
    channel_std_dev = _channel_std_dev;

    for (int32_t j = 0; j < channels; j++) {
        cerr << "setting pixel variance for channel " << j << ": " << channel_std_dev[j] << endl;
        cerr << "setting pixel standard deviation for channel " << j << ": " << channel_std_dev[j] << endl;
    }
}


int MosaicImages::get_class_size(int i) const {
    return class_sizes[i];
}

int MosaicImages::get_number_classes() const {
    return number_classes;
}

int MosaicImages::get_number_images() const {
    return number_images;
}

int MosaicImages::get_number_large_images() const {
    return images.size();
}

int MosaicImages::get_number_subimages(int i) const {
    return images[i].get_number_subimages();
}


int MosaicImages::get_padding() const {
    return padding;
}

int MosaicImages::get_image_channels() const {
    return channels;
}

int MosaicImages::get_image_width() const {
    return subimage_width + (2 * padding);
}

int MosaicImages::get_image_height() const {
    return subimage_height + (2 * padding);
}

int MosaicImages::get_large_image_height(int image) const {
    return images[image].get_height();
}

int MosaicImages::get_large_image_width(int image) const {
    return images[image].get_width();
}

int MosaicImages::get_large_image_channels(int image) const {
    return images[image].get_channels();
}


int MosaicImages::get_image_classification(int image) const {
    return images[image].get_classification();
}

int MosaicImages::get_classification(int subimage) const {
    for (int32_t i = 0; i < images.size(); i++) {
        if (subimage < images[i].get_number_subimages()) {
            return images[i].get_classification();
        } else {
            subimage -= images[i].get_number_subimages();

            if (i == images.size() - 1) return images[i].get_classification();
        }
    }

    cerr << "Error getting classification, subimage was: " << subimage << " and there are not that many subimages!" << endl;
    exit(1);
    return 0;
}

float MosaicImages::get_pixel(int subimage, int z, int y, int x) const {
    //cout << "getting pixel from subimage: " << subimage << ", z: " << z << ", y: " << y << ", x: " << x << endl;

    int32_t i;
    for (i = 0; i < images.size(); i++) {
        if (subimage < images[i].get_number_subimages()) {
            const LargeImage &image = images[i];

            if (y < padding || x < padding) return 0;
            else if (y >= subimage_height + padding || x >= subimage_width + padding) return 0;
            else {
                int subimages_along_width = image.get_width() - subimage_width + 1;

                int subimage_y_offset = subimage / subimages_along_width;
                int subimage_x_offset = subimage % subimages_along_width;

                return ((image.get_pixel(z, subimage_y_offset + y, subimage_x_offset + x) / 255.0) - channel_avg[z]) / channel_std_dev[z];
            }
        } else {
            subimage -= images[i].get_number_subimages();
        }
    }

    cerr << "Error getting normalized pixel, subimage was: " << subimage << " and there are not that many subimages!" << endl;
    cerr << "images[" << i << "] had " << images[i].get_number_subimages() << " subimages" << endl;
    cerr << "images[" << i << "].get_height(): " << images[i].get_height() << endl;
    cerr << "images[" << i << "].get_width(): " << images[i].get_width() << endl;
    exit(1);
    return 0;
}


const vector<float>& MosaicImages::get_average() const {
    return channel_avg;
}

const vector<float>& MosaicImages::get_std_dev() const {
    return channel_std_dev;
}

float MosaicImages::get_channel_avg(int channel) const {
    return channel_avg[channel];
}

float MosaicImages::get_channel_std_dev(int channel) const {
    return channel_std_dev[channel];
}

void MosaicImages::calculate_avg_std_dev() {
    //cerr << "calculating averages and standard deviations for images" << endl;
    channel_avg.clear();
    channel_avg.assign(channels, 0.0);

    //cerr << "number images: " << number_images << endl;

    vector<float> image_avg;
    for (int32_t i = 0; i < images.size(); i++) {
        //cout << "getting pixel avg for image: " << i << endl;
        images[i].get_pixel_avg(image_avg);

        for (int32_t j = 0; j < channels; j++) {
            channel_avg[j] += image_avg[j];
            //cout << "images[" << i << "], channel_avg[" << j << "]: " << channel_avg[j] << endl;
        }

        //cerr << "image width: " << images[i].get_width() << endl;
    }

    //TODO: calculate average and standard deviation for varying sized images
    //may want to average this over total pixels instead of per image

    for (int32_t j = 0; j < channels; j++) {
        channel_avg[j] /= images.size();
        cerr << "average pixel value for channel " << j << ": " << channel_avg[j] << endl;
    }

    channel_std_dev.clear();
    channel_std_dev.assign(channels, 0.0);

    vector<float> image_variance;
    for (int i = 0; i < images.size(); i++) {
        //cout << "getting pixel variance for image: " << i << endl;
        images[i].get_pixel_variance(channel_avg, image_variance);

        for (int32_t j = 0; j < channels; j++) {
            channel_std_dev[j] += image_variance[j];
            //cout << "images[" << i << "], channel_std_dev[" << j << "]: " << channel_std_dev[j] << endl;
        }
    }

    for (int32_t j = 0; j < channels; j++) {
        channel_std_dev[j] /= images.size();
        cerr << "pixel variance for channel " << j << ": " << channel_std_dev[j] << endl;
        channel_std_dev[j] = sqrt(channel_std_dev[j]);
        cerr << "pixel standard deviation for channel " << j << ": " << channel_std_dev[j] << endl;
    }
}

 LargeImage* MosaicImages::copy_image(int i) const {
    return images[i].copy();
}

float MosaicImages::get_raw_pixel(int subimage, int z, int y, int x) const {
    return images[subimage].get_pixel(z, y + padding, x + padding);
}

string MosaicImages::get_filename() const {
    string merged_filename;

    for (uint32_t i = 0; i < filenames.size(); i++) {
        if (i != 0) merged_filename += " ";
        merged_filename += filenames[i];
    }

    return merged_filename;
}

void MosaicImages::set_alpha(int i, const vector< vector<float> > &_alpha) {
    images[i].set_alpha(_alpha);
}

void MosaicImages::set_alpha(int i, const vector< vector<uint8_t> > &_alpha) {
    images[i].set_alpha(_alpha);
}



#ifdef MOSAIC_IMAGES_TEST
int main(int argc, char **argv) {
    int subimage_y = 32;
    int subimage_x = 32;
    int padding = 3;

    int box_radius = 32;

    vector<string> filenames;
    for (uint32_t i = 1; i < argc; i++) {
        filenames.push_back(argv[i]);

    }

    vector< vector<Point> > box_centers;
    vector<Point> mosaic_box_centers;

    mosaic_box_centers.push_back(Point(100,100));
    mosaic_box_centers.push_back(Point(1709,369));
    mosaic_box_centers.push_back(Point(2388,1478));
    mosaic_box_centers.push_back(Point(24990,5));
    box_centers.push_back(mosaic_box_centers);

    vector< vector<int> > box_classes;
    vector<int> mosaic_box_classes;
    mosaic_box_classes.push_back(0);
    mosaic_box_classes.push_back(0);
    mosaic_box_classes.push_back(0);
    box_classes.push_back(mosaic_box_classes);

    MosaicImages box_mosaic_images(filenames, box_centers, box_radius, box_classes, padding, subimage_y, subimage_x);

    vector< vector< Line > > lines;
    vector< Line > mosaic_lines;

    mosaic_lines.push_back(Line(12710,5579,22481,4344));
    mosaic_lines.push_back(Line(4300,7300,4500,7500));
    mosaic_lines.push_back(Line(5360,13794,5360,14092));
    mosaic_lines.push_back(Line(1963,16277,2081,16522));
    mosaic_lines.push_back(Line(500,500,600,600));
    mosaic_lines.push_back(Line(6117,14477,7238,15115));
    lines.push_back(mosaic_lines);

    int line_height = 64;

    vector< vector<int> > line_classes;
    vector<int> mosaic_line_classes;
    mosaic_line_classes.push_back(0);
    mosaic_line_classes.push_back(1);
    mosaic_line_classes.push_back(2);
    mosaic_line_classes.push_back(2);
    mosaic_line_classes.push_back(2);
    mosaic_line_classes.push_back(3);
    line_classes.push_back(mosaic_line_classes);

    MosaicImages line_mosaic_images(filenames, lines, line_height, line_classes, padding, subimage_y, subimage_x);

    cout << "number classes: " << line_mosaic_images.get_number_classes() << endl;
    cout << "number images: " << line_mosaic_images.get_number_images() << endl;

}
#endif
