#ifndef MOSAIC_IMAGE_SET_HXX
#define MOSAIC_IMAGE_SET_HXX

#include <fstream>
using std::ifstream;

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_set_interface.hxx"
#include "large_image_set.hxx"

typedef class MosaicImages MosaicImages;

class Point {
    public:
        int y;
        int x;

        Point(int _y, int _x);
};

class Line {
    public:
        int y1;
        int x1;
        int y2;
        int x2;

        Line(int _y1, int _x1, int _y2, int _x2);
};

class Rectangle {
    public:
        int y1;
        int x1;
        int y2;
        int x2;

        Rectangle(int _y1, int _x1, int _y2, int _x2);
};



class MosaicImages : public MultiImagesInterface {
    private:
        vector<string> filenames;

        int number_classes;
        int number_images;

        vector<int> class_sizes;

        int padding;
        int channels;
        int subimage_width, subimage_height;

        vector<LargeImage> images;

        vector<float> channel_avg;
        vector<float> channel_std_dev;

    public:
        void get_mosaic_pixels(string filename, vector< vector< vector<uint8_t> > > &pixels, uint32_t &height, uint32_t &width);

        void read_mosaic(string filename, const vector<Point> &box_centers, int box_radius, const vector<int> &box_classes);
        void read_mosaic(string filename, const vector<Line> &lines, int line_height, const vector<int> &line_classes);

        void initialize_counts(const vector< vector<int> > &classes);

        MosaicImages(vector<string> _filenames, const vector< vector<Point> > &_box_centers, int _box_radius, const vector< vector<int> > &_box_classes, int _padding, int _subimage_height, int _subimage_width);

        MosaicImages(vector<string> _filenames, const vector< vector<Point> > &_box_centers, int _box_radius, const vector< vector<int> > &_box_classes, int _padding, int _subimage_height, int _subimage_width, const vector<float> &_channel_avg, const vector<float> &_channel_std_dev);

        MosaicImages(vector<string> _filenames, const vector< vector<Line> > &_lines, int _line_height, const vector< vector<int> > &_line_classes, int _padding, int _subimage_height, int _subimage_width);

        MosaicImages(vector<string> _filenames, const vector< vector<Line> > &_lines, int _line_height, const vector< vector<int> > &_line_classes, int _padding, int _subimage_height, int _subimage_width, const vector<float> &_channel_avg, const vector<float> &_channel_std_dev);


        string get_filename() const;

        int get_class_size(int i) const;

        int get_number_classes() const;

        int get_number_images() const;
        int get_number_large_images() const;
        int get_number_subimages(int i) const;

        int get_padding() const;

        int get_image_channels() const;
        int get_image_width() const;
        int get_image_height() const;

        int get_large_image_channels(int image) const;
        int get_large_image_width(int image) const;
        int get_large_image_height(int image) const;


        int get_image_classification(int image) const;
        int get_classification(int subimage) const;
        float get_pixel(int subimage, int z, int y, int x) const;
        float get_raw_pixel(int subimage, int z, int y, int x) const;

        void calculate_avg_std_dev();

        float get_channel_avg(int channel) const;

        float get_channel_std_dev(int channel) const;


        const vector<float>& get_average() const;
        const vector<float>& get_std_dev() const;

        void normalize();

        LargeImage* copy_image(int i) const;

        void set_alpha(int i, const vector< vector<float> > &_alpha);
        void set_alpha(int i, const vector< vector<uint8_t> > &_alpha);
};

#endif

