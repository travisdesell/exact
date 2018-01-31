#ifndef LARGE_IMAGE_SET_HXX
#define LARGE_IMAGE_SET_HXX

#include <fstream>
using std::ifstream;

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_set_interface.hxx"

typedef class LargeImages LargeImages;

class LargeImage : public ImageInterface {
    private:
        friend class LargeImages;

        int number_subimages;
        int padding;
        int channels;
        int height;
        int width;
        int classification;
        vector< vector< vector<uint8_t> > > pixels;

        //reference to images to get channel avgs and std_Devs
        const LargeImages *images;

    public:

        LargeImage(ifstream &infile, int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const LargeImages *_images);
        LargeImage(int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const vector< vector< vector<uint8_t> > > &_pixels);

        int get_classification() const;

        int get_number_subimages() const;
        int get_channels() const;
        int get_height() const;
        int get_width() const;

        void scale_0_1();
        void get_pixel_avg(vector<float> &channel_avgs) const;
        void get_pixel_variance(const vector<float> &channel_avgs, vector<float> &channel_variances) const;
        void normalize(const vector<float> &channel_avgs, const vector<float> &channel_std_dev);

        void print(ostream &out);

        int8_t get_pixel_unnormalized(int z, int y, int x) const;
        void set_pixel(int z, int y, int x, int8_t value);

        LargeImage* copy() const;

#ifdef _HAS_TIFF_
        void draw_image(string filename) const;
#endif
};

class LargeImages : public ImagesInterface {
    private:
        string filename;

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
        int read_images_from_file(string binary_filename);
        int read_images_from_directory(string directory);
        int read_images_from_directories(string directory);

        LargeImages(string binary_filename, int _padding, int _subimage_height, int _subimage_width);
        LargeImages(string binary_filename, int _padding, int _subimage_height, int _subimage_width, const vector<float> &_channeL_avg, const vector<float> &channel_std_dev);

        string get_filename() const;

        int get_class_size(int i) const;

        int get_number_classes() const;

        int get_number_images() const;
        int get_number_large_images() const;
        int get_number_subimages(int i) const;

        int get_image_channels() const;
        int get_image_width() const;
        int get_image_height() const;

        int get_large_image_channels(int image) const;
        int get_large_image_width(int image) const;
        int get_large_image_height(int image) const;


        int get_image_classification(int image) const;
        int get_classification(int subimage) const;
        float get_pixel(int subimage, int z, int y, int x) const;

        void calculate_avg_std_dev();

        float get_channel_avg(int channel) const;

        float get_channel_std_dev(int channel) const;


        const vector<float>& get_average() const;
        const vector<float>& get_std_dev() const;

        void normalize();

        LargeImage* copy_image(int i) const;

#ifdef _HAS_TIFF_
        void draw_image(int i, string filename) const;
#endif
};

#endif

