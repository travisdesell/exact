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
        vector< vector<uint8_t> > alpha;

        //reference to images to get channel avgs and std_Devs
        const LargeImages *images;

    public:

        LargeImage(ifstream &infile, int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const LargeImages *_images);
        LargeImage(int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const vector< vector< vector<uint8_t> > > &_pixels);
        LargeImage(int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const vector< vector< vector<uint8_t> > > &_pixels, const vector< vector<uint8_t> > &_alpha);

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

        uint8_t get_pixel_unnormalized(int z, int y, int x) const;
        uint8_t get_alpha_unnormalized(int y, int x) const;
        void set_pixel(int z, int y, int x, uint8_t value);
        uint8_t get_pixel(int z, int y, int x) const;

        void set_alpha(const vector< vector<uint8_t> > &_alpha);
        void set_alpha(const vector< vector<float> > &_alpha);

        LargeImage* copy() const;

        void draw_png(string filename) const;
        void draw_png_4channel(string filename) const;
        void draw_png_alpha(string filename) const;

#ifdef _HAS_TIFF_
        void draw_tiff(string filename) const;
        void draw_tiff_4channel(string filename) const;
        void draw_tiff_alpha(string filename) const;
#endif
};

class LargeImages : public MultiImagesInterface {
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

        LargeImages(string binary_filename, int _padding, int _subimage_height, int _subimage_width);
        LargeImages(string binary_filename, int _padding, int _subimage_height, int _subimage_width, const vector<float> &_channel_avg, const vector<float> &channel_std_dev);

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

#ifdef _HAS_TIFF_
        int read_images_from_directory(string directory);
        int read_images_from_directories(string directory);
#endif
};

#endif

