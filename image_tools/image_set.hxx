#ifndef IMAGE_SET_HXX
#define IMAGE_SET_HXX

#include <fstream>
using std::ifstream;

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_set_interface.hxx"

typedef class Images Images;

class Image : public ImageInterface {
    friend class Images;

    private:
        int padding;
        int channels;
        int height;
        int width;
        int classification;
        vector< vector< vector<uint8_t> > > pixels;

        //reference to images to get channel avgs and std_Devs
        const Images *images;

    public:

        Image(ifstream &infile, int _channels, int _width, int _height, int _padding, int _classification, const Images *_images);

        int get_classification() const;

        void scale_0_1();

        float get_pixel(int z, int y, int x) const;

        void get_pixel_avg(vector<float> &channel_avgs) const;
        void get_pixel_variance(const vector<float> &channel_avgs, vector<float> &channel_variances) const;
        //void normalize(const vector<float> &channel_avgs, const vector<float> &channel_std_dev);

        void print(ostream &out);
};

class Images : public ImagesInterface {
    private:
        string filename;

        int number_classes;
        int number_images;

        vector<int> class_sizes;

        int padding;
        int channels, width, height;

        vector<Image> images;

        vector<float> channel_avg;
        vector<float> channel_std_dev;

        bool had_error;
    public:
        int read_images(string binary_filename);

        Images(string binary_filename, int _padding);
        Images(string binary_filename, int _padding, const vector<float> &_channeL_avg, const vector<float> &channel_std_dev);

        string get_filename() const;

        int get_class_size(int i) const;

        int get_number_classes() const;

        int get_number_images() const;

        int get_image_channels() const;
        int get_image_width() const;
        int get_image_height() const;

        int get_classification(int image) const;
        float get_pixel(int image, int z, int y, int x) const;

        void calculate_avg_std_dev();

        float get_channel_avg(int channel) const;

        float get_channel_std_dev(int channel) const;

        bool loaded_correctly() const;


        const vector<float>& get_average() const;
        const vector<float>& get_std_dev() const;

        //void normalize();
};

#endif

