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

typedef class Images Images;

class Image {
    private:
        int channels;
        int rows;
        int cols;
        int classification;
        vector< vector< vector<uint8_t> > > pixels;

        //reference to images to get channel avgs and std_Devs
        const Images *images;

    public:

        Image(ifstream &infile, int _channels, int _cols, int _rows, int _classification, const Images *_images);
        double get_pixel(int z, int y, int x) const;

        int get_classification() const;

        int get_channels() const;
        int get_rows() const;
        int get_cols() const;

        void scale_0_1();
        void get_pixel_avg(vector<double> &channel_avgs) const;
        void get_pixel_variance(const vector<double> &channel_avgs, vector<double> &channel_variances) const;
        void normalize(const vector<double> &channel_avgs, const vector<double> &channel_std_dev);

        void print(ostream &out);
};

class Images {
    private:
        string filename;

        int number_classes;
        int number_images;

        vector<int> class_sizes;

        int channels, rows, cols;

        vector<Image> images;

        vector<double> channel_avg;
        vector<double> channel_std_dev;

    public:
        void read_images(string binary_filename);

        Images(string binary_filename);
        Images(string binary_filename, const vector<double> &_channeL_avg, const vector<double> &channel_std_dev);

        string get_filename() const;

        int get_class_size(int i) const;

        int get_number_classes() const;

        int get_number_images() const;

        int get_image_channels() const;
        int get_image_cols() const;
        int get_image_rows() const;

        const Image& get_image(int image) const;

        void calculate_avg_std_dev();

        double get_channel_avg(int channel) const;

        double get_channel_std_dev(int channel) const;


        const vector<double>& get_average() const;
        const vector<double>& get_std_dev() const;

        void normalize();
};

#endif

