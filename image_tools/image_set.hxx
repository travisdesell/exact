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

class Image {
    private:
        int rows, cols;
        int classification;
        double **pixels;


    public:

        Image(ifstream &infile, int size, int _rows, int _cols, int _classification);
        double get_pixel(int x, int y) const;

        int get_classification() const;

        int get_rows() const;
        int get_cols() const;

        void scale_0_1();
        double get_pixel_avg() const;
        double get_pixel_variance(double avg) const;
        void normalize(double avg, double variance);

        void print(ostream &out);
};

class Images {
    private:
        int number_classes;
        int number_images;

        vector<int> class_sizes;

        int rows, cols, vals_per_pixel;

        vector<Image> images;

        double avg;
        double std_dev;

    public:
        void read_images(string binary_filename);

        Images(string binary_filename);
        Images(string binary_filename, double avg, double variance);

        int get_class_size(int i) const;

        int get_number_classes() const;

        int get_number_images() const;

        int get_image_rows() const;

        int get_image_cols() const;

        const Image& get_image(int image) const;

        void calculate_avg_std_dev();

        double get_average() const;
        double get_std_dev() const;

        void normalize();
};

#endif

