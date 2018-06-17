#ifndef IMAGE_SET_INTERFACE_HXX
#define IMAGE_SET_INTERFACE_HXX

#include <fstream>
using std::ifstream;

#include <iostream>
using std::ostream;

#include <string>
using std::string;

#include <vector>
using std::vector;

typedef class Image Image;

class ImageInterface {
    public:

        virtual int get_classification() const = 0;

        virtual void print(ostream &out) = 0;

        virtual ~ImageInterface() = 0;
};

class ImagesInterface {
    public:
        virtual string get_filename() const = 0;

        virtual int get_class_size(int i) const = 0;

        virtual int get_number_classes() const = 0;

        virtual int get_number_images() const = 0;

        virtual int get_image_channels() const = 0;
        virtual int get_image_width() const = 0;
        virtual int get_image_height() const = 0;

        virtual int get_classification(int image) const = 0;
        virtual float get_pixel(int image, int z, int y, int x) const = 0;

        virtual float get_channel_avg(int channel) const = 0;
        virtual float get_channel_std_dev(int channel) const = 0;

        virtual const vector<float>& get_average() const = 0;
        virtual const vector<float>& get_std_dev() const = 0;
};

class MultiImagesInterface : public ImagesInterface {
    public:
        virtual int get_number_large_images() const = 0;
        virtual int get_number_subimages(int i) const = 0;

        virtual int get_padding() const = 0;

        virtual int get_large_image_channels(int image) const = 0;
        virtual int get_large_image_width(int image) const = 0;
        virtual int get_large_image_height(int image) const = 0;

        virtual int get_number_classes() const = 0;

        virtual int get_image_classification(int image) const = 0;

        virtual float get_raw_pixel(int subimage, int z, int y, int x) const = 0;
};

#endif

