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

#include "stdint.h"

#include "lodepng.h"

#ifdef _HAS_TIFF_
#include <sstream>
using std::ostringstream;

#include "tiff.h"
#include "tiffio.h"
#endif

#include "lodepng.h"

ImageInterface::~ImageInterface() {
}

LargeImage::LargeImage(ifstream &infile, int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const LargeImages *_images) {
    number_subimages = _number_subimages;
    channels = _channels;
    width = _width;
    height = _height;
    padding = _padding;
    classification = _classification;
    images = _images;

    //cout << "channels: " << channels << ", height: " << height << ", width: " << width << endl;

    pixels = vector< vector< vector<uint8_t> > >(channels, vector< vector<uint8_t> >(height, vector<uint8_t>(width, 0)));

    char* c_pixels = new char[channels * width * height];

    infile.read( c_pixels, sizeof(char) * channels * width * height);

    int current = 0;
    for (int z = 0; z < channels; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pixels[z][y][x] = (uint8_t)c_pixels[current];
                current++;
                //cout << "pixels[" << z << "][" << y << "][" << x << "]: " << (int)pixels[z][y][x] << endl;
            }
        }
    }
    delete [] c_pixels;
}

LargeImage::LargeImage(int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const vector< vector< vector<uint8_t> > > &_pixels) {
    number_subimages = _number_subimages;
    channels = _channels;
    width = _width;
    height = _height;
    padding = _padding;
    classification = _classification;
    pixels = _pixels;
}

LargeImage::LargeImage(int _number_subimages, int _channels, int _width, int _height, int _padding, int _classification, const vector< vector< vector<uint8_t> > > &_pixels, const vector< vector<uint8_t> > &_alpha) {
    number_subimages = _number_subimages;
    channels = _channels;
    width = _width;
    height = _height;
    padding = _padding;
    classification = _classification;
    pixels = _pixels;
    alpha = _alpha;
}



LargeImage* LargeImage::copy() const {
    return new LargeImage(number_subimages, channels, width, height, padding, classification, pixels);
}

uint8_t LargeImage::get_pixel_unnormalized(int z, int y, int x) const {
    if (y < padding || x < padding) return 0;
    else if (y >= height + padding || x >= width + padding) return 0;
    else {
        return pixels[z][y - padding][x - padding];
    }
}

uint8_t LargeImage::get_alpha_unnormalized(int y, int x) const {
    if (y < padding || x < padding) return 0;
    else if (y >= height + padding || x >= width + padding) return 0;
    else {
        return alpha[y - padding][x - padding];
    }
}


void LargeImage::set_pixel(int z, int y, int x, uint8_t value) {
    if (y < padding || x < padding) return;
    else if (y >= height + padding || x >= width + padding) return;
    else {
        pixels[z][y - padding][x - padding] = value;
    }
}

uint8_t LargeImage::get_pixel(int z, int y, int x) const {
    if (y < padding || x < padding) return 0;
    else if (y >= height + padding || x >= width + padding) return 0;
    else {
        return pixels[z][y - padding][x - padding];
    }
}

void LargeImage::draw_png(string filename) const {

    cout << "drawing a PNG with height: " << height << " and width: " << width << " and padding: " << padding << endl;

    vector<uint8_t> flat_pixels;

    for (int32_t y = 0; y < height + (padding * 2);  y++) {
        for (int32_t x = 0; x < width + (padding * 2); x++) {
            flat_pixels.push_back( get_pixel_unnormalized(0, y, x) );
            flat_pixels.push_back( get_pixel_unnormalized(1, y, x) );
            flat_pixels.push_back( get_pixel_unnormalized(2, y, x) );
            flat_pixels.push_back( 255 );
        }
    }

    //Encode the image
    uint32_t error = lodepng::encode(filename.c_str(), flat_pixels, width + (padding * 2), height + (padding * 2));

    //if there's an error, display it
    if (error) cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;
}

void LargeImage::draw_png_4channel(string filename) const {

    cout << "drawing a PNG with height: " << height << " and width: " << width << " and padding: " << padding << endl;

    vector<uint8_t> flat_pixels;

    for (int32_t y = 0; y < height + (padding * 2);  y++) {
        for (int32_t x = 0; x < width + (padding * 2); x++) {
            flat_pixels.push_back( get_pixel_unnormalized(0, y, x) );
            flat_pixels.push_back( get_pixel_unnormalized(1, y, x) );
            flat_pixels.push_back( get_pixel_unnormalized(2, y, x) );
            flat_pixels.push_back( get_alpha_unnormalized(y, x) );
        }
    }

    //Encode the image
    uint32_t error = lodepng::encode(filename.c_str(), flat_pixels, width + (padding * 2), height + (padding * 2));

    //if there's an error, display it
    if (error) cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;
}

void LargeImage::draw_png_alpha(string filename) const {

    cout << "drawing a PNG with height: " << height << " and width: " << width << " and padding: " << padding << endl;

    vector<uint8_t> flat_pixels;

    for (int32_t y = 0; y < height + (padding * 2);  y++) {
        for (int32_t x = 0; x < width + (padding * 2); x++) {
            flat_pixels.push_back( get_alpha_unnormalized(y, x) );
            flat_pixels.push_back( get_alpha_unnormalized(y, x) );
            flat_pixels.push_back( get_alpha_unnormalized(y, x) );
            flat_pixels.push_back( 255 );
        }
    }

    //Encode the image
    uint32_t error = lodepng::encode(filename.c_str(), flat_pixels, width + (padding * 2), height + (padding * 2));

    //if there's an error, display it
    if (error) cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;
}



#ifdef _HAS_TIFF_

void LargeImage::draw_tiff(string filename) const {
    // Open the TIFF file
    TIFF *output_image = NULL;

    cout << "drawing a TIFF with height: " << height << " and width: " << width << " and padding: " << padding << endl;

    uint8_t *values = new uint8_t[(height + (padding * 2)) * (width + (padding * 2)) * 3];
    int current = 0;

    for (int32_t y = 0; y < height + (padding * 2);  y++) {
        for (int32_t x = 0; x < width + (padding * 2); x++) {
            //cout << "pushing back values for [" << y << "][" << x << "]!" << endl;
            values[current] = get_pixel_unnormalized(0, y, x);
            values[current + 1] = get_pixel_unnormalized(1, y, x);
            values[current + 2] = get_pixel_unnormalized(2, y, x);
            current += 3;
            //cout << "pushed back values for [" << y << "][" << x << "]!" << endl;
        }
    }

    if ((output_image = TIFFOpen(filename.c_str(), "w")) == NULL) {
        std::cerr << "Unable to write tif file: " << "image.tiff" << std::endl;
    }

    TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 3);
    TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, height + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);
    TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

    // Write the information to the file

    tsize_t image_s;
    if ( (image_s = TIFFWriteEncodedStrip(output_image, 0, values, 3 * (width + (padding * 2)) * (height + (padding * 2)) * sizeof(int8_t))) == -1) {
        std::cerr << "Unable to write tif file '" << filename << "'" << endl;
    }
    else {
        std::cout << "Image is saved to '" << filename << "', size is : " << image_s << endl;
    }

    TIFFWriteDirectory(output_image);
    TIFFClose(output_image);
}

void LargeImage::draw_tiff_alpha(string filename) const {
    // Open the TIFF file
    TIFF *output_image = NULL;

    cout << "drawing a TIFF with height: " << height << " and width: " << width << " and padding: " << padding << endl;

    uint8_t *values = new uint8_t[(height + (padding * 2)) * (width + (padding * 2)) * 1];
    int current = 0;

    for (int32_t y = 0; y < height + (padding * 2);  y++) {
        for (int32_t x = 0; x < width + (padding * 2); x++) {
            //cout << "pushing back values for [" << y << "][" << x << "]!" << endl;
            values[current] = get_alpha_unnormalized(y, x);
            current += 1;
            //cout << "pushed back values for [" << y << "][" << x << "]!" << endl;
        }
    }

    if ((output_image = TIFFOpen(filename.c_str(), "w")) == NULL) {
        std::cerr << "Unable to write tif file: " << "image.tiff" << std::endl;
    }

    TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, height + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);
    TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

    // Write the information to the file

    tsize_t image_s;
    if ( (image_s = TIFFWriteEncodedStrip(output_image, 0, values, (width + (padding * 2)) * (height + (padding * 2)) * sizeof(int8_t))) == -1) {
        std::cerr << "Unable to write tif file '" << filename << "'" << endl;
    }
    else {
        std::cout << "Image is saved to '" << filename << "', size is : " << image_s << endl;
    }

    TIFFWriteDirectory(output_image);
    TIFFClose(output_image);
}


void LargeImage::draw_tiff_4channel(string filename) const {
    // Open the TIFF file
    TIFF *output_image = NULL;

    cout << "drawing a TIFF with height: " << height << " and width: " << width << " and padding: " << padding << endl;

    uint8_t *values = new uint8_t[(height + (padding * 2)) * (width + (padding * 2)) * 4];
    int current = 0;

    for (int32_t y = 0; y < height + (padding * 2);  y++) {
        for (int32_t x = 0; x < width + (padding * 2); x++) {
            //cout << "pushing back values for [" << y << "][" << x << "]!" << endl;
            values[current] = get_pixel_unnormalized(0, y, x);
            values[current + 1] = get_pixel_unnormalized(1, y, x);
            values[current + 2] = get_pixel_unnormalized(2, y, x);
            values[current + 3] = get_alpha_unnormalized(y, x);
            current += 4;
            //cout << "pushed back values for [" << y << "][" << x << "]!" << endl;
        }
    }

    if ((output_image = TIFFOpen(filename.c_str(), "w")) == NULL) {
        std::cerr << "Unable to write tif file: " << "image.tiff" << std::endl;
    }

    int samples_per_pixel = 4;
    TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height + (padding * 2));
    TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, samples_per_pixel);
    //TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, height + (padding * 2));
    //TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(output_image, width * samples_per_pixel));
    TIFFSetField(output_image, TIFFTAG_ORIENTATION, (int)ORIENTATION_TOPLEFT);
    TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    //TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    //TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    // Write the information to the file

    tsize_t image_s;
    if ( (image_s = TIFFWriteEncodedStrip(output_image, 0, values, samples_per_pixel * (width + (padding * 2)) * (height + (padding * 2)) )) == -1) {
        std::cerr << "Unable to write tif file '" << filename << "'" << endl;
    }
    else {
        std::cout << "Image is saved to '" << filename << "', size is : " << image_s << endl;
    }

    TIFFWriteDirectory(output_image);
    TIFFClose(output_image);
}
#endif

int LargeImage::get_classification() const {
    return classification;
}

int LargeImage::get_number_subimages() const {
    return number_subimages;
}

int LargeImage::get_channels() const {
    return channels;
}

int LargeImage::get_height() const {
    return height;
}

int LargeImage::get_width() const {
    return width;
}

void LargeImage::get_pixel_avg(vector<float> &channel_avgs) const {
    channel_avgs.clear();
    channel_avgs.assign(channels, 0.0);

    //cout << "channels: " << channels << ", height: " << height << ", width: " << width << endl;

    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                channel_avgs[z] += pixels[z][y][x] / 255.0;
            }
        }
        channel_avgs[z] /= (height * width);
    }
}

void LargeImage::get_pixel_variance(const vector<float> &channel_avgs, vector<float> &channel_variances) const {
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

void LargeImage::set_alpha(const vector< vector<uint8_t> > &_alpha) {
    alpha = _alpha;
}

void LargeImage::set_alpha(const vector< vector<float> > &_alpha) {
    if (_alpha.size() != height) {
        cerr << "ERROR: could not set alpha of image because it was incorrectly sized. alpha.size(): " << alpha.size() << " != height: " << height << endl;
        exit(1);
    }
    alpha.assign(height, vector<uint8_t>(width, 0));

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            if (_alpha[y].size() != width) {
                cerr << "ERROR: could not set alpha of image because it was incorrectly sized. alpha[" << y << "].size(): " << alpha[y].size() << " != width: " << width<< endl;
                exit(1);
            }

            alpha[y][x] = _alpha[y][x] * 255.0;
        }
    }
}


void LargeImage::print(ostream &out) {
    out << "LargeImage Class: " << classification << endl;
    for (int32_t z = 0; z < channels; z++) {
        for (int32_t y = 0; y < height; y++) {
            for (int32_t x = 0; x < width; x++) {
                out << setw(7) << pixels[z][y][x];
            }
            out << endl;
        }
    }
}

string LargeImages::get_filename() const {
    return filename;
}

int LargeImages::read_images_from_file(string _filename) {
    filename = _filename;

    cout << "reading filename: " << filename << endl;

    ifstream infile(filename.c_str(), ios::in | ios::binary);

    if (!infile.is_open()) {
        cerr << "Could not open '" << filename << "' for reading." << endl;
        return 1;
    }

    int initial_vals[2];
    infile.read( (char*)&initial_vals, sizeof(initial_vals) );

    number_classes = initial_vals[0];
    number_images = initial_vals[1];

    cerr << "number_classes: " << number_classes << endl;
    cerr << "number_images: " << number_images << endl;


    class_sizes = vector<int>(number_classes, 0);

    for (int i = 0; i < number_images; i++) {
        int image_vals[4];
        infile.read( (char*)&image_vals, sizeof(image_vals) );

        int image_class = image_vals[0];
        channels = image_vals[1];
        int height = image_vals[2];
        int width = image_vals[3];

        cerr << "image[" << i << "] class: " << image_class << ", channels: " << channels << ", height: " << height << ", width: " << width << endl;

        class_sizes[image_class]++;

        int subimages_along_width = (width - subimage_width) + 1;
        int subimages_along_height = (height - subimage_height) + 1;
        int number_subimages = subimages_along_width * subimages_along_height;

        LargeImage large_image(infile, number_subimages, channels, width, height, padding, image_class, this);

        if (number_subimages < 0) {
            cerr << "ERROR! number subimages < 0!" << endl;
            continue;
        }

        images.push_back(large_image);
    }


    infile.close();

    cerr << "read " << images.size() << " images." << endl;
    for (int i = 0; i < (int32_t)class_sizes.size(); i++) {
        cerr << "    class " << setw(4) << i << ": " << class_sizes[i] << endl;
    }

    //update number images to number of subimages
    number_images = 0;
    for (int i = 0; i < images.size(); i++) {
        number_images += images[i].get_number_subimages();
    }

    cerr << "number_subimages: " << number_images << endl;

    /*
   for (int i = 0; i < images.size(); i++) {
       images[i].print(cerr);
   }
   */

    return 0;
}

#ifdef _HAS_TIFF_
int LargeImages::read_images_from_directory(string directory) {
    DIR *dir;
    struct dirent *ent;

    int image_class = 0;
    if ((dir = opendir(directory.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            cout << "file: '" << ent->d_name << "'" << endl;

            string image_filename(directory + ent->d_name);
            if (image_filename.size() < 4 || image_filename.substr(image_filename.size() - 4, 4).compare(".tif") != 0) {
                cout << "skipping non-TIFF file: '" << image_filename << "'" << endl;
                continue;
            }

            TIFF *tif = TIFFOpen(image_filename.c_str(), "r");
            uint32_t height, width;
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);        // uint32 height;
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);           // uint32 width;
            TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels);

            cout << image_filename << ", height: " << height << ", width: " << width << ", channels: " << channels << endl;

            uint32_t *raster = (uint32_t*)_TIFFmalloc(height * width * sizeof(uint32_t));
            TIFFReadRGBAImage(tif, width, height, raster, 0);

            vector< vector< vector<uint8_t> > > pixels(channels, vector< vector<uint8_t> >(height, vector<uint8_t>(width, 0)));

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

            int subimages_along_width = (width - subimage_width) + 1;
            int subimages_along_height = (height - subimage_height) + 1;
            int number_subimages = subimages_along_width * subimages_along_height;

            LargeImage large_image(number_subimages, channels, width, height, padding, image_class, pixels);
            images.push_back(large_image);

            ///string test_filename(image_filename.substr(0, image_filename.length() - 4) + "_test.tif");
            //cout << "writing test filename '" << test_filename << "'" << endl;
            //large_image.draw_tiff(test_filename);
        }

        //_TIFFfree(raster);

        closedir(dir);
    } else {
        /* could not open directory */
        cerr << "Attemped to read images from directory '" << directory << "', but could not open directory for reading." << endl;
        return EXIT_FAILURE;
    }

    return 0;
}

int LargeImages::read_images_from_directories(string directory) {
    DIR *dir;
    struct dirent *ent;

    int image_class = 0;
    if ((dir = opendir(directory.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            cout << "file: '" << ent->d_name << "'" << endl;

            string filename(ent->d_name);
            if (filename.compare(".") == 0) {
                cout << "skipping current directory" << endl;
                continue;
            }

            if (filename.compare("..") == 0) {
                cout << "skipping parent directory" << endl;
                continue;
            }

            string subdir_filename(directory + ent->d_name);
            DIR *subdir = opendir(subdir_filename.c_str());
            struct dirent *subdir_ent;

            if (subdir == NULL) {
                cout << "was not a directory, skipping!" << endl;
                continue;
            }

            cout << "entering subdir: '" << subdir_filename << "' for class: " << image_class << endl;
            while ((subdir_ent = readdir(subdir)) != NULL) {
                string image_filename(subdir_filename + "/" + subdir_ent->d_name);
                if (image_filename.size() < 4 || image_filename.substr(image_filename.size() - 4, 4).compare(".tif") != 0) {
                    cout << "skipping non-TIFF file: '" << image_filename << "'" << endl;
                    continue;
                }

                TIFF *tif = TIFFOpen(image_filename.c_str(), "r");
                uint32_t height, width;
                TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);        // uint32 height;
                TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);           // uint32 width;
                TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels);

                cout << image_filename << ", height: " << height << ", width: " << width << ", channels: " << channels << endl;

                uint32_t *raster = (uint32_t*)_TIFFmalloc(height * width * sizeof(uint32_t));
                TIFFReadRGBAImage(tif, width, height, raster, 0);

                vector< vector< vector<uint8_t> > > pixels(channels, vector< vector<uint8_t> >(height, vector<uint8_t>(width, 0)));

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

                int subimages_along_width = (width - subimage_width) + 1;
                int subimages_along_height = (height - subimage_height) + 1;
                int number_subimages = subimages_along_width * subimages_along_height;

                LargeImage large_image(number_subimages, channels, width, height, padding, image_class, pixels);
                images.push_back(large_image);

                ///string test_filename(image_filename.substr(0, image_filename.length() - 4) + "_test.tif");
                //cout << "writing test filename '" << test_filename << "'" << endl;
                //large_image.draw_tiff(test_filename);
            }
            image_class++;

            //_TIFFfree(raster);

        }
        closedir(dir);
    } else {
        /* could not open directory */
        cerr << "Attemped to read images from directory '" << directory << "', but could not open directory for reading." << endl;
        return EXIT_FAILURE;
    }

    return 0;
}
#endif


LargeImages::LargeImages(string _filename, int _padding, int _subimage_height, int _subimage_width, const vector<float> &_channel_avg, const vector<float> &_channel_std_dev) {
    padding = _padding;
    subimage_height = _subimage_height;
    subimage_width = _subimage_width;

    filename = _filename;
    cout << "filename substr: " << filename.substr(filename.size() - 4, 4) << endl;
    if (filename.size() >= 4 && filename.substr(filename.size() - 4, 4).compare(".bin") == 0) {
        cout << "reading images from file: " << endl;
        read_images_from_file(filename);
#ifdef _HAS_TIFF_
    } else {
        cout << "reading images from directory: " << endl;
        read_images_from_directory(filename);
#endif
    }

    channel_avg = _channel_avg;
    channel_std_dev = _channel_std_dev;

    for (int32_t j = 0; j < channels; j++) {
        cerr << "setting pixel variance for channel " << j << ": " << channel_std_dev[j] << endl;
        cerr << "setting pixel standard deviation for channel " << j << ": " << channel_std_dev[j] << endl;
    }
}

LargeImages::LargeImages(string _filename, int _padding, int _subimage_height, int _subimage_width) {
    padding = _padding;
    subimage_height = _subimage_height;
    subimage_width = _subimage_width;

    filename = _filename;
    cout << "filename substr: " << filename.substr(filename.size() - 4, 4) << endl;
    if (filename.size() >= 4 && filename.substr(filename.size() - 4, 4).compare(".bin") == 0) {
        cout << "reading images from binary file: " << endl;
        read_images_from_file(filename);
#ifdef _HAS_TIFF_
    } else {
        cout << "reading images from directory: " << endl;
        read_images_from_directory(filename);
#endif
    }

    calculate_avg_std_dev();
}

int LargeImages::get_class_size(int i) const {
    return class_sizes[i];
}

int LargeImages::get_number_classes() const {
    return number_classes;
}

int LargeImages::get_number_images() const {
    return number_images;
}

int LargeImages::get_number_large_images() const {
    return images.size();
}

int LargeImages::get_number_subimages(int i) const {
    return images[i].get_number_subimages();
}


int LargeImages::get_padding() const {
    return padding;
}

int LargeImages::get_image_channels() const {
    return channels;
}

int LargeImages::get_image_width() const {
    return subimage_width + (2 * padding);
}

int LargeImages::get_image_height() const {
    return subimage_height + (2 * padding);
}

int LargeImages::get_large_image_height(int image) const {
    return images[image].get_height();
}

int LargeImages::get_large_image_width(int image) const {
    return images[image].get_width();
}

int LargeImages::get_large_image_channels(int image) const {
    return images[image].get_channels();
}


int LargeImages::get_image_classification(int image) const {
    return images[image].get_classification();
}

int LargeImages::get_classification(int subimage) const {
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

float LargeImages::get_pixel(int subimage, int z, int y, int x) const {
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
    cerr << "images[" << i << "] had " << images[subimage].get_number_subimages() << " subimages" << endl;
    cerr << "images[" << i << "].get_height(): " << images[subimage].get_height() << endl;
    cerr << "images[" << i << "].get_width(): " << images[subimage].get_width() << endl;
    exit(1);
    return 0;
}


const vector<float>& LargeImages::get_average() const {
    return channel_avg;
}

const vector<float>& LargeImages::get_std_dev() const {
    return channel_std_dev;
}

float LargeImages::get_channel_avg(int channel) const {
    return channel_avg[channel];
}

float LargeImages::get_channel_std_dev(int channel) const {
    return channel_std_dev[channel];
}

void LargeImages::calculate_avg_std_dev() {
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

LargeImage* LargeImages::copy_image(int i) const {
    return images[i].copy();
}


float LargeImages::get_raw_pixel(int subimage, int z, int y, int x) const {
    return images[subimage].pixels[z][y][x];
}

#ifdef LARGE_IMAGES_TEST
int main(int argc, char **argv) {
    int subimage_y = 32;
    int subimage_x = 32;
    int padding = 3;

    LargeImages large_images(argv[1], padding, subimage_y, subimage_x);

    cout << "number classes: " << large_images.get_number_classes() << endl;
    cout << "number images: " << large_images.get_number_images() << endl;

}
#endif
