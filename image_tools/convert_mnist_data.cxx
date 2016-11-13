#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

#include <boost/filesystem.hpp>
using boost::filesystem::directory_iterator;

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

void write_images(ofstream &outfile, std::vector<Mat> images) {
    for (int i = 0; i < images.size(); i++) {
        unsigned char pixel;

        for (int j = 0; j < images[i].rows; j++) {
            for (int k = 0; k < images[i].cols; k++) {
                pixel = images[i].at<uchar>(j,k);

                outfile.write( (char*)&pixel, sizeof(char));

//                cout << " " << (uint32_t)pixel;
            }
//            cout << endl;
        }
//        cout << endl;
    }
}

void flip_bytes(char *bytes, size_t size) {
    for (uint32_t i = 0; i < size / 2; i++) {
        char tmp = bytes[i];

        bytes[i] = bytes[(size - i) - 1];
        bytes[(size - i) - 1] = tmp;
    }
}

uint32_t fread_uint32_t(ifstream &file, string filename, const char *name, uint32_t expected_value) {
    uint32_t value;
    file.read( (char*)&value, sizeof(uint32_t) );
    flip_bytes( (char*)&value, sizeof(uint32_t) );

    cout << name << " is: " << value << endl;
    if (value != expected_value) {
        cerr << "Error reading from image file '" << filename << "'. Invalid format, " << name << " != " << expected_value << endl;
        exit(1);
    }

    return value;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "error: incorrect arguments." << endl;
        cerr << "usage: " << endl;
        cerr << "    " << argv[0] << " <mnist image file> <mnist label file> <output file>" << endl;
        exit(1);
    }

    string image_filename(argv[1]);
    string label_filename(argv[2]);
    string output_filename(argv[3]);

    ifstream image_file(image_filename.c_str(), ios::in | ios::binary);
    ifstream label_file(label_filename.c_str(), ios::in | ios::binary);

    if (!image_file.is_open()) {
        cerr << "Could not open '" << image_filename << "' for reading." << endl;
    }

    if (!label_file.is_open()) {
        cerr << "Could not open '" << label_filename << "' for reading." << endl;
    }

    uint32_t image_magic_number = fread_uint32_t(image_file, image_filename, "image file magic number", 2051);
    uint32_t number_images = fread_uint32_t(image_file, image_filename, "number images", 60000);
    uint32_t number_rows = fread_uint32_t(image_file, image_filename, "number rows", 28);
    uint32_t number_cols = fread_uint32_t(image_file, image_filename, "number cols", 28);

    uint32_t label_magic_number = fread_uint32_t(label_file, label_filename, "label file magic number", 2049);
    uint32_t number_labels = fread_uint32_t(label_file, label_filename, "number labels", 60000);

    std::vector< std::vector<Mat> > images(10);

    unsigned char pixel;
    unsigned char label;
    for (uint32_t i = 0; i < 60000; i++) {
        Mat image;
        image.create(28, 28, CV_8UC1);

        for (uint32_t j = 0; j < 28; j++) {
            for (uint32_t k = 0; k < 28; k++) {
                image_file.read( (char*)&pixel, sizeof(char) );

                //pixel = 255 - pixel;
                image.at<uchar>(j, k) = pixel;
            }
        }

        /*
        cout << "opencv Mat: " << image << endl;
        ostringstream name;
        name << "image " << i;

        imshow(name.str(), image);
        */

        label_file.read( (char*)&label, sizeof(char) );
        //cout << "label: " << (int)label << endl;

        images[ (int)label ].push_back(image);

        //char keystroke;
        //cin >> keystroke;
    }

    image_file.close();
    label_file.close();


    ofstream outfile;
    outfile.open(output_filename.c_str(), ios::out | ios::binary);

    int img_size = number_rows;
    int vals_per_pixel = 1;
    int number_classes = 10;

    vector<int> initial_vals;
    initial_vals.push_back(number_classes);
    initial_vals.push_back(img_size);
    initial_vals.push_back(vals_per_pixel);

    uint32_t sum = 0;
    for (int i = 0; i < images.size(); i++) {
        cout << "read " << images[i].size() << " images of class " << i << endl;
        initial_vals.push_back(images[i].size());
        sum += images[i].size();
    }
    cout << "read " << sum << " images in total" << endl;

    outfile.write( (char*)&initial_vals[0], initial_vals.size() * sizeof(int) );

    for (int i = 0; i < images.size(); i++) {
        write_images(outfile, images[i]);
        cout << "wrote " << images[i].size() << " images." << endl;
    }

    outfile.close();

    return 0;
}

