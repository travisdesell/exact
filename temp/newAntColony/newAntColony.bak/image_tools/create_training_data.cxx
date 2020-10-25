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

void read_images(string directory, std::vector<Mat> &images) {
    directory_iterator end_itr;
    for (directory_iterator itr(directory); itr != end_itr; itr++) {
        if (!is_directory(itr->status())) {
            if (itr->path().leaf().c_str()[0] == '.') continue;

            images.push_back(imread( itr->path().c_str(), CV_LOAD_IMAGE_COLOR ));
        }
    }
}

void write_images(ofstream &outfile, std::vector<Mat> images) {
    for (int i = 0; i < images.size(); i++) {
        Vec3b pixel;

        for (int j = 0; j < images[i].rows; j++) {
            for (int k = 0; k < images[i].cols; k++) {
                pixel = images[i].at<Vec3b>(j,k);

                outfile.write( (char*)&pixel.val, sizeof(char) * 3 );

                //cout << " " << pixel.val[0] << " " << pixel.val[1] << " " << pixel.val[2];
            }
        }
        //cout << endl;
    }
}


int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "error: incorrect arguments." << endl;
        cerr << "usage: " << endl;
        cerr << "    " << argv[0] << " <binary output file> <class 1 directory> <class 2 directory> ... <class N directory>" << endl;
        exit(1);
    }

    string binary_output_file = argv[1];
    vector<string> classes_files;
    for (int i = 2; i < argc; i++) {
        classes_files.push_back(argv[i]);
    }

    std::vector< std::vector<Mat> > images;
    for (int i = 0; i < classes_files.size(); i++) {
        images.push_back( std::vector<Mat>() );
        read_images(classes_files[i], images[i]);

        cout << "read " << images[i].size() << " images of class " << i << endl;
    }

    ofstream outfile;
    outfile.open(binary_output_file.c_str(), ios::out | ios::binary);

    int img_size = images[0][0].cols;
    cout << "images are: " << img_size << " x " << img_size << endl;

    int vals_per_pixel = 3;

    vector<int> initial_vals;
    initial_vals.push_back(images.size());
    initial_vals.push_back(img_size);
    initial_vals.push_back(vals_per_pixel);
    for (int i = 0; i < images.size(); i++) {
        initial_vals.push_back(images[i].size());
    }

    outfile.write( (char*)&initial_vals[0], initial_vals.size() * sizeof(int) );

    for (int i = 0; i < images.size(); i++) {
        write_images(outfile, images[i]);
        cout << "wrote " << images[i].size() << " images." << endl;
    }

    outfile.close();

    return 0;
}

