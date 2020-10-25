#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include <boost/filesystem.hpp>
using boost::filesystem::create_directories;
using boost::filesystem::directory_iterator;
using boost::filesystem::is_directory;

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    Mat image;

    if (argc != 6) {
        cerr << "error: incorrect arguments." << endl;
        cerr << "usage: " << endl;
        cerr << "    ./" << argv[0] << " <input directory> <output directory> <img size> <rotate?> <convert to HSV?>" << endl;
        exit(1);
    }

    string input_directory = argv[1];
    string output_directory = argv[2];
    int img_size = atoi(argv[3]);
    int rotate = atoi(argv[4]);
    int hsv = atoi(argv[5]);

    cout << "creating directory (if it does not exist): '" << output_directory.c_str() << "'" << endl;
    create_directories(output_directory);

    int count = 0;
    directory_iterator end_itr;
    for (directory_iterator itr(input_directory); itr != end_itr; itr++) {
        if (!is_directory(itr->status())) {
            if (itr->path().leaf().c_str()[0] == '.') {
                cout << "skipping: " << itr->path().c_str() << endl;
                continue;
            }

            cout << "resizing file: '" << itr->path().c_str() << endl;

            ostringstream output_filename;
            output_filename << output_directory << "/" << itr->path().leaf().c_str();

            cout << "writing to:    '" << output_filename.str() << "'" << endl;

            Size size(img_size, img_size);
            Mat src = imread( itr->path().c_str() );
            Mat dst;
            if (img_size != 0) {
                resize(src, dst, size);
            } else {
                dst = src;
            }

            if (hsv) {
                cvtColor(dst, dst, CV_BGR2HSV);
            }

            imwrite(output_filename.str().c_str(), dst);

            if (rotate == 1) {
                int file_pos = output_filename.str().rfind('.');
                string filebase = output_filename.str().substr(0, file_pos);
                string filetype = output_filename.str().substr(file_pos, output_filename.str().size() - file_pos);

                //cout << "base: '" << filebase << "'" << endl;
                //cout << "type: '" << filetype << "'" << endl;

                Mat rot(dst);
                for (int i = 1; i < 4; i++) {
                    transpose(rot, rot);
                    flip(rot, rot, 1);

                    ostringstream of;
                    of << filebase << "_" << i << filetype;

                    cout << "writing to:    '" << of.str() << "'" << endl;
                    imwrite( of.str().c_str(), rot );
                }

            }

            count++;
        } else {
            cout << "skipping directory: " << itr->path().c_str() << endl;
        }
    }

    cout << "resized " << count << " files." << endl;
    return 0;
}

