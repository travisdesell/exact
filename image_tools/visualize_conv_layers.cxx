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

//from UNDVC_COMMON
#include "vector_io.hxx"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    Mat image;

    if (argc != 4) {
        cerr << "error: incorrect arguments." << endl;
        cerr << "usage: " << endl;
        cerr << "    ./" << argv[0] << " <img reduction> <conv net file> <input image>" << endl;
        exit(1);
    }

    int img_reduction = atoi(argv[1]);
    string conv_net_file = argv[2];
    string input_image = argv[3];
   

    cout << "opening conv net file: " << conv_net_file << endl;
    ifstream infile(conv_net_file.c_str());

    string line;
    getline(infile, line);
    getline(infile, line);

    vector<double> weights;
    string_to_vector(line, weights);

    cout << "weights vector size: " << weights.size() << endl;

    infile.close();

    cout << "opening image: " << input_image << endl;
    Mat src = imread( input_image.c_str(), CV_LOAD_IMAGE_COLOR);

    cout << "source image size:       " << src.cols << ", rows: " << src.rows << endl;

    Size size(src.cols / img_reduction, src.rows / img_reduction); /*width, height*/

    Mat resized_image;
    resize(src, resized_image, size);

    imshow("resized image", resized_image);
    imwrite("resized_image.png", resized_image);

    cout << "resized image size:      " << resized_image.cols << "x" << resized_image.rows << endl;

    Mat color_filter_image = Mat::zeros( resized_image.rows, resized_image.cols, CV_64F );   //I guess zeros is backwards???
    cout << "color filter image size: " << color_filter_image.cols << "x" << color_filter_image.rows << endl;

    double result;

    for (int i = 0; i < resized_image.rows; i++) {
        for (int j = 0; j < resized_image.cols; j++) {
            Vec3b pixel = resized_image.at<Vec3b>(i,j);
            //cout << "got pixel [" << i << ", " << j << "]: " << pixel << endl;
            double r = pixel[0] / 256.0;
            double g = pixel[1] / 256.0;
            double b = pixel[2] / 256.0;
            //cout << "rgb: " << r << ", " << g << ", " << b << endl;

            result  = weights[0] * r;
            result += weights[1] * g;
            result += weights[2] * b;
            result += weights[3];

            result = 1.0 / (1.0 + exp(result));
            //result *= 1000.0;

            //cout << "result: " << result << ", uchar: " << (unsigned short)result << endl;

            color_filter_image.at<double>(i,j) = (double)result;
            //cout << "set pixel [" << j << ", " << i << "]: " << (unsigned short)color_filter_image.at<uchar>(i,j) << endl;
        }
    }

    //imshow("color filter image", color_filter_image);

    int img_size = 32;
    int current_weight = 4;
    vector<double> conv_sizes;
    vector<double> max_pool_sizes;

    conv_sizes.push_back(8);
    conv_sizes.push_back(6);
    max_pool_sizes.push_back(2);
    max_pool_sizes.push_back(2);

    Mat conv_images[2];
    Mat pool_images[2];

    for (int layer = 0; layer < conv_sizes.size(); layer++) {
        int conv_size = conv_sizes[layer];
        conv_images[layer] = Mat::zeros( color_filter_image.rows - conv_size, color_filter_image.cols - conv_size, CV_64F );   //I guess zeros is backwards???

        double val;
        //int bias_weight = current_weight + (conv_size * conv_size);
        for (int j = 0; j < conv_images[layer].rows; j++) {
            for (int k = 0; k < conv_images[layer].cols; k++) {

                val = 0;
                //cout << "calculating nodes[" << out_layer << "][" << j << "][" << k << "]: " << endl;
                for (int l = 0; l < conv_size; l++) {
                    for (int m = 0; m < conv_size; m++) {
                        val += weights[current_weight + (l * conv_size) + m] * color_filter_image.at<double>(j + l, k + m);
                    }
                }

                //val += weights[bias_weight];
                //bias_weight++;

                val = 1.0 / (1.0 + exp(val));
                conv_images[layer].at<double>(j,k)  = val;
            }
        }
        current_weight += (img_size - conv_size * img_size - conv_size);
        img_size -= conv_size;

        ostringstream conv_name;
        conv_name << "convolutional layer " << layer;

        imshow(conv_name.str().c_str(), conv_images[layer]);

        /*
           Mat conv_out(conv_images[layer]);
           for (int i = 0; i < conv_out.rows; i++) {
           for (int j = 0; j < conv_out.cols; j++) {
           conv_out.at<double>(i, j) *= 256;
           }
           }

           imwrite("conv.png", conv_out);
           */

        int max_pool_size = max_pool_sizes[layer];
        pool_images[layer] = Mat::zeros( conv_images[layer].rows / max_pool_size, conv_images[layer].cols / max_pool_size, CV_64F );   //I guess zeros is backwards???

        for (int j = 0; j < pool_images[layer].rows; j++) {
            for (int k = 0; k < pool_images[layer].cols; k++) {

                for (int l = 0; l < max_pool_size; l++) {
                    for (int m = 0; m < max_pool_size; m++) {
                        double val = conv_images[layer].at<double>((j * max_pool_size) + l,(k * max_pool_size) + m);
                        //double val = weights[current_weight + (l * max_pool_size) + m] * nodes[in_layer][(j * max_pool_size) + l][(k * max_pool_size) + m];
                        if (pool_images[layer].at<double>(j, k) < val) pool_images[layer].at<double>(j,k) = val;
                    }
                }
            }
        }
        img_size /= conv_size;

        ostringstream pool_name;
        pool_name << "pooling layer " << layer;

        imshow(pool_name.str().c_str(), pool_images[layer]);

        color_filter_image = pool_images[layer];
    }


    waitKey(0);

    return 0;
}

