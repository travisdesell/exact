#include <sstream>
#include <vector>
#include <iostream>
#include <cstdlib>

using namespace std;

bool argument_exists(vector<string> arguments, string argument) {
    for (unsigned int i = 0; i < arguments.size(); i++) {
        if (argument.compare(arguments.at(i)) == 0) {
//            cerr << "parsed argument '" << argument << "' successfully." << endl;
            return true;
        }
    }
    return false;
}

//template <>
//bool get_argument<string>(vector<string> arguments, string argument, bool required, string &result) {
bool get_argument(vector<string> arguments, string argument, bool required, string &result) {
    bool found = false;
    for (unsigned int i = 0; i < arguments.size(); i++) {
        if (argument.compare(arguments.at(i)) == 0) {
            result = arguments.at(++i);
            found = true;
            break;
        }
    }

    if (required && !found) {
        cerr << "ERROR: argument '" << argument << "' required and not found." << endl;
        exit(1);
    }

    if (found) {
//        cerr << "parsed argument '" << argument << "' successfully: " << result << endl;
    }
    return found;
}


//template <>
//bool get_argument_vector<string>(vector<string> arguments, string argument, bool required, vector<string> &results) {
bool get_argument_vector(vector<string> arguments, string argument, bool required, vector<string> &results) {
    bool found = false;
    for (unsigned int i = 0; i < arguments.size(); i++) {
        if (argument.compare(arguments.at(i)) == 0) {
            i++;
            while (i < arguments.size() && arguments.at(i).substr(0,2).compare("--") != 0) {
                results.push_back(arguments.at(i++));
            }
            found = true;
            break;
        }
    }

    if (required && !found) {
        cerr << "ERROR: argument '" << argument << "' required and not found." << endl;
        exit(1);
    }

    if (found) {
        /*
        cerr << "parsed argument '" << argument << "' successfully:";
        for (unsigned int i = 0; i < results.size(); i++) {
            cerr << " " << results.at(i);
        }
        cerr << endl;
        */
    }
    return found;
}


