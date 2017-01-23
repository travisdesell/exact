To compile BOINC make sure c++11 is used, after ./_autosetup

export CXXFLAGS="-std=c++11"
./configure --disable-client --disable-manager
