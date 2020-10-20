#ifndef STIRNG_FORMAT_HXX
#define STIRNG_FORMAT_HXX

#include <memory>

#include <string>
using namespace std;

#include <stdexcept>

// TODO: this can go away when C++20 is adopted, string::format
// https://en.cppreference.com/w/cpp/utility/format/formatter#Standard_format_specification
// Credit to https://stackoverflow.com/a/26221725/4102299
template<typename ... Args>
inline std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] ); 
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

#endif
