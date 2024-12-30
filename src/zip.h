#ifndef ZIP_H
#define ZIP_H

#include <vector>
#include <functional>
#include <cstdint>
#include <string> // Add this for std::string

class Zip {
    std::vector<unsigned char> data;

    void procData(int o, int cS, int uS, const std::function<void(const char*, int)>& p);

public:
    Zip() = default; // Default constructor
    explicit Zip(const std::string& filePath); // Constructor to initialize with file path
    void ext(const std::string& fn, const std::function<void(const char*, int)>& cb); // Add this declaration
    void read(const std::vector<unsigned char>& d);
};


#endif // ZIP_H

