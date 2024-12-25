#include <iostream>  // For std::cout, std::cin, std::endl
#include "Xi.h"

int main() {
    Xi::loadModel("data/model.bz2");
    Xi::train(10);
    // Example conversation loop
    std::string userInput;
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, userInput);
        if (userInput == "exit") break;
        std::cout << "Xi: " << Xi::generateResponse(userInput) << std::endl;
    }
    return 0;
}
