#include "Xi.h"

int main() {
    Xi xi(100, 0.01, 0.001);
    xi.init("data/model.bz2", "data/conversations.json");
    xi.train(10);
    xi.chat();

    return 0;
}
