#include "class.h"
#include <iostream>

int main(int argc, char const *argv[])
{
    Bar<int> b(4);
    Foo<decltype(b)> f(b);
    std::cout << "test successful" << std::endl;
    return 0;
}

