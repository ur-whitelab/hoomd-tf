#include <iostream>

template <typename T>
class Bar {
    public:
    Bar(T v) : value(v)
    {}
    using value_type = T;
    T value;
};

template <typename T>
class Foo {
    public:
    Foo(const T& other) : value(other.value)
    {std::cout << "the value is " << value << std::endl;}
    typename T::value_type value;
};
