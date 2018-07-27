#include "hoomd/HOOMDMath.h"
#include <string.h>
#include <iostream>

#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif
#include <memory>
#include <string>
#include <cstdlib>

//https://stackoverflow.com/a/20170989
template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

int main() {

  //make some data in scalar4s
  unsigned int N = 4;
  Scalar4* input = new Scalar4[N];
  for (unsigned int i = 0; i < N; i++) {
    input[i].x = 0.;
    input[i].y = 2.;
    input[i].z = 3.;
    input[i].w = 4.;
  }

  std::cout << type_name<decltype(input[0].x)>() << std::endl;

  //check struct alignment
  float* test = reinterpret_cast<float*> (input);
  std::cout << "[ ";
  for (unsigned int i = 0; i < 4; i++)
      std::cout << test[i] << " ";
  std::cout << "]" << std::endl;

  test += 2 * 4;
  std::cout << "[ ";
  for (unsigned int i = 0; i < 4; i++)
      std::cout << test[i] << " ";
  std::cout << "]" << std::endl;

  float* buffer = new float[N * 4];
  std::memcpy(buffer, input, sizeof(float) * N * 4);
  for (unsigned int i = 0; i < N; i++) {
    std::cout << "[ ";
    for (unsigned int j = 0; j < 4; j++)
      std::cout << buffer[i * 4 + j] << " ";
    std::cout << "]" << std::endl;
  }

  delete buffer;
  delete input;
}