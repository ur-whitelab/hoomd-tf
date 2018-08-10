#include "../../ipcarraycomm/IPCArrayComm.h"
#include <iostream>

int main() {
    GPUArray<Scalar4> array(100, NULL);
    //IPCArrayComm<IPCCommMode::CPU, Scalar4> ipc(array);
    std::cout << "Hi";
}