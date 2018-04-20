// ----------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

// ----------------------------------------------------------

// We provide a small library so we can easily setup OpenCL
#include "clutils.h"

// ----------------------------------------------------------

int main(int argc, char **argv) {

    const char *clu_file = SRC_PATH "basics.cl";

    cluInit();

    cl::Program *prg = cluLoadProgram(clu_file);
    cl::Kernel *kernel = cluLoadKernel(prg, "wsumsV1");

    int p2 = 20;
    int n = (1 << p2);
    int k = 128;

    //création du buffer = allocation mémoire du GPU
    cl::Buffer bufferA(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));
    cl::Buffer bufferB(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));

    kernel->setArg(0, bufferA);
    kernel->setArg(1, bufferB);
    kernel->setArg(2, n);
    kernel->setArg(3, k);

    int *table = new int[n];
    int *wsums = new int[n];


    //Init du tableau
    for (int i = 0; i < n; i++) {
        table[i] = 2*i;
        wsums[i] = 0;
    }


    //init des buffers avec les tableaux
    clu_Queue->enqueueWriteBuffer(bufferA, true, 0, n * sizeof(int), table);
    clu_Queue->enqueueWriteBuffer(bufferB, true, 0, n * sizeof(int), wsums);


    cl::Event ev;

    //Ordre par file de commande
    cl_int err = clu_Queue->enqueueNDRangeKernel(
            *kernel, //kernel
            cl::NullRange, //NullRange
            cl::NDRange(n), //NB de Threads
            cl::NDRange((1<<10)), //Taille de groupe, n doit etre un multiple de taille de groupe
            0,
            &ev //Event de mesure de performances
    );


    cluCheckError(err, "Error executing kernel");

    ev.wait();

    cluDisplayEventMilliseconds("[+] kernel time", ev);
    cerr << endl;


    clu_Queue->enqueueReadBuffer(bufferB, true, 0, n * sizeof(int), wsums);


    for (int i = 0; i < 10; i++) {
        cerr << wsums[i] << " " << endl;
    }


    delete[](table);
    delete[](wsums);

    return 0;
}

// ----------------------------------------------------------

