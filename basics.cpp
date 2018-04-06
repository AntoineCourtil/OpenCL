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
    cl::Kernel *krn = cluLoadKernel(prg, "pascal");

    int width = 10;
    int height = 10;
    int n = width * height;

    //création du buffer = allocation mémoire du GPU
    cl::Buffer buffer(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));

    krn->setArg(0, buffer);
    krn->setArg(1, width);

    int *table = new int[n];


    //Init du tableau
    for (int i = 0; i < n; i++) {
        table[i] = 1 ;
    }

    //Init du buffer avec le tableau
    clu_Queue->enqueueWriteBuffer(buffer, false, 0, n * sizeof(int), table);


    cl::Event ev;

    //Ordre par file de commande
    cl_int err = clu_Queue->enqueueNDRangeKernel(
            *krn, //kernel
            cl::NullRange, //NullRange
            cl::NDRange(n), //NB de Threads
            cl::NDRange(2), //Taille de groupe, n doit etre un multiple de taille de groupe
            0,
            &ev //Event de mesure de performances
    );


    cluCheckError(err, " - Error executing kernel");


    clu_Queue->finish();

    clu_Queue->enqueueReadBuffer(buffer, false, 0, n * sizeof(int), table);

    clu_Queue->finish();

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if(table[i] != 0) {
                cerr << table[i] << ", ";
            }
        }
        cerr << endl;
    }

    cerr << endl << endl;

    ev.wait();
    cluDisplayEventMilliseconds("kernel time", ev);


    return 0;
}

// ----------------------------------------------------------

