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
    cl::Kernel *krnV1 = cluLoadKernel(prg, "reduceV1");

    int p2 = 16;
    int n = (1 << p2);

    //création du buffer = allocation mémoire du GPU
    cl::Buffer bufferA(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));
    cl::Buffer bufferB(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));

    cl::Buffer *buffer[2];
    buffer[0] = &bufferA;
    buffer[1] = &bufferB;

    int swap = 0;
    int *table = new int[n];
    double total_chrono_V1 = 0;


    //Init du tableau
    for (int i = 0; i < n; i++) {
        table[i] = 1;
    }


    //init des buffers avec les tableaux
    clu_Queue->enqueueWriteBuffer(*buffer[swap], false, 0, n * sizeof(int), table);


    for (int i = 1; i <= p2; i++) {
        krnV1->setArg(0, *buffer[swap]);
        krnV1->setArg(1, *buffer[1 - swap]);
        krnV1->setArg(2, i);


        int numT = n / (1 << i);
        int numG = min(32, n / (1 << i));


        cl::Event ev;

        //Ordre par file de commande
        cl_int err = clu_Queue->enqueueNDRangeKernel(
                *krnV1, //kernel
                cl::NullRange, //NullRange
                cl::NDRange(numT), //NB de Threads
                cl::NDRange(numG), //Taille de groupe, n doit etre un multiple de taille de groupe
                0,
                &ev //Event de mesure de performances
        );


        cluCheckError(err, "Error executing kernel");

        ev.wait();

        total_chrono_V1 += cluEventMilliseconds(ev);


        swap = 1 - swap;


        cerr << "Pass " << i << " | numT : " << numT << " | numG : " << numG << endl;


    }


    cerr << endl << "[+] V1, total time : " << total_chrono_V1 << endl;

    int sumV1 = 0;
    clu_Queue->enqueueReadBuffer(*buffer[swap], true, 0, sizeof(int), &sumV1);

    cerr << "   sumV1 : " << sumV1 <<endl;


    delete[](table);

    return 0;
}

// ----------------------------------------------------------

