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
    cl::Kernel *krn = cluLoadKernel(prg, "addVector");

    int n = 1024 * 1024 * 64;

    //création du buffer = allocation mémoire du GPU
    cl::Buffer bufferA(*clu_Context, CL_MEM_WRITE_ONLY, n * sizeof(int));
    cl::Buffer bufferB(*clu_Context, CL_MEM_WRITE_ONLY, n * sizeof(int));
    cl::Buffer bufferC(*clu_Context, CL_MEM_WRITE_ONLY, n * sizeof(int));

    krn->setArg(0, bufferA);
    krn->setArg(1, bufferB);
    krn->setArg(2, bufferC);

    int *tableA = new int[n];
    int *tableB = new int[n];
    int *tableC = new int[n];


    //Init des tableaux A et B
    for (int i = 0; i < n; i++) {
        tableA[i] = i ;
    }

    for (int i = 0; i < n; i++) {
        tableB[i] = n-i ;
    }

    //init des buffers avec les tableaux
    clu_Queue->enqueueWriteBuffer(bufferA, false, 0, n * sizeof(int), tableA);
    clu_Queue->enqueueWriteBuffer(bufferB, false, 0, n * sizeof(int), tableB);



    cl::Event ev;

    //Ordre par file de commande
    cl_int err = clu_Queue->enqueueNDRangeKernel(
            *krn, //kernel
            cl::NullRange, //NullRange
            cl::NDRange(n), //NB de Threads
            cl::NDRange(32), //Taille de groupe, n doit etre un multiple de taille de groupe
            0,
            &ev //Event de mesure de performances
    );


    cluCheckError(err, "Error executing kernel");



    //int table[n]; //mémoire sur pile d'éxécution + pas de variable n dans la déclaration

    //int table[1024]; //pareil mémoire sur pile d'éxécution, + si grosse structure = égale car pas assez de place



    //copie de bufferC dans tableC
    clu_Queue->enqueueReadBuffer(bufferC, false, 0, n * sizeof(int), tableC);

    clu_Queue->finish();

    for (int i = 0; i < 10; i++) {
        cerr << tableC[i] << ", ";
    }

    cerr << endl << endl;

    ev.wait();
    cluDisplayEventMilliseconds("kernel time : ", ev);


    long long tm_start = cluCPUMilliseconds();
    for (int i = 0; i < n; i++) {
        tableC[i] = tableA[i] + tableB[i] ;
    }
    long long tm_stop = cluCPUMilliseconds();
    cerr << "CPU time : " << (tm_stop - tm_start) << " msecs" << endl;


    //suppression de mémoire
    delete[](tableA);
    delete[](tableB);
    delete[](tableC);

    return 0;
}

// ----------------------------------------------------------

