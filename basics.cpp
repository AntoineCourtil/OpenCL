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
    cl::Kernel *kernelEven = cluLoadKernel(prg, "even");
    cl::Kernel *kernelOdd = cluLoadKernel(prg, "odd");

    int n = 24;
    int groupSize = 4;

    //création du buffer = allocation mémoire du GPU
    cl::Buffer buffer(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));

    kernelEven->setArg(0, buffer);
    kernelEven->setArg(1, n);

    kernelOdd->setArg(0, buffer);
    kernelOdd->setArg(1, n);

    int *table = new int[n];

    double total_chrono = 0;


    //Init du tableau
    for (int i = 0; i < n; i++) {
        table[i] = i;
    }


    //init des buffers avec les tableaux
    clu_Queue->enqueueWriteBuffer(buffer, true, 0, n * sizeof(int), table);


    for (int i = 0; i < n; i++) {

        if (i % 2 == 0) {

            cl::Event ev;

            //Ordre par file de commande
            cl_int err = clu_Queue->enqueueNDRangeKernel(
                    *kernelEven, //kernelOdd
                    cl::NullRange, //NullRange
                    cl::NDRange(n / 2), //NB de Threads
                    cl::NDRange(groupSize), //Taille de groupe, n doit etre un multiple de taille de groupe
                    0,
                    &ev //Event de mesure de performances
            );


            cluCheckError(err, "Error executing kernelOdd");

            ev.wait();

//            cluDisplayEventMilliseconds("[+] kernelOdd time", ev);
//            cerr << endl;

            total_chrono += cluEventMilliseconds(ev);

        } else {

            cl::Event ev;

            //Ordre par file de commande
            cl_int err = clu_Queue->enqueueNDRangeKernel(
                    *kernelOdd, //kernelOdd
                    cl::NullRange, //NullRange
                    cl::NDRange(n / 2), //NB de Threads
                    cl::NDRange(groupSize), //Taille de groupe, n doit etre un multiple de taille de groupe
                    0,
                    &ev //Event de mesure de performances
            );


            cluCheckError(err, "Error executing kernelOdd");

            ev.wait();

//            cluDisplayEventMilliseconds("[+] kernelOdd time", ev);
//            cerr << endl;

            total_chrono += cluEventMilliseconds(ev);

        }

    }


    cerr << "[+] Total time : " << total_chrono << " ms" << endl << endl << endlg;


    clu_Queue->enqueueReadBuffer(buffer, true, 0, n * sizeof(int), table);


    for (int i = 0; i < n; i++) {
        cerr << table[i] << " " << endl;
    }


    ///////////////////////////////////
    ///             V2              ///
    ///////////////////////////////////




    delete[](table);


    return 0;
}

// ----------------------------------------------------------

