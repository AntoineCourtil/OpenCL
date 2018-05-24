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

#define ASC 0
#define DESC 1


// ----------------------------------------------------------

void swap(int *T, int id1, int id2) {
    int swap = T[id1];

    T[id1] = T[id2];
    T[id2] = swap;
}


void compare(int *T, int L, int R, int d) {
    int k = (R - L + 1) / 2;

    for (int i = 0; i < k; i++) {
        if (d == DESC) {
            if (T[L + i] < T[L + i + k]) {
                swap(T, L + i, L + i + k);
            }
        } else { ///DESC
            if (T[L + i] > T[L + i + k]) {
                swap(T, L + i, L + i + k);
            }
        }
    }
}


void merge(int *T, int L, int R, int d) {
    if ((R - L + 1) > 1) {
        compare(T, L, R, d);
        merge(T, L, (L + R) / 2, d);
        merge(T, ((L + R) / 2) + 1, R, d);
    }
}


void sort(int *T, int L, int R, int d) {
    if ((R - L + 1) > 1) {
        sort(T, L, (L + R) / 2, ASC);
        sort(T, ((L + R) / 2) + 1, R, DESC);
        merge(T, L, R, d);
    }
}

// ----------------------------------------------------------

int main(int argc, char **argv) {

    const char *clu_file = SRC_PATH "basics.cl";

    cluInit();

    cl::Program *prg = cluLoadProgram(clu_file);
    cl::Kernel *kernelEven = cluLoadKernel(prg, "even");
    cl::Kernel *kernelOdd = cluLoadKernel(prg, "odd");
    cl::Kernel *kernel = cluLoadKernel(prg, "bitonic");

    int n = 24;
    int groupSize = 4;

    //création du buffer = allocation mémoire du GPU
    cl::Buffer buffer(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));
    cl::Buffer buffer2(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));

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


    cerr << "[+] Total time : " << total_chrono << " ms" << endl << endl << endl;


    clu_Queue->enqueueReadBuffer(buffer, true, 0, n * sizeof(int), table);


    for (int i = 0; i < n; i++) {
        cerr << table[i] << " " << endl;
    }


    ///////////////////////////////////
    ///       BITONIC CPU           ///
    ///////////////////////////////////


//    cerr << endl << endl << "###    BITONIC CPU    ###" << endl << endl;
//
//    sort(table2, 0, n - 1, DESC);
//
//
//    for (int i = 0; i < n; i++) {
//        cerr << table2[i] << " " << endl;
//    }





    ///////////////////////////////////
    ///       BITONIC GPU           ///
    ///////////////////////////////////

    cerr << endl << endl << "###    BITONIC GPU    ###" << endl << endl;

    int logn = 4;
    int nn = 1 << logn;
    int *table2 = new int[n];

    //Init du tableau
    for (int i = 0; i < nn; i++) {
        table2[i] = i;
    }


    clu_Queue->enqueueWriteBuffer(buffer2, true, 0, n * sizeof(int), table2);

    for(int stage=0; stage < logn; stage++){
        for(int col = stage; col >= 0; col--){

            kernel->setArg(0, buffer2);
            kernel->setArg(1, stage);
            kernel->setArg(2, col);

            cl::Event ev;

            //Ordre par file de commande
            cl_int err = clu_Queue->enqueueNDRangeKernel(
                    *kernel, //kernelOdd
                    cl::NullRange, //NullRange
                    cl::NDRange(nn / 2), //NB de Threads
                    cl::NDRange(1), //Taille de groupe, n doit etre un multiple de taille de groupe
                    0,
                    &ev //Event de mesure de performances
            );


            cluCheckError(err, "Error executing kernel");

            ev.wait();

        }
    }


    clu_Queue->enqueueReadBuffer(buffer2, true, 0, n * sizeof(int), table2);

    for (int i = 0; i < nn; i++) {
        cerr << table2[i] << " " << endl;
    }

    delete[](table);
    delete[](table2);


    return 0;
}

// ----------------------------------------------------------

