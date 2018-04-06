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

//    int width = 10;
//    int height = 10;
//    int n = width * height;
    int n = 30;

    //création du buffer = allocation mémoire du GPU
    cl::Buffer ancienneLigne(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));
    cl::Buffer nouvelleLigne(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));

    krn->setArg(0, ancienneLigne);
    krn->setArg(1, nouvelleLigne);
    krn->setArg(2, n);

    int *table = new int[n];


    //Init du tableau
    for (int i = 0; i < n; i++) {
        table[i] = 0;
    }

    //Init du buffer avec le tableau
    clu_Queue->enqueueWriteBuffer(ancienneLigne, false, 0, n * sizeof(int), table);
    clu_Queue->enqueueWriteBuffer(nouvelleLigne, false, 0, n * sizeof(int), table);


    cl::Event ev;






    for (int i = 0; i < n; i++) {


        krn->setArg(2, i);

        //Ordre par file de commande
        cl_int err = clu_Queue->enqueueNDRangeKernel(
                *krn, //kernel
                cl::NullRange, //NullRange
                cl::NDRange(i+1), //NB de Threads
                cl::NDRange(1), //Taille de groupe, n doit etre un multiple de taille de groupe
                0,
                &ev //Event de mesure de performances
        );

        cluCheckError(err, "[+] Error executing query to kernel");
        clu_Queue->finish();


        //Lecture du résultat dans table
        clu_Queue->enqueueReadBuffer(nouvelleLigne, false, 0, (i+1) * sizeof(int), table);
        clu_Queue->finish();

        for (int j = 0; j <= i; j++) {
            cerr << table[j] << ", ";
        }
        cerr << endl;



        //Met table dans ancienneLigne
        clu_Queue->enqueueWriteBuffer(ancienneLigne, false, 0, (i+1) * sizeof(int), table);
        clu_Queue->finish();
    }

    cerr << endl << endl;

    ev.wait();
    cluDisplayEventMilliseconds("kernel time", ev);


    //suppression de mémoire
    delete[](table);


    return 0;
}

// ----------------------------------------------------------

