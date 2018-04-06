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

    ///######################################################
    ///#                     EXERCICE 1                     #
    ///######################################################

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
                cl::NDRange(i + 1), //NB de Threads
                cl::NDRange(1), //Taille de groupe, n doit etre un multiple de taille de groupe
                0,
                &ev //Event de mesure de performances
        );

        cluCheckError(err, "[+] Error executing query to kernel");
        clu_Queue->finish();


        //Lecture du résultat dans table
        clu_Queue->enqueueReadBuffer(nouvelleLigne, false, 0, (i + 1) * sizeof(int), table);
        clu_Queue->finish();

        for (int j = 0; j <= i; j++) {
            cerr << table[j] << ", ";
        }
        cerr << endl;



        //Met table dans ancienneLigne
        clu_Queue->enqueueWriteBuffer(ancienneLigne, false, 0, (i + 1) * sizeof(int), table);
        clu_Queue->finish();
    }

    cerr << endl << endl;

    ev.wait();
    cluDisplayEventMilliseconds("kernel time", ev);


    //suppression de mémoire
    delete[](table);




    ///######################################################
    ///#                     EXERCICE 2                     #
    ///######################################################


    ///------------------- METHODE 1 ------------------------

    cl::Kernel *krnMethod1 = cluLoadKernel(prg, "method1");

    n = 1024;

    cl::Buffer buffer_(*clu_Context, CL_MEM_READ_WRITE, n * sizeof(int));
    cl::Buffer res_(*clu_Context, CL_MEM_READ_WRITE, sizeof(int));


    int *table_ = new int[n];

    //Init du tableau
    for (int i = 0; i < n; i++) {
        table_[i] = i;
    }

    krnMethod1->setArg(0, buffer_);
    krnMethod1->setArg(1, res_);

    int resCPU = 0;
    int resGPU = 0;

    clu_Queue->enqueueWriteBuffer(buffer_, false, 0, n * sizeof(int), table_);
    clu_Queue->enqueueWriteBuffer(res_, false, 0, sizeof(int), nullptr);
    clu_Queue->finish();

    clu_Queue->enqueueNDRangeKernel(*krnMethod1, cl::NullRange, cl::NDRange(n), cl::NDRange(32));
    clu_Queue->finish();

    clu_Queue->enqueueReadBuffer(res_, false, 0, sizeof(int), &resGPU);

    clu_Queue->finish();


    long long tm_start = cluCPUMilliseconds();
    for (int i = 0; i < n; i++) {
        resCPU += table_[i];
    }
    long long tm_stop = cluCPUMilliseconds();

    cerr << endl << endl << "CPU  = " << resCPU << "     -     [CPU time] " << (tm_stop - tm_start) << " msecs" << endl
         << endl;
    cerr << "GPU  = " << resGPU << "     -     ";

    ev.wait();
    cluDisplayEventMilliseconds("GPU time", ev);



    ///------------------- METHODE 2 ------------------------

    cl::Kernel *krnMethod2 = cluLoadKernel(prg, "method2");

    n = 1024;
    int k = 64;


    krnMethod2->setArg(0, buffer_);
    krnMethod2->setArg(1, res_);
    krnMethod2->setArg(2, k);

    resGPU = 0;

    clu_Queue->enqueueWriteBuffer(buffer_, false, 0, n * sizeof(int), table_);
    clu_Queue->enqueueWriteBuffer(res_, false, 0, sizeof(int), nullptr);
    clu_Queue->finish();

    clu_Queue->enqueueNDRangeKernel(*krnMethod1, cl::NullRange, cl::NDRange(n / k), cl::NDRange(32));
    clu_Queue->finish();

    clu_Queue->enqueueReadBuffer(res_, false, 0, sizeof(int), &resGPU);

    clu_Queue->finish();


    cerr << "GPU2 = " << resGPU << "     -     ";

    ev.wait();
    cluDisplayEventMilliseconds("GPU2 time", ev);



    ///------------------- METHODE 3 ------------------------

    cl::Kernel *krnMethod3 = cluLoadKernel(prg, "method3");


    n = 1024;
    int s = 32;


    cl::Buffer res3_(*clu_Context, CL_MEM_READ_WRITE, s * sizeof(int));

    int *table3_ = new int[s];

    //Init du tableau
    for (int i = 0; i < s; i++) {
        table3_[i] = 0;
    }


    krnMethod3->setArg(0, buffer_);
    krnMethod3->setArg(1, res3_);
    krnMethod3->setArg(2, s);

    resGPU = 0;

    clu_Queue->enqueueWriteBuffer(buffer_, false, 0, n * sizeof(int), table_);
    clu_Queue->enqueueWriteBuffer(res3_, false, 0, s * sizeof(int), table3_);
    clu_Queue->finish();

    clu_Queue->enqueueNDRangeKernel(*krnMethod1, cl::NullRange, cl::NDRange(n), cl::NDRange(32));
    clu_Queue->finish();

    clu_Queue->enqueueReadBuffer(res3_, false, 0, s * sizeof(int), table3_);
    clu_Queue->finish();

    for (int i = 0; i < s; i++) {
        resGPU += table3_[i];
    }

    cerr << "GPU3 = " << resGPU << "     -     ";

    ev.wait();
    cluDisplayEventMilliseconds("GPU3 time", ev);




    //suppression de mémoire
    delete[](table_);

    return 0;
}

// ----------------------------------------------------------

