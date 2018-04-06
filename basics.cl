// ----------------------------------------------------------

__kernel void pascal(__global int *table, __global int *res, int noligne) {

//    table[get_global_id(0) + width * get_global_id(1)] = 0; //2D

    if (get_global_id(0) == 0 || get_global_id(0) == noligne) {
        res[get_global_id(0)] = 1;
    } else {
        res[get_global_id(0)] = table[get_global_id(0)] + table[get_global_id(0)-1]; //1D
//        res[get_global_id(0)] = 88; //1D
    }

}


__kernel void method1(__global int *table, __global int *res){

    atomic_add(res, table[get_global_id(0)]);
}


__kernel void method2(__global int *table, __global int *res, int k){

    int indiceDepart = get_global_id(0) * k;
    int add = 0;

    for(int i=0; i<k; i++) {
        add += table[indiceDepart+i];
    }

    atomic_add(res, add);
}

__kernel void method3(__global int *table, __global int *res, int s){

    int idSubCounter = get_global_id(0) % s;

    atomic_add(&res[idSubCounter], table[get_global_id(0)]);
//    atomic_add(&res[0], table[get_global_id(0)]);
}

// ----------------------------------------------------------
