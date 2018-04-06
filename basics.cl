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

// ----------------------------------------------------------
