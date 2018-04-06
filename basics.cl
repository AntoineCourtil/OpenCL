// ----------------------------------------------------------

__kernel void pascal(__global int *table, int size){

//    table[get_global_id(0) + width * get_global_id(1)] = 0; //2D

    table[get_global_id(0)] = 0; //1D

}

// ----------------------------------------------------------
