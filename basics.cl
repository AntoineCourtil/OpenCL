// ----------------------------------------------------------

__kernel void pascal(__global int *table, int width){

    table[get_global_id(0) + width * get_global_id(1)] = 1; //2D

}

// ----------------------------------------------------------
