
__kernel void addVector(__global int *a, __global int *b, __global int *c){

    int id = get_global_id(0);



    c[id] = a[id] + b[id];

}