
__kernel void reduceV1(__global const int *a, __global int *b, int i){

    int tid = get_global_id(0);
    int u = tid * (1<<i);
    int v = tid * (1<<i) + (1<<(i-1));

    int sum = a[u] + a[v];

    b[u] = sum;

}


__kernel void reduceV2(__global const int *a, __global int *b, int i){

    int tid = get_global_id(0);
    int u = tid * 2;
    int v = tid * 2 + 1;

    int sum = a[u] + a[v];

    b[tid] = sum;

}