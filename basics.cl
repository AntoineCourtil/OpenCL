
__kernel void reduceV1(__global const int *a, __global int *b, int i) {

    int tid = get_global_id(0);
    int u = tid * (1 << i);
    int v = tid * (1 << i) + (1 << (i - 1));

    int sum = a[u] + a[v];

    b[u] = sum;

}


__kernel void reduceV2(__global const int *a, __global int *b, int i) {

    int tid = get_global_id(0);
    int u = tid * 2;
    int v = tid * 2 + 1;

    int sum = a[u] + a[v];

    b[tid] = sum;

}


__kernel void reduceV3(__global int *a, __global int *b, int i) {

    int tid = get_global_id(0);
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    for (int j = 1; j <= 9; j++) {
        int v0, v1;
        bool active = lid < (1 << (8 - j));


        if (active) { //Thread active ?
            //read values
            v0 = a[gid * 256 + lid * 2];
            v1 = a[gid * 256 + lid * 2 + 1];
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (active) { //Thread active ?
            //write value
            a[gid * 256 + lid] = v0 + v1;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    b[gid] = a[gid * 256];

}