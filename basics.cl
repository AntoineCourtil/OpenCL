
__kernel void wsumsV1(__global const int *a, __global int *b, int n, int k) {

    int id = get_global_id(0);

    int wsum = 0;

    for (int j = -k; j <= k; j++) {
        wsum += a[(id + j) % n];
    }

    b[id] = wsum;

}


__kernel void wsumsV2(__global const int *a, __global int *b, __local int *tmp, int n, int k, int groupSize) {

    int id = get_global_id(0);
    int lid = get_local_id(0);
    int gid = get_group_id(0);

    int wsum = 0;



    if(gid == 0 && lid == 0) {
        for (int i = 0; i <= k; i++) {
            tmp[i] = 0;
            tmp[groupSize-i] = a[id+k+groupSize+i];
        }
    } else {

        if(lid == 0){
            for (int i = 0; i <= k; i++) {
                tmp[i] = a[id-k+i];

                tmp[k+groupSize+i] = 999999999999999;
            }
        }

    }




    tmp[lid + k] = a[id];

    barrier(CLK_LOCAL_MEM_FENCE);


    for (int j = -k; j <= k; j++) {
        wsum += tmp[(lid + k + j) % groupSize];
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    b[id] = wsum;

}