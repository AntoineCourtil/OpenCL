
__kernel void wsumsV1(__global const int *a, __global int *b, int n, int k) {

    int id = get_global_id(0);

    int wsum = 0;

    for (int j = -k; j <= k; j++) {
        wsum += a[(id + j) % n];
    }

    b[id] = wsum;

}