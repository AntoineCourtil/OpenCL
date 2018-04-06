
/**
 *
 * Casse la coalescence
 *
 */
int F(int id, int n){
    const int SZ = 1024;

    int bloc = id % SZ;
    int cell = id / SZ;

    return bloc * SZ + cell;

    const int N = n;
    const int S = 2; //décalage du prochain voisin
    const int B = N / S;

    int bloc2 = id % S;
    int cell2 = id / S;
    return (bloc2 * B) + cell2;

}


__kernel void addVector(__global int *a, __global int *b, __global int *c, int n){

    int id = get_global_id(0);

    int k = F(id, n);

    c[k] = a[k] + b[k];

}

__kernel void sumOfAll(__global int *d){
//    d[0] ++;
    atomic_inc(d);
}