
__kernel void odd(__global int *table, int n) {

    int id1 = get_global_id(0) * 2 + 1;
    int id2 = id1 + 1;

    int swap;

    if (table[id2] > table[id1]) {
        swap = table[id2];
        table[id2] = table[id1];
        table[id1] = swap;
    }
}


__kernel void even(__global int *table, int n) {

    int id1 = get_global_id(0) * 2;
    int id2 = id1 + 1;

    int swap;

    if (table[id2] > table[id1]) {
        swap = table[id2];
        table[id2] = table[id1];
        table[id1] = swap;
    }
}


__kernel void bitonic(__global int *table, int stage, int col) {

    int gid = get_global_id(0);

    int spacing = 1 << col;
    int bucket = gid / spacing;
    int bucketID = gid % spacing;

    int a = bucket * spacing * 2 + bucketID;
    int b = a + spacing;
    int swap = 0;

    int dir = (gid / (1 << stage)) % 2;

    if(dir){
        swap = a;
        a = b;
        b = swap;
    }

    if (table[a] < table[b]) {
        swap = table[a];
        table[a] = table[b];
        table[b] = swap;
    }

}
