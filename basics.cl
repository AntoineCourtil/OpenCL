
__kernel void odd(__global int *table, int n) {

    int id1 = get_global_id(0) * 2 + 1;
    int id2 = id1 + 1;

    int swap;

    if(table[id2] > table[id1]){
        swap = table[id2];
        table[id2] = table[id1];
        table[id1] = swap;
    }
}


__kernel void even(__global int *table, int n) {

    int id1 = get_global_id(0) * 2;
    int id2 = id1 + 1;

    int swap;

    if(table[id2] > table[id1]){
        swap = table[id2];
        table[id2] = table[id1];
        table[id1] = swap;
    }
}
