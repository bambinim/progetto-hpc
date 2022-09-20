/*
* Matteo Bambini - 0000916140
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for ceil() */
#include <assert.h>
#include <omp.h>
#include "hpc.h"

typedef enum {
    WALL,
    GAS,
    EMPTY
} cell_value_t;

typedef enum {
    ODD_PHASE = -1,
    EVEN_PHASE = 1
} phase_t;

/* type of a cell of the domain */
typedef unsigned char cell_t;

/* Simplifies indexing on a N*N grid */
int IDX(int i, int j, int N)
{
    /* wrap-around */
    i = (i+N) % N;
    j = (j+N) % N;
    return i*N + j;
}

/* Swap the content of cells a and b, provided that neither is a WALL;
   otherwise, do nothing. */
void swap_cells(cell_t *a, cell_t *b)
{
    if ((*a != WALL) && (*b != WALL)) {
        const cell_t tmp = *a;
        *a = *b;
        *b = tmp;
    }
}

/* Compute the `next` grid given the `cur`-rent configuration. */
void step( const cell_t *cur, cell_t *next, int N, phase_t phase )
{
    int i, j;

    assert(cur != NULL);
    assert(next != NULL);

    /* Loop over all coordinates (i,j) s.t. both i and j are even */
#pragma omp parallel for
    for (i=0; i<N; i+=2) {
        for (j=0; j<N; j+=2) {
            /**
             * If phase==EVEN_PHASE:
             * ab
             * cd
             *
             * If phase==ODD_PHASE:
             * dc
             * ba
             */
            const int a = IDX(i      , j      , N);
            const int b = IDX(i      , j+phase, N);
            const int c = IDX(i+phase, j      , N);
            const int d = IDX(i+phase, j+phase, N);
            next[a] = cur[a];
            next[b] = cur[b];
            next[c] = cur[c];
            next[d] = cur[d];
            if ((((next[a] == EMPTY) != (next[b] == EMPTY)) &&
                 ((next[c] == EMPTY) != (next[d] == EMPTY))) ||
                (next[a] == WALL) || (next[b] == WALL) ||
                (next[c] == WALL) || (next[d] == WALL)) {
                swap_cells(&next[a], &next[b]);
                swap_cells(&next[c], &next[d]);
            } else {
                swap_cells(&next[a], &next[d]);
                swap_cells(&next[b], &next[c]);
            }
        }
    }
}

/**
 ** The functions below are used to draw onto the grid; since they are
 ** called during initialization only, they do not need to be
 ** parallelized.
 **/
void box( cell_t *grid, int N, float x1, float y1, float x2, float y2, cell_value_t t )
{
    const int ix1 = ceil(fminf(x1, x2) * N);
    const int ix2 = ceil(fmaxf(x1, x2) * N);
    const int iy1 = ceil(fminf(y1, y1) * N);
    const int iy2 = ceil(fmaxf(y1, y2) * N);
    int i, j;
    for (i = iy1; i <= iy2; i++) {
        for (j = ix1; j <= ix2; j++) {
            const int ij = IDX(N-1-i, j, N);
            grid[ij] = t;
        }
    }
}

void circle( cell_t *grid, int N, float x, float y, float r, cell_value_t t )
{
    const int ix = ceil(x * N);
    const int iy = ceil(y * N);
    const int ir = ceil(r * N);
    int dx, dy;
    for (dy = -ir; dy <= ir; dy++) {
        for (dx = -ir; dx <= ir; dx++) {
            if (dx*dx + dy*dy <= ir*ir) {
                const int ij = IDX(N-1-iy-dy, ix+dx, N);
                grid[ij] = t;
            }
        }
    }
}

void random_fill( cell_t *grid, int N, float x1, float y1, float x2, float y2, float p )
{
    const int ix1 = ceil(fminf(x1, x2) * N);
    const int ix2 = ceil(fmaxf(x1, x2) * N);
    const int iy1 = ceil(fminf(y1, y1) * N);
    const int iy2 = ceil(fmaxf(y1, y2) * N);
    int i, j;
    for (i = iy1; i <= iy2; i++) {
        for (j = ix1; j <= ix2; j++) {
            const int ij = IDX(N-1-i, j, N);
            if (grid[ij] == EMPTY && ((float)rand())/RAND_MAX < p)
                grid[ij] = GAS;
        }
    }
}

void read_problem( FILE *filein, cell_t *grid, int N )
{
    int i,j;
    int nread;
    char op;

    for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
            const int ij = IDX(i,j,N);
            grid[ij] = EMPTY;
        }
    }

    while ((nread = fscanf(filein, " %c", &op)) == 1) {
        int t;
        float x1, y1, x2, y2, r, p;
        int retval;

        switch (op) {
        case 'c' : /* circle */
            retval = fscanf(filein, "%f %f %f %d", &x1, &y1, &r, &t);
            assert(retval == 4);
            circle(grid, N, x1, y1, r, t);
            break;
        case 'b': /* box */
            retval = fscanf(filein, "%f %f %f %f %d", &x1, &y1, &x2, &y2, &t);
            assert(retval == 5);
            box(grid, N, x1, y1, x2, y2, t);
            break;
        case 'r': /* random_fill */
            retval = fscanf(filein, "%f %f %f %f %f", &x1, &y1, &x2, &y2, &p);
            assert(retval == 5);
            random_fill(grid, N, x1, y1, x2, y2, p);
            break;
        default:
            fprintf(stderr, "FATAL: Unrecognized command `%c`\n", op);
            exit(EXIT_FAILURE);
        }
    }
}


/* Write an image of `grid` to a file in PGM (Portable Graymap)
   format. `frameno` is the time step number, used for labeling the
   output file. */
void write_image( const cell_t *grid, int N, int frameno )
{
    FILE *f;
    char fname[128];

    snprintf(fname, sizeof(fname), "hpp%05d.pgm", frameno);
    if ((f = fopen(fname, "w")) == NULL) {
        printf("Cannot open \"%s\" for writing\n", fname);
        abort();
    }
    fprintf(f, "P5\n");
    fprintf(f, "# produced by hpp\n");
    fprintf(f, "%d %d\n", N, N);
    fprintf(f, "%d\n", EMPTY); /* highest shade of grey (0=black) */
    fwrite(grid, 1, N*N, f);
    fclose(f);
}

int main( int argc, char* argv[] )
{
    int t, N, nsteps;
    FILE *filein;

    srand(1234); /* Initialize PRNG deterministically */

    if ( (argc < 2) || (argc > 4) ) {
        fprintf(stderr, "Usage: %s [N [S]] input\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (argc > 2) {
        N = atoi(argv[1]);
    } else {
        N = 512;
    }

    if (argc > 3) {
        nsteps = atoi(argv[2]);
    } else {
        nsteps = 32;
    }

    if (N % 2 != 0) {
        fprintf(stderr, "FATAL: the domain size N must be even\n");
        return EXIT_FAILURE;
    }

    if ((filein = fopen(argv[argc-1], "r")) == NULL) {
        fprintf(stderr, "FATAL: can not open \"%s\" for reading\n", argv[argc-1]);
        return EXIT_FAILURE;
    }

    const size_t GRID_SIZE = N*N*sizeof(cell_t);
    cell_t *cur = (cell_t*)malloc(GRID_SIZE);
    assert(cur != NULL);
    cell_t *next = (cell_t*)malloc(GRID_SIZE);
    assert(next != NULL);

    read_problem(filein, cur, N);

    double start_time = hpc_gettime();
    for (t=0; t<nsteps; t++) {
#ifdef DUMP_ALL
        write_image(cur, N, t);
#endif
        step(cur, next, N, EVEN_PHASE);
        step(next, cur, N, ODD_PHASE);
    }
    double finish_time = hpc_gettime();
    printf("Execution time: %fs\n", finish_time - start_time);
#ifdef DUMP_ALL
    /* Reverse all particles and go back to the initial state */
    for (; t<2*nsteps; t++) {
        write_image(cur, N, t);
        step(cur, next, N, ODD_PHASE);
        step(next, cur, N, EVEN_PHASE);
    }
#endif
    write_image(cur, N, t);
    free(cur);
    free(next);
    fclose(filein);
    return EXIT_SUCCESS;
}
