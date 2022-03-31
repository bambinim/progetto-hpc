## Progetto di High Performance Computing 2021/2022, corso di laurea
## in Ingegneria e Scienze Informatiche, Universita' di Bologna.

## Ultima modifica: 2021-11-20, Moreno Marzolla <moreno.marzolla(at)unibo.it>

## Questo e' un frammento del Makefile utilizzato negli esempi
## illustrati durante il corso. Questo file puo' essere modificato e
## personalizzato in base alle proprie esigenze. Si puo' anche
## decidere di non usarlo; in tal caso indicare le istruzioni di
## compilazione nel file README presente nella directory a livello
## superiore.
##
## Se si decide di usare questo makefile, il comando "make" dovrebbe
## compilare tutti i programmi consegnati senza errori né warning.  Si
## consiglia pertanto di rimuovere eventuali target non presenti
## nell'archivio consegnato.
##
## Questo Makefile compila i file "omp-*.c" usando il flag -fopenmp, i
## file "cuda-*.cu" con il compilatore nvcc, e i file "mpi-*.c" con
## mpicc.
##
## I principali target definiti da questo makefile sono:
##
## make         compila tutti i sorgenti disponibili
## make clean   cancella i file temporanei e gli eseguibili
## make openmp  compila la versione OpenMP
## make mpi     compila la versione MPI
## make cuda    compila la versione CUDA

EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_CUDA:=$(basename $(wildcard cuda-*.cu))
DATAFILES:=
EXE_SERIAL:=hpp
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_CUDA)
CFLAGS+=-std=c99 -Wall -Wpedantic -O2 -g
LDLIBS+=-lm
NVCC:=nvcc
NVCFLAGS+=-g -G -gencode arch=compute_37,code=sm_37 -Wno-deprecated-gpu-targets
NVLDLIBS+=-lm

.PHONY: clean

ALL: $(EXE)

movie:
	$(CC) $(CFLAGS) -DDUMP_ALL hpp.c -o hpp-movie $(LDLIBS)
	./hpp-movie 256 256 cannon.in
	ffmpeg -y -i "hpp%05d.pgm" -vcodec mpeg4 movie.avi

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@ $(NVLDLIBS)

$(EXE_OMP): CFLAGS+=-fopenmp
openmp: $(EXE_OMP)

$(EXE_MPI): CC=mpicc
mpi: $(EXE_MPI)

cuda: $(EXE_CUDA)

clean:
	\rm -f $(EXE) hpp-movie *.o *~ *.pbm *.pgm *.avi
