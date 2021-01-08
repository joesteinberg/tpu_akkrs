CC = gcc
CFLAGS = -Wall -pedantic -mcmodel=medium -ffast-math -DHAVE_INLINE=1 -g -fopenmp -O3

LDFLAGS = -fopenmp -lgsl -lgslcblas -lm -ldl -lgomp -g -O3

EXDIR = bin
ODIR = src/obj

_OBJ = model.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: src/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

model: src/obj/model.o
	$(CC) -o $(EXDIR)/$@ $^ $(CFLAGS) $(LDFLAGS)
