#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// inspired from: https://youtu.be/L1TbWe8bVOc

float rand_float() {
  return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

#define ARRAY_LEN(arr) sizeof((arr))/sizeof((arr)[0])

typedef struct {
  size_t rows;
  size_t cols;
  float *data;
} Matrix;

typedef struct {
  size_t depth;
  Matrix *ws; // weights
  Matrix *bs; // biases
  Matrix *as; // activations
} NN;

#define MAT_AT(m, i, j) m.data[(i)*m.cols + (j)]
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).depth]

void mat_print(Matrix m, const char *name, int padding) {
  printf("%*s%s = [\n", padding, "", name);
  for (size_t i=0; i<m.rows; i++) {
    printf("%*s", padding, "");
    for (size_t j=0; j<m.cols; j++) {
      printf("  %f", MAT_AT(m, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", padding, "");
}

#define MAT_PRINT(nn) mat_print(nn, #nn, 0)

Matrix mat_alloc(size_t rows, size_t cols) {
  Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.data = malloc(sizeof(*m.data) * rows * cols);
  assert(m.data != NULL);
  return m;
}

Matrix mat_row(Matrix m, size_t row) {
  return (Matrix) {
    .rows = 1,
    .cols = m.cols,
    .data = &MAT_AT(m, row, 0),
  };
}

void mat_fill(Matrix m, float x) {
  for (size_t i=0; i<m.rows; i++) {
    for (size_t j=0; j<m.cols; j++) {
      MAT_AT(m, i, j) = x;
    }
  }
}

void mat_rand(Matrix m, float low, float high) {
  for (size_t i=0; i<m.rows; i++) {
    for (size_t j=0; j<m.cols; j++) {
      MAT_AT(m, i, j) = low + rand_float() * (high-low);
    }
  }
}

void mat_sig(Matrix dst) {
  for (size_t i=0; i<dst.rows; i++) {
    for (size_t j=0; j<dst.cols; j++) {
      MAT_AT(dst, i, j) = sigmoidf(MAT_AT(dst,i,j));
    }
  }
}

void mat_copy(Matrix dst, Matrix src) {
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for (size_t i=0; i<dst.rows; i++) {
    for (size_t j=0; j<dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_sum(Matrix dst, Matrix src) {
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for (size_t i=0; i<dst.rows; i++) {
    for (size_t j=0; j<dst.cols; j++) {
      MAT_AT(dst, i, j) += MAT_AT(src, i, j);
    }
  }
}

void mat_dot(Matrix dst, Matrix a, Matrix b) {
  assert(a.cols == b.rows);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  size_t n = a.cols;

  for (size_t i=0; i<dst.rows; i++) {
    for (size_t j=0; j<dst.cols; j++) {
      MAT_AT(dst, i, j) = 0;
      for (size_t k=0; k<n; k++) {
        MAT_AT(dst,i,j) += MAT_AT(a,i,k) * MAT_AT(b,k,j);
      }
    }
  }
}

void nn_print(NN nn, const char *name) {
  printf("%s = {{\n", name);
  char buff[64];
  for (size_t i=0; i<nn.depth; i++) {
    snprintf(buff, sizeof(buff), "ws%zu", i);
    mat_print(nn.ws[i], buff, 2);
    snprintf(buff, sizeof(buff), "bs%zu", i);
    mat_print(nn.bs[i], buff, 2);
  }
  printf("}}\n");
}

#define NN_PRINT(nn) nn_print(nn, #nn)

// Architecture + network depth
NN nn_alloc(size_t *arch, size_t depth) {
  assert(depth > 1);
  NN nn;
  nn.depth = depth - 1;

  nn.ws = malloc(sizeof(*nn.ws) * nn.depth);
  assert(nn.ws != NULL);
  nn.bs = malloc(sizeof(*nn.bs) * nn.depth);
  assert(nn.bs != NULL);
  nn.as = malloc(sizeof(*nn.as) * depth);
  assert(nn.as != NULL);

  // input data
  nn.as[0] = mat_alloc(1, arch[0]);

  for (size_t i=1; i<depth; i++) {
    nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
    nn.bs[i-1] = mat_alloc(1, arch[i]);
    nn.as[i] = mat_alloc(1, arch[i]);
  }

  return nn;
}

void nn_rand(NN nn, float low, float high) {
  for (size_t i=0; i<nn.depth; i++) {
    mat_rand(nn.ws[i], low, high);
    mat_rand(nn.bs[i], low, high);
  }
}

void nn_forward(NN nn) {
  for (size_t i=0; i<nn.depth; i++) {
    mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
    mat_sum(nn.as[i+1], nn.bs[i]);
    mat_sig(nn.as[i+1]);
  }
}

float nn_cost(NN nn, Matrix ti, Matrix to) {
  assert(ti.rows == to.rows);
  assert(to.cols == NN_OUTPUT(nn).cols);
  size_t n = ti.rows;

  float c = 0.f;
  for (size_t i=0; i<n; i++) {
    Matrix x = mat_row(ti, i);
    Matrix y = mat_row(to, i);
    mat_copy(NN_INPUT(nn), x);
    nn_forward(nn);
    size_t q = to.cols;
    for (size_t j=0; j<q; j++) {
      float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
      c += d*d;
    }
  }
  return c/n;
}

void nn_diff(NN nn, NN g, float eps, Matrix ti, Matrix to) {
  float saved;
  float c = nn_cost(nn, ti, to);
  for (size_t i=0; i<nn.depth; i++) {
    for (size_t j=0; j<nn.ws[i].rows; j++) {
      for (size_t k=0; k<nn.ws[i].cols; k++) {
        saved = MAT_AT(nn.ws[i], j, k);
        MAT_AT(nn.ws[i], j, k) += eps;
        MAT_AT(g.ws[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
        MAT_AT(nn.ws[i], j, k) = saved;
      }
    }

    for (size_t j=0; j<nn.bs[i].rows; j++) {
      for (size_t k=0; k<nn.bs[i].cols; k++) {
        saved = MAT_AT(nn.bs[i], j, k);
        MAT_AT(nn.bs[i], j, k) += eps;
        MAT_AT(g.bs[i], j, k) = (nn_cost(nn, ti, to) - c) / eps;
        MAT_AT(nn.bs[i], j, k) = saved;
      }
    }
  }
}

void nn_learn(NN nn, NN gr, float rate) {
  for (size_t i=0; i<nn.depth; i++) {
    for (size_t j=0; j<nn.ws[i].rows; j++) {
      for (size_t k=0; k<nn.ws[i].cols; k++) {
        MAT_AT(nn.ws[i], j, k) -= rate*MAT_AT(gr.ws[i], j, k);
      }
    }
    for (size_t j=0; j<nn.bs[i].rows; j++) {
      for (size_t k=0; k<nn.bs[i].cols; k++) {
        MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(gr.bs[i], j, k);
      }
    }
  }
}



// input
float ti0[] = {
  0, 0,
  1, 0,
  0, 1,
  1, 1,
};

// expected
float to0[] = {
  0, 1, 1, 0,
};


int main() {
  // srand(1);
  srand(time(0));

  Matrix ti = {
    .rows = 4,
    .cols = 2,
    .data = ti0,
  };
  Matrix to = {
    .rows = 4,
    .cols = 1,
    .data = to0,
  };

  float eps = 1e-3;
  float rate = 1e-1;

  // the architecture
  size_t arch[] = { 2, 2, 1 };
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN gr = nn_alloc(arch, ARRAY_LEN(arch));
  nn_rand(nn, 0, 2);

  printf("i-Cost = %f\n", nn_cost(nn, ti, to));
  for (size_t i=0; i<10*1000; i++) {
    nn_diff(nn, gr, eps, ti, to);
    nn_learn(nn, gr, rate);
    // printf("Cost = %f\n", nn_cost(nn, ti, to));
  }
  printf("f-Cost = %f\n", nn_cost(nn, ti, to));

  NN_PRINT(nn);

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      MAT_AT(NN_INPUT(nn), 0, 0) = i;
      MAT_AT(NN_INPUT(nn), 0, 1) = j;
      nn_forward(nn);
      printf("%zu | %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
    }
  }

  return 0;
}
