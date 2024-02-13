#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// from: https://youtube.com/watch?v=PGSba51aRYU

// OR
float train[][3] = {
  // input -> expected
  {0, 0, 0},
  {1, 0, 1},
  {0, 1, 1},
  {1, 1, 1},
};

#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float() {
  return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x) {
  return 1.f / (1.f + expf(-x));
}

float forward(float w1, float w2, float b, int x1, int x2) {
  float result = x1 * w1 + x2 * w2 + b;
  return sigmoidf(result);
}

float loss(float w1, float w2, float b) {
  float result = 0.f;
  for (size_t i=0; i<train_count; i++) {
    // input data
    float x1 = train[i][0];
    float x2 = train[i][1];
    // predicted
    float y = forward(w1, w2, b, x1, x2);
    // diff predicted / expected
    float e = y - train[i][2];
    result += e*e;
  }
  return result;
}

int main() {
  // srand(1);
  srand(time(0));
  float w1 = rand_float() * 10.f;
  float w2 = rand_float() * 10.f;
  float b = rand_float() * 5.f;

  float eps = 1e-3;
  float rate = 1e-1;

  printf("W1=%f, W2=%f, B0=%f, Loss0=%f\n", w1, w2, b, loss(w1,w2,b));

  for (size_t i=0; i<10000; i++) {
    // dumb derivative to tweak the "model" towards loss=0
    float c = loss(w1, w2, b);
    float dw1 = (loss(w1+eps, w2, b) - c) / eps;
    float dw2 = (loss(w1, w2+eps, b) - c) / eps;
    float db = (loss(w1, w2, b+eps) - c) / eps;
    w1 -= dw1 * rate;
    w2 -= dw2 * rate;
    b -= db * rate;
    // printf("W1=%f, W2=%f, B=%f, Loss=%f\n", w1, w2, b, loss(w1,w2,b));
  }

  printf("----------------------------------------\n");
  printf("W1=%f, W2=%f, B=%f, Loss=%f\n", w1, w2, b, loss(w1,w2,b));

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      printf("%zu | %zu = %f\n", i, j, forward(w1, w2, b, i, j));
    }
  }

  return 0;
}



