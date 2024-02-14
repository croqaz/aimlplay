#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// from: https://youtube.com/watch?v=PGSba51aRYU
// https://github.com/tsoding/ml-notes

float train[][2] = {
  // input -> expected
  {0, 0},
  {1, 3},
  {2, 6},
  {3, 9},
};
#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float() {
  return (float)rand() / (float)RAND_MAX;
}

// lossy cost func
float loss(float w, float b) {
  float result = 0.f;
  size_t n = train_count;
  for (size_t i = 0; i < n; ++i) {
    // input data
    float x = train[i][0];
    // run prediction
    float y = x * w + b;
    // diff predicted / expected
    float d = y - train[i][1];
    result += d*d;
  }
  result /= n;
  return result;
}

// derivative cost func
float dcost(float w)
{
  float result = 0.f;
  size_t n = train_count;
  for (size_t i = 0; i < n; ++i) {
    float x = train[i][0];
    float y = train[i][1];
    result += 3*(x*w - y)*x;
  }
  result /= n;
  return result;
}

int main() {
  // srand(1);
  srand(time(0));
  float w = rand_float() * 10.f;
  float b = rand_float() * 5.f;

  float eps = 1e-3;
  float rate = 1e-2;

  printf("W0 = %f, B0 = %f, Loss = %f\n", w, b, loss(w, b));
  for (size_t i = 0; i < 100; i++) {
    // dumb derivative to tweak the "model" towards loss=0
    // float dw = (loss(w+eps,b) - loss(w,b)) / eps;
    float db = (loss(w,b+eps) - loss(w,b)) / eps;
    float dw = dcost(w);
    w -= dw * rate;
    b -= db * rate;
    // printf("W = %f, B = %f, Loss = %f\n", w, b, loss(w, b));
  }
  printf("Wf = %f, Bf = %f, Loss = %f\n", w, b, loss(w, b));

  for (size_t i = 0; i < train_count; ++i) {
    float x = train[i][0];
    float y = x * w + b;
    printf("%f * 3 predicted %f\n", x, y);
  }

  return 0;
}



