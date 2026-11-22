#include <stdint.h>
#include <string.h>
// Euler step for the dynamics specified in model/flyvis.json.
// CSR columns are grouped by presynaptic neuron.
void step(int n, const int32_t *ptr, const int32_t *dst, const float *w,
          const float *bias, const float *gain, const float *input,
          float *voltage, float *acc) {
    memcpy(acc, input, n * sizeof(float));
    for (int i = 0; i < n; i++) {
        float rate = voltage[i];
        if (rate > 0)
            for (int j = ptr[i]; j < ptr[i + 1]; j++)
                acc[dst[j]] += rate * w[j];
    }
    for (int i = 0; i < n; i++)
        voltage[i] += gain[i] * (bias[i] + acc[i] - voltage[i]);
}
