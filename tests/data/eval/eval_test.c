#include <stdio.h>
#include <stdlib.h>

#define MAX_SENSORS 64

typedef enum {
    TYPE_TEMPERATURE,
    TYPE_HUMIDITY,
    TYPE_PRESSURE
} SensorType;

typedef struct {
    unsigned int device_id;
    double timestamp;
} Metadata;

union ValueStore {
    int raw_binary;
    float scaled_decimal;
};

typedef void (*AlertCallback)(float);

void trigger_alarm(float value) {
    if (value > 90.0f) {
        printf("Critical High: %.2f\n", value);
    } 
    else {
        printf("Normal Range: %.2f\n", value);
    }
}

int compute_factorial(int val) {
    if (val <= 1) {
        return 1;
    }
    return val * compute_factorial(val - 1);
}

int analyze_signals(SensorType type, float *readings, int length) {
    int status_code = 0;

    switch (type) {
        case TYPE_TEMPERATURE:
            status_code = 200;
            break;
        case TYPE_PRESSURE:
            goto hardware_fault;
        default:
            status_code = 404;
            break;
    }

    for (int j = 0; j < length; j++) {
        status_code += (int)readings[j];
    }

    while (status_code < MAX_SENSORS) {
        status_code += 5;
    }

    do {
        if (status_code % 2 != 0) {
            status_code++;
            continue;
        }
        break;
    } while (1);

    return status_code;

hardware_fault:
    return -99;
}

void allocate_sensor_memory() {
    Metadata *info = malloc(sizeof(Metadata));
    
    if (info != NULL) {
        info->device_id = 101;
        info->timestamp = 1711968000.0;
        
        float samples[2] = {23.5f, 24.1f};
        float *current = samples;
        current++;

        free(info);
    }
}