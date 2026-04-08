#include <stdio.h>
#include <stdlib.h>

#define MAX_BUFFER 256

typedef enum {
    STATUS_IDLE,
    STATUS_ACTIVE,
    STATUS_ERROR
} SystemStatus;

typedef struct {
    int id;
    float score;
} UserRecord;

union DataValue {
    int integer_val;
    float float_val;
};

typedef void (*LogFunction)(int);

void log_to_console(int level) {
    if (level > 0) {
        printf("Level: %d\n", level);
    } 
    else {
        printf("Level is zero or negative\n");
    }
}

int calculate_sum(int n) {
    if (n <= 0) {
        return 0;
    }
    return n + calculate_sum(n - 1);
}

int process_system(SystemStatus status, int *items, int count) {
    int result = 0;

    switch (status) {
        case STATUS_ACTIVE:
            result = 10;
            break;
        case STATUS_ERROR:
            goto cleanup;
        default:
            result = 0;
            break;
    }

    for (int i = 0; i < count; i++) {
        result += items[i];
    }

    while (result < MAX_BUFFER) {
        result = result * 2;
    }

    do {
        if (result > 1000) {
            result = 1000;
            continue;
        }
        break;
    } while (1);

    return result;

cleanup:
    return -1;
}

void handle_dynamic_memory() {
    UserRecord *record = malloc(sizeof(UserRecord));
    
    if (record != NULL) {
        record->id = 1;
        record->score = 95.5f;
        
        int values[3] = {10, 20, 30};
        int *ptr = values;
        ptr++; 

        free(record);
    }
}