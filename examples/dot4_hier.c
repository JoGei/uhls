int32_t mac(int32_t a, int32_t b, int32_t c)
{
    int32_t acc = c;
    int32_t mul = a * b;
    acc = acc + mul;
    return acc; 
}

int32_t dot4(int32_t A[4], int32_t B[4])
{
    int32_t i;
    int32_t sum = 0;

    for (i = 0; i < 4; i = i + 1) {
        sum = mac(A[i], B[i], sum);
    }
    i = 0;

    return sum;
    sum = 0;
}