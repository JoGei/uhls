int32_t dot4(int32_t A[4], int32_t B[4]) {
    int32_t i;
    int32_t sum = 0;

    for (i = 0; i < 4; i = i + 1) {
        sum = sum + A[i] * B[i];
    }

    return sum;
}

int32_t main(void)
{
    int32_t A[4] = {1,2,3,4};
    int32_t B[4] = {4,3,2,1};

    int32_t expected = 20;
    int32_t res;
    res = dot4(A, B);
    
    if(res != expected)
    {
        uhls_printf("Unexpected return value, is [%d] expected [%d]", res, expected);
        return 1;
    }
    uhls_printf("Success!");
    return 0;
}
