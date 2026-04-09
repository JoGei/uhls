int32_t mac(int32_t a, int32_t b, int32_t c)
{
    int32_t acc = c;
    int32_t mul = a * b;
    acc = acc + mul;
    return acc; 
}

int32_t dot4_relu(int32_t A[4], int32_t B[4])
{
    int32_t i;
    int32_t sum = 0;

    for (i = 0; i < 4; i = i + 1) {
        sum = mac(A[i], B[i], sum);
    }
    i = 0;
    if(sum < 0)
    {
        sum = 0;
    }
    return sum;
}

int32_t main(void)
{
    int32_t errs = 0;
    bool eflag = false;
    int32_t A[4] = {1,2,3,4};
    int32_t B[4] = {4,3,2,1};

    int32_t expected = 20;
    int32_t res;
    res = dot4_relu(A, B);
    eflag = (res != expected);
    if(eflag)
    {
        uhls_printf("Unexpected return value, is [%d] expected [%d]", res, expected);
        eflag = false;
        errs = errs + 1;
    }

    expected = 4;
    int32_t A1[4] = {1,1,1,1};
    int32_t B1[4] = {1,1,1,1};
    res = dot4_relu(A1, B1);
    eflag = (res != expected);
    if(eflag)
    {
        uhls_printf("Unexpected return value, is [%d] expected [%d]", res, expected);
        eflag = false;
        errs = errs + 1;
    }

    if(errs == 0)
    {
        uhls_printf("Success!");
    }
    return errs;
}
