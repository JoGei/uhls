int32_t mac(int8_t a, int8_t b, int32_t c)
{
    int32_t acc = c;
    int16_t mul = a * b;
    acc = acc + mul;
    return acc; 
}

int32_t dot4_i8_i32_relu_packed(uint32_t Apacked_4xi8, uint32_t Bpacked_4xi8, int32_t C)
{
    int8_t i;
    int32_t sum = C;
    int8_t a;
    int8_t b;
    for (i = 0; i < 4; i = i + 1) {
        a = Apacked_4xi8 >> (i * 8);
        b = Bpacked_4xi8 >> (i * 8);
        sum = mac(a, b, sum);
    }
    i = 0;
    if(sum < 0)
    {
        sum = 0;
    }
    return sum;
}

int32_t test_top(uint32_t lhs, uint32_t rhs, int32_t carry, int32_t expected)
{
    int32_t res;
    res = dot4_i8_i32_relu_packed(lhs, rhs, carry);
    if (res != expected)
    {
        uhls_printf("Unexpected return value, is [%d] expected [%d]", res, expected);
        return 1;
    }
    return 0;
}

int32_t main(void)
{
    int32_t errs = 0;
    bool eflag = false;

    errs = errs + test_top(0x01010101, 0x01010101, 0, 4);
    errs = errs + test_top(( 1 << 24 | 1 << 16 | 1 << 8 | 1), ( 1 << 24 | 1 << 16 | 1 << 8 | 1), 1, 5);
    errs = errs + test_top(0xffffffff, 0xffffffff, 0, 4);
    errs = errs + test_top(0xffffffff, 0x01010101, 0, 0);

    if(errs == 0)
    {
        uhls_printf("Success!");
    }
    return errs;
}