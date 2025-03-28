#include <iostream>
#include <windows.h>
#include <vector>
#include <stdlib.h>
using namespace std;

int main()
{
    long long head, tail, freq;
    int n = 114514;
    long long sum = 0;
    int times = 100;

    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    for (; n > 0 && n <= 59000000; n *= 2)
    {
        std::vector<int> a(n);
        for (int i = 0; i < n; i++)
        {
            a[i] = i;
        }

        long long total_time = 0;

        for (int num = 0; num < times; num++)
        {
            QueryPerformanceCounter((LARGE_INTEGER *)&head);

            long long sum0 = 0;
            long long sum1 = 0;
            int i = 0;

            for (; i < n - 1; i += 2)
            {
                sum0 += a[i];
                sum1 += a[i + 1];
            }

            if (i < n)
            {
                sum0 += a[i];
            }

            sum += sum0 + sum1;

            QueryPerformanceCounter((LARGE_INTEGER *)&tail);
            total_time += (tail - head);
        }

        cout << (total_time) / (double)times * 1000.0 / freq << endl;
    }

    return 0;
}