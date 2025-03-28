#include <iostream>
#include <windows.h>
#include <vector>
#include<stdlib.h>
using namespace std;

int main()
{
    long long head, tail, freq;
    double times = 100;

    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    for (int n = 0; n <= 10000; n += 500)
    {
        if (n == 0)
        {
            cout << "0" << endl;
            continue;
        }

        vector<vector<int>> b(n, vector<int>(n));

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                b[i][j] = i + j;
            }
        }

        vector<int> a(n);
        for (int i = 0; i < n; i++)
        {
            a[i] = i;
        }

        vector<int> sum(n);

        long long total_time = 0;

        for (int num = 0; num < times; num++)
        {
            QueryPerformanceCounter((LARGE_INTEGER *)&head);

            for (int i = 0; i < n; i++)
            {
                sum[i] = 0;
                for (int j = 0; j < n; j++)
                {
                    sum[i] += b[j][i] * a[j];
                }
            }

            QueryPerformanceCounter((LARGE_INTEGER *)&tail);
            total_time += (tail - head);
        }

        cout << (total_time) / times * 1000.0 / freq << endl;
    }

    return 0;
}