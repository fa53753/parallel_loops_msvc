// parallel_loops.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <vector>
//#include <format>
#include <string>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <set>
using namespace std;
namespace fs = filesystem;

static string dirPath = "matrices";

static void checkCompiler()
{
#ifdef _MSC_VER
    std::cout << "Using MSVC, version: " << _MSC_VER << std::endl;
#endif

#ifdef __GNUC__
    std::cout << "Using GCC, version: " << __GNUC__ << "." << __GNUC_MINOR__ << std::endl;
#endif

#ifdef __clang__
    std::cout << "Using Clang" << std::endl;
#endif
}

static string getCurrentTimestamp()
{
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    tm localTime;

#ifdef _WIN32
    localtime_s(&localTime, &t);  // Windows
#else
    localtime_r(&t, &localTime);  // Linux / Unix
#endif

    stringstream ss;
    ss << put_time(&localTime, "%Y%m%d%H%M%S");
    return ss.str();
}

static string getNameForFile(string matrixName, int n, int threads = 0, int multiplicationVariant = 0)
{
    switch (multiplicationVariant)
    {
    case 1:
        //return format("{}\\{}_{}_1_seq.txt", dirPath, matrixName, to_string(n));
        return dirPath + "\\" + matrixName + "_" + to_string(n) + "_1_seq.txt";
    case 2:
        //return format("{}\\{}_{}_{}_par1.txt", dirPath, matrixName, to_string(n), to_string(threads));
        return dirPath + "\\" + matrixName + "_" + to_string(n) + "_" + to_string(threads) + "_par1.txt";
    case 3:
        //return format("{}\\{}_{}_{}_par2.txt", dirPath, matrixName, to_string(n), to_string(threads));
        return dirPath + "\\" + matrixName + "_" + to_string(n) + "_" + to_string(threads) + "_par2.txt";
    case 4:
        //return format("{}\\{}_{}_{}_par3.txt", dirPath, matrixName, to_string(n), to_string(threads));
        return dirPath + "\\" + matrixName + "_" + to_string(n) + "_" + to_string(threads) + "_par3.txt";
    default:
        if (n > 0) return dirPath + "\\" + matrixName + "_" + to_string(n) + ".txt";
        else return dirPath + "\\" + matrixName + ".txt";
    }
}

static void writeMatrixToFile(const string& filename, const int* matrix, int n)
{
    ofstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    file << n << endl;
    for (int i = 0; i < n * n; ++i)
    {
        file << matrix[i];
        if ((i + 1) % n == 0) file << endl;
        else file << ' ';
    }

    file.close();
}

static void generateFileWithMatrix(const string& filename, int n)
{
    srand(time(0));
    int* matrix = new int[n * n];
    for (int i = 0; i < n * n; i++)
    {
        matrix[i] = rand() % 100 + 1;
    }
    writeMatrixToFile(filename, matrix, n);
    cout << "  Generated matrix (" << n << "x" << n << ") in file '" << filename << "'.\n";
}

static int* readMatrixFromFile(const string& filename, int& n)
{
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return nullptr;
    }

    int* matrix = new int[n * n];

    file >> n;
    for (int i = 0; i < n * n; ++i)
    {
        file >> matrix[i];
    }

    file.close();
    return matrix;
}

static void clearMatrix(int* matrix, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        matrix[i] = 0;
    }
}

static void prepareFilesForMatricesAB(const vector<int> matrixSizes)
{
    if (!fs::exists(dirPath)) fs::create_directory(dirPath);

    for (int n : matrixSizes)
    {
        generateFileWithMatrix(getNameForFile("A", n), n);
        generateFileWithMatrix(getNameForFile("B", n), n);
    }
}

static vector<int> joinUnique(const vector<int>& a, const vector<int>& b)
{
    set<int> uniqueSet;
    uniqueSet.insert(a.begin(), a.end());
    uniqueSet.insert(b.begin(), b.end());

    return vector<int>(uniqueSet.begin(), uniqueSet.end());
}

static void prepare(vector<int> matrixSizes, vector<int> matrixSizesSchedule)
{
    cout << "###################" << endl;
    cout << "### Preparation ###" << endl;
    cout << "###################" << endl;
    checkCompiler();
    vector<int> allMatrixSizes = joinUnique(matrixSizes, matrixSizesSchedule);
    prepareFilesForMatricesAB(allMatrixSizes);
    cout << endl << endl;
}

static void multiplySeq(const int* A, const int* B, int* C, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void multiplyParFirstLoop(const int* A, const int* B, int* C, int n, int threads)
{
    int i, j, k, sum;
#pragma omp parallel for num_threads(threads) shared(n, A, B, C) private(i, j, sum, k)
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            sum = 0;
            for (k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void multiplyParSecondLoop(const int* A, const int* B, int* C, int n, int threads)
{
    int j, k, sum;
    for (int i = 0; i < n; i++)
    {
#pragma omp parallel for num_threads(threads) shared(i, n, A, B, C) private(j, sum, k)
        for (j = 0; j < n; j++)
        {
            sum = 0;
            for (k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void multiplyParThirdLoop(const int* A, const int* B, int* C, int n, int threads)
{
    int k;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int sum = 0;
#pragma omp parallel for num_threads(threads) shared(i, j, sum, n, A, B, C) private(k)
            for (k = 0; k < n; k++)
            {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

static void multiply(const int* A, const int* B, int* C, int n, int threads, int variant)
{
    switch (variant)
    {
    case 1:
        multiplySeq(A, B, C, n);
        break;
    case 2:
        multiplyParFirstLoop(A, B, C, n, threads);
        break;
    case 3:
        multiplyParSecondLoop(A, B, C, n, threads);
        break;
    case 4:
        multiplyParThirdLoop(A, B, C, n, threads);
        break;
    default:
        throw invalid_argument("Wrong variant for multiplication!");
    }
}

static chrono::nanoseconds getDurationOfMultiplication(const int* A, const int* B, int* C, int n, int threads, int multiplicationVariant)
{
    clearMatrix(C, n);
    auto start = std::chrono::high_resolution_clock::now();
    multiply(A, B, C, n, threads, multiplicationVariant);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    string filename = getNameForFile("C", n, threads, multiplicationVariant);
    writeMatrixToFile(filename, C, n);
    return duration;
}

static chrono::nanoseconds readMatricesParallel(int*& A, int*& B, int n)
{
    int nA, nB;
    auto start = chrono::high_resolution_clock::now();
#pragma omp parallel sections
    {
#pragma omp section
        {
            A = readMatrixFromFile(getNameForFile("A", n), nA);
        }
#pragma omp section
        {
            B = readMatrixFromFile(getNameForFile("B", n), nB);
        }
    }
    auto end = chrono::high_resolution_clock::now();
    auto durationPar = end - start;

    if (nA != nB || nA != n)
    {
        cerr << "Matrices size mismatch!\n";
    }
    return durationPar;
}

static void compareParallelAndSequentialRead(int*& A, int*& B, int n)
{
    auto start = chrono::high_resolution_clock::now();
    A = readMatrixFromFile(getNameForFile("A", n), n);
    B = readMatrixFromFile(getNameForFile("B", n), n);
    auto end = chrono::high_resolution_clock::now();
    auto durationSeq = end - start;

    auto durationPar = readMatricesParallel(A, B, n);

    cout << "  Sequential reading matrices took " << chrono::duration_cast<chrono::duration<double>>(durationSeq) << ".\n";
    cout << "  Parallel reading matrices took " << chrono::duration_cast<chrono::duration<double>>(durationPar)
        << ".   [" << (durationSeq > durationPar ? "faster" : "slower") << " than sequential (" << (chrono::duration_cast<chrono::duration<double>>(durationSeq - durationPar)) << ")]\n";
}

static void printParMultiplyingTime(chrono::nanoseconds seqTime, chrono::nanoseconds parTime, string parLoop, int threads)
{
    cout << fixed << setprecision(7);
    cout << "  Parallel multiplying matrices (" << parLoop << " loop) with " << threads << " threads took " << chrono::duration_cast<chrono::duration<double>>(parTime) << ".   ["
        << (seqTime > parTime ? "faster" : "slower") << " than sequential (" << (chrono::duration_cast<chrono::duration<double>>(seqTime - parTime)) << ")]\n";
}

static void saveTimes(chrono::nanoseconds durSeq, chrono::nanoseconds* durPar1, chrono::nanoseconds* durPar2, chrono::nanoseconds* durPar3, int n, vector<int> threads)
{
    string filename = getNameForFile("Times", n);
    ofstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    file << fixed << setprecision(7);
    file << "Multiplication times for matrices " << n << "x" << n << endl << endl;
    file << "Type\t| Thr\t| Time [s]\t| Diff [s]\t| Verdict" << endl;
    file << "Seq\t| 1\t| " << chrono::duration_cast<chrono::duration<double>>(durSeq).count() << "\t| -\t\t| -" << endl;
    for (int i = 0; i < threads.size(); ++i)
    {
        file << "ParL1\t| " << threads[i] << "\t| " << chrono::duration_cast<chrono::duration<double>>(durPar1[i]).count()
            << "\t| " << (chrono::duration_cast<chrono::duration<double>>(durSeq - durPar1[i])).count() << "\t| " << (durSeq > durPar1[i] ? "faster" : "slower") << endl;
        file << "ParL2\t| " << threads[i] << "\t| " << chrono::duration_cast<chrono::duration<double>>(durPar2[i]).count()
            << "\t| " << (chrono::duration_cast<chrono::duration<double>>(durSeq - durPar2[i])).count() << "\t| " << (durSeq > durPar2[i] ? "faster" : "slower") << endl;
        file << "ParL3\t| " << threads[i] << "\t| " << chrono::duration_cast<chrono::duration<double>>(durPar3[i]).count()
            << "\t| " << (chrono::duration_cast<chrono::duration<double>>(durSeq - durPar3[i])).count() << "\t| " << (durSeq > durPar3[i] ? "faster" : "slower") << endl;
    }

    file.close();
    cout << "  Times saved in file '" << filename << "'." << endl;
}

static double* compareParallelAndSequentialMultiplication(int*& A, int*& B, int*& C, int n, vector<int> threads)
{
    auto durationSeq = getDurationOfMultiplication(A, B, C, n, 1, 1);

    int tSize = threads.size();
    int tI = 0;
    chrono::nanoseconds* durationPar1 = new chrono::nanoseconds[tSize];
    chrono::nanoseconds* durationPar2 = new chrono::nanoseconds[tSize];
    chrono::nanoseconds* durationPar3 = new chrono::nanoseconds[tSize];
    for (int t : threads)
    {
        durationPar1[tI] = getDurationOfMultiplication(A, B, C, n, t, 2);
        durationPar2[tI] = getDurationOfMultiplication(A, B, C, n, t, 3);
        durationPar3[tI++] = getDurationOfMultiplication(A, B, C, n, t, 4);
    }

    cout << "  Sequential multiplying matrices took " << chrono::duration_cast<chrono::duration<double>>(durationSeq) << ".\n";
    for (int i = 0; i < tSize; ++i)
    {
        printParMultiplyingTime(durationSeq, durationPar1[i], "1st", threads[i]);
        printParMultiplyingTime(durationSeq, durationPar2[i], "2nd", threads[i]);
        printParMultiplyingTime(durationSeq, durationPar3[i], "3rd", threads[i]);
    }

    saveTimes(durationSeq, durationPar1, durationPar2, durationPar3, n, threads);

    double* avg = new double[] { 0, 0, 0};
    for (int i = 0; i < tSize; ++i)
    {
        avg[0] += chrono::duration_cast<chrono::duration<double>>(durationPar1[i]).count();
        avg[1] += chrono::duration_cast<chrono::duration<double>>(durationPar2[i]).count();
        avg[2] += chrono::duration_cast<chrono::duration<double>>(durationPar3[i]).count();
    }
    avg[0] /= tSize;
    avg[1] /= tSize;
    avg[2] /= tSize;

    delete[] durationPar1;
    delete[] durationPar2;
    delete[] durationPar3;
    return avg;
}

static string FindBest(const double* avg, int n)
{
    cout << "  Averages:: ";
    for (int i = 0; i < n; ++i)
    {
        cout << "Loop #" << (i + 1) << ": " << avg[i] << "s; ";
    }
    cout << endl;
    double min = avg[0];
    string minTxt = "1st loop";
    if (avg[1] < min)
    {
        min = avg[1];
        minTxt = "2nd loop";
    }
    if (avg[2] < min)
    {
        min = avg[2];
        minTxt = "3rd loop";
    }
    return minTxt;
}

static void SaveAverages(double* avg, int avgCnt, int n, string fileName)
{
    string filename = getNameForFile(fileName, n);
    ofstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    file << fixed << setprecision(7);
    if (n > 0) file << "Averages for matrices " << n << "x" << n << ":" << endl;
    else file << "Total averages" << endl;
    file << "Loop\t| Time" << endl;
    for (int i = 0; i < avgCnt; ++i)
    {
        file << "#" << (i+1) << "\t| " << avg[i] << endl;
    }

    file.close();
}

static string analyzeSequentialVsParallel(vector<int> matrixSizes, vector<int> threads)
{
    double* totalAvg = new double[] { 0, 0, 0 };
    for (int n : matrixSizes)
    {
        int* A = nullptr;
        int* B = nullptr;
        int* C = new int[n * n];
        double* avg = nullptr;

        cout << "###################" << endl;
        cout << "###\t" << n << "\t###" << endl;
        cout << "###################" << endl;

        cout << "### Comparing sequential and parallel reading of matrices " << n << "x" << n << " ###\n";
        compareParallelAndSequentialRead(A, B, n);
        cout << endl;

        cout << "### Comparing sequential and parallel multiplication of matrices " << n << "x" << n << " ###\n";
        avg = compareParallelAndSequentialMultiplication(A, B, C, n, threads);
        string best = FindBest(avg, 3);
        cout << "Best for " << to_string(n) << "x" << to_string(n) << ": " << best << endl << endl;
        SaveAverages(avg, 3, n, "Averages");
        totalAvg[0] += avg[0];
        totalAvg[1] += avg[1];
        totalAvg[2] += avg[2];

        delete[] avg;
        delete[] A;
        delete[] B;
        delete[] C;
    }
    totalAvg[0] /= matrixSizes.size();
    totalAvg[1] /= matrixSizes.size();
    totalAvg[2] /= matrixSizes.size();
    string verdict = FindBest(totalAvg, 3);
    SaveAverages(totalAvg, 3, 0, "TotalAverages");
    cout << "Best overall: " << verdict << endl << endl;

    delete[] totalAvg;
    return verdict;
}

enum class ScheduleType
{
    Static,
    Dynamic,
    Guided
};

static string getString(ScheduleType type)
{
    switch (type)
    {
    case ScheduleType::Static: return "static";
    case ScheduleType::Dynamic: return "dynamic";
    case ScheduleType::Guided: return "guided";
    default: return "unknown";
    }
}

static void saveTimesSchedule(chrono::nanoseconds* times, int n, vector<int> threads, vector<ScheduleType> scheduleTypes, int chunk, string loop)
{
    string filename = getNameForFile("TimesSchedule", n);
    ofstream file(filename);

    if (!file.is_open())
    {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    int timesI = 0;
    file << fixed << setprecision(7);
    file << "Multiplication times for matrices " << n << "x" << n << "parallel on " << loop << endl << endl;
    file << "Type\t| Chunk\t| Thr\t| Time [s]" << endl;
    for (int t : threads)
    {
        for (auto scheduleType : scheduleTypes)
        {
            file << getString(scheduleType) << "\t| " << chunk << "\t| " << t << "\t| " << chrono::duration_cast<chrono::duration<double>>(times[timesI++]).count() << endl;
        }
    }

    file.close();
    cout << "  Times saved in file '" << filename << "'." << endl;
}

static chrono::nanoseconds multiplyParFirstLoopSchedule(const int* A, const int* B, int* C, int n, int threads, ScheduleType scheduleType, int chunk)
{ // MSVC nie wspiera omp_set_schedule() i schedule(runtime), wiec niestety trzeba powieliæ
    /*switch (scheduleType)
    {
        case ScheduleType::Guided:
            omp_set_schedule(omp_sched_t::omp_sched_guided, chunk);
            break;
        case ScheduleType::Dynamic:
            omp_set_schedule(omp_sched_t::omp_sched_dynamic, chunk);
            break;
        case ScheduleType::Static:
        default:
            omp_set_schedule(omp_sched_t::omp_sched_static, chunk);
            break;
    }*/

    chrono::steady_clock::time_point start, end;
    switch (scheduleType)
    {
    case ScheduleType::Guided:
        start = chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(threads) schedule(guided) shared(n, A, B, C) private(i, j, sum, k)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    case ScheduleType::Dynamic:
        start = chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(threads) schedule(dynamic) shared(n, A, B, C) private(i, j, sum, k)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    case ScheduleType::Static:
    default:
        start = chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(threads) schedule(static) shared(n, A, B, C) private(i, j, sum, k)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    }

    cout << "  Parallel (1st loop) multiplying matrices " << to_string(n) << "x" << to_string(n) << " with schedule(" << getString(scheduleType) << ", " << chunk << ") on " << to_string(threads) << " threads took "
        << chrono::duration_cast<chrono::duration<double>>(end - start) << endl;
    return end - start;
}

static chrono::nanoseconds multiplyParSecondLoopSchedule(const int* A, const int* B, int* C, int n, int threads, ScheduleType scheduleType, int chunk)
{
    chrono::steady_clock::time_point start, end;
    switch (scheduleType)
    {
    case ScheduleType::Guided:
        start = chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
        {
#pragma omp parallel for num_threads(threads) schedule(guided) shared(n, A, B, C, i) private(j, sum, k)
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    case ScheduleType::Dynamic:
        start = chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
        {
#pragma omp parallel for num_threads(threads) schedule(dynamic) shared(n, A, B, C, i) private(j, sum, k)
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    case ScheduleType::Static:
    default:
        start = chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
        {
#pragma omp parallel for num_threads(threads) schedule(static) shared(n, A, B, C, i) private(j, sum, k)
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    }

    cout << "  Parallel (2nd loop) multiplying matrices " << to_string(n) << "x" << to_string(n) << " with schedule(" << getString(scheduleType) << ", " << chunk << ") on " << to_string(threads) << " threads took "
        << chrono::duration_cast<chrono::duration<double>>(end - start) << endl;
    return end - start;
}

static chrono::nanoseconds multiplyParThirdLoopSchedule(const int* A, const int* B, int* C, int n, int threads, ScheduleType scheduleType, int chunk)
{
    chrono::steady_clock::time_point start, end;
    switch (scheduleType)
    {
    case ScheduleType::Guided:
        start = chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
#pragma omp parallel for num_threads(threads) schedule(guided) shared(n, A, B, C, i, j, sum) private(k)
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    case ScheduleType::Dynamic:
        start = chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
#pragma omp parallel for num_threads(threads) schedule(guided) shared(n, A, B, C, i, j, sum) private(k)
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    case ScheduleType::Static:
    default:
        start = chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int sum = 0;
#pragma omp parallel for num_threads(threads) schedule(guided) shared(n, A, B, C, i, j, sum) private(k)
                for (int k = 0; k < n; k++)
                {
                    sum += A[i * n + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        end = chrono::high_resolution_clock::now();
        break;
    }

    cout << "  Parallel (3rd loop) multiplying matrices " << to_string(n) << "x" << to_string(n) << " with schedule(" << getString(scheduleType) << ", " << chunk << ") on " << to_string(threads) << " threads took "
        << chrono::duration_cast<chrono::duration<double>>(end - start) << endl;
    return end - start;
}

static chrono::nanoseconds multiplySchedule(string parallelType, const int* A, const int* B, int* C, int n, int threads, ScheduleType scheduleType, int chunk)
{
    if (parallelType == "1st loop")
    {
        return multiplyParFirstLoopSchedule(A, B, C, n, threads, scheduleType, chunk);
    }
    else if (parallelType == "2nd loop")
    {
        return multiplyParSecondLoopSchedule(A, B, C, n, threads, scheduleType, chunk);
    }
    else if (parallelType == "3rd loop")
    {
        return multiplyParThirdLoopSchedule(A, B, C, n, threads, scheduleType, chunk);
    }
    throw invalid_argument("Wrong parallelType for multiplication!");
}

static void analyzeSchedule(vector<int> matrixSizes, vector<int> threads, vector<ScheduleType> scheduleTypes, int chunk, string parallelType)
{
    cout << "###################" << endl;
    cout << "###  Schedules  ###" << endl;
    cout << "###################" << endl;

    for (int n : matrixSizes)
    {
        int* A = nullptr;
        int* B = nullptr;
        int* C = new int[n * n];

        readMatricesParallel(A, B, n);

        chrono::nanoseconds* times = new chrono::nanoseconds[threads.size() * scheduleTypes.size()];
        int currentTimeI = 0;
        for (int t : threads)
        {
            for (auto scheduleType : scheduleTypes)
            {
                times[currentTimeI++] = multiplySchedule(parallelType, A, B, C, n, t, scheduleType, chunk);
            }
        }
        saveTimesSchedule(times, n, threads, scheduleTypes, chunk, parallelType);

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] times;
    }
}

int main()
{
    // ####################################### Config
    dirPath += getCurrentTimestamp();

    //vector<int> matrixSizes = { 100, 500 };
    vector<int> matrixSizes = { 100, 500, 1000, 2000 };
    vector<int> threads = { 2, 4, 8, 16 };

    vector<int> matrixSizesSchedule = { 1000, 2000 };
    vector<int> threadsSchedule = { 2, 4, 8, 16 };
    vector<ScheduleType> scheduleTypes = { ScheduleType::Static, ScheduleType::Dynamic, ScheduleType::Guided };
    int chunkSchedule = 50;

    // ####################################### Program

    prepare(matrixSizes, matrixSizesSchedule);

    string verdict = analyzeSequentialVsParallel(matrixSizes, threads);

    analyzeSchedule(matrixSizesSchedule, threadsSchedule, scheduleTypes, chunkSchedule, verdict);
}
