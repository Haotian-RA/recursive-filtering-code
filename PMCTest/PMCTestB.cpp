//                       PMCTestB.cpp                    2014-04-15 Agner Fog
//
//          Multithread PMC Test program for Windows and Linux
//
// This program is intended for testing the performance of a little piece of 
// code written in C or C++. The code to test is inserted at the place marked
// "Test code start".
// All sections that can be modified by the user are marked with ###########. 
//
// The code to test will be executed REPETITIONS times and the test results
// will be output for each repetition. This program measures how many clock
// cycles the code to test takes in each repetition. Furthermore, it is 
// possible to set a number of Performance Monitor Counters (PMC) to count 
// the number of micro-operations (uops), cache misses, branch mispredictions,
// etc.
// 
// The setup of the Performance Monitor Counters is microprocessor-specific.
// The specifications for PMC setup for each microprocessor family is defined
// in the tables CounterDefinitions and CounterTypesDesired.
//
// See PMCTest.txt for instructions.
//
// � 2000-2014 GNU General Public License www.gnu.org/licences
//////////////////////////////////////////////////////////////////////////////

#include "PMCTest.h"

#include "../include/zero_init_condition.h"
#include "../include/permuteV.h"
#include "../include/init_cond_correction.h"
#include "../include/iir_cores.h"

#include <random>


/*############################################################################
#
#        Define constants
#
############################################################################*/

// number of repetitions of test. You may change this up to MAXREPEAT
#define REPETITIONS  20

// Number of threads
#define NUM_THREADS  1

// Use performance monitor counters. Set to 0 if not used
#define USE_PERFORMANCE_COUNTERS  1

// Subtract overhead from counts (0 if not)
#define SUBTRACT_OVERHEAD 1

// Number of repetitions in loop to find overhead
#define OVERHEAD_REPETITIONS  5

// Cache line size (for preventing threads using same cache lines)
#define CACHELINESIZE  64


/*############################################################################
#
#        list of desired counter types
#
############################################################################*/
// 
// Here you can select which performance monitor counters you want for your test.
// Select id numbers from the table CounterDefinitions[] in PMCTestA.cpp.
// The maximum number of counters you can have is MAXCOUNTERS.
// Insert zeroes if you have less than MAXCOUNTERS counters.

extern "C" {
    int CounterTypesDesired[MAXCOUNTERS] = { // from the skylake microarchitecture
      1,      // core clock cycles (Intel Core 2 and later)
      9,      // instructions (not P4)
      100,    // micro-operations
      150,    // uops port 0: FP FMA
      151,    // uops port 1: FP FMA
      155,    // uops port 5: Vect Shuffle
        // 152,    // uops port 2: load, vbroadcastss, AGU
        // 153,    // uops port 3: load, vbroadcastss, AGU
        // 154,    // uops port 4: store
    //   156,    // uops port 6: branch
    //   157,    // uops port 7: AGU
    //   160,    // uops port 0~7
    //   311     // data cache misses
    };
}


/*############################################################################
#
#        Thread data
#
############################################################################*/
// Align SThreadData structure by cache line size to avoid multiple threads
// writing to the same cache line
ALIGNEDSTRUCTURE(SThreadData, CACHELINESIZE) {
    //__declspec(align(CACHELINESIZE)) struct SThreadData {
    // Data for each thread
    int CountTemp[MAXCOUNTERS+1];      // temporary storage of clock counts and PMC counts
    int CountOverhead[MAXCOUNTERS+1];  // temporary storage of count overhead
    int ClockResults[REPETITIONS];     // clock count results
    int PMCResults[REPETITIONS*MAXCOUNTERS]; // PMC count results
};

extern "C" {
    SThreadData ThreadData[NUM_THREADS];// Results for all threads
    int NumThreads = NUM_THREADS;       // Number of threads
    int NumCounters = 0;                // Number of valid PMC counters in Counters[]
    int MaxNumCounters = MAXCOUNTERS;   // Maximum number of PMC counters
    int UsePMC = USE_PERFORMANCE_COUNTERS;// 0 if no PMC counters used
    int *PThreadData = (int*)ThreadData;// Pointer to measured data for all threads
    int ThreadDataSize = sizeof(SThreadData);// Size of per-thread counter data block (bytes)
    // offset of clock results of first thread into ThreadData (bytes)
    int ClockResultsOS = int(ThreadData[0].ClockResults-ThreadData[0].CountTemp)*sizeof(int);
    // offset of PMC results of first thread into ThreadData (bytes)
    int PMCResultsOS = int(ThreadData[0].PMCResults-ThreadData[0].CountTemp)*sizeof(int);
    // counter register numbers used
    int Counters[MAXCOUNTERS] = {0};
    int EventRegistersUsed[MAXCOUNTERS] = {0};
    // optional extra output
    int RatioOut[4] = {0};              // See PMCTest.h for explanation
    int TempOut = 0;                    // See PMCTest.h for explanation
	const char * RatioOutTitle = "?";   // Column heading for optional extra output of ratio
    const char * TempOutTitle = "?";    // Column heading for optional arbitrary output
}


/*############################################################################
#
#        User data
#
############################################################################*/

// Put any data definitions your test code needs here:

#define ROUND_UP(A,B)  ((A+B-1)/B*B)  // Round up A to nearest multiple of B

// Make sure USER_DATA_SIZE is a multiple of the cache line size, because there
// is a penalty if multiple threads access the same cache line:
#define USER_DATA_SIZE  ROUND_UP(1000,CACHELINESIZE) 

int UserData[NUM_THREADS][USER_DATA_SIZE];

// using V = Vec16f;
using V = Vec8f;
// using V = Vec4f;

using T = float;

constexpr static int M = 8;
// constexpr static int M = 4;
// constexpr static int M = 16;

using VBlock = std::array<V, M>;

constexpr static int N = 1; // vector, scalar 

alignas(64) std::array<VBlock, N> Inputs, Outputs;
alignas(64) std::array<V, N> V_Inputs, V_Outputs;
// alignas(256) std::array<VBlock, N> Inputs, Outputs;
// alignas(256) std::array<V, N> V_Inputs, V_Outputs;

T b1 = 0.2, b2 = 0.3, a1 = 0.5, a2 = 0.1;

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<T> d{0.,1.};


ZeroInitCond<V> ZIC(b1, b2, a1, a2);
InitCondCorc<V> ICC(a1, a2);
IirCoreOrderTwo<V> I(b1, b2, a1, a2);
IirCoreOrderTwo<V> I1(b1, b2, a1, a2),I2(b1, b2, a1, a2);


//////////////////////////////////////////////////////////////////////////////
//    Test Loop
//////////////////////////////////////////////////////////////////////////////

int TestLoop (int thread) {
    // this function runs the code to test REPETITIONS times
    // and reads the counters before and after each run:
    int i;                        // counter index
    int repi;                     // repetition index

    for (i = 0; i < MAXCOUNTERS+1; i++) {
        ThreadData[thread].CountOverhead[i] = 0x7FFFFFFF;
    }

    /*############################################################################
    #
    #        Initializations
    #
    ############################################################################*/

    // place any user initializations here:
    // with initialization, the pmctest results become more stable.

    for (auto n=0; n<N; n++) {  
        for (auto m=0; m<M; m++) {
            T tmp[M] = {d(gen), d(gen), d(gen), d(gen), d(gen), d(gen), d(gen), d(gen)};
            Inputs[n][m].load(tmp);
        }
    }

    for (auto n=0; n<N; n++) {  
            T tmp[M] = {d(gen), d(gen), d(gen), d(gen), d(gen), d(gen), d(gen), d(gen)};
            V_Inputs[n].load(tmp);
    }

        

    /*############################################################################
    #
    #        Initializations end
    #
    ############################################################################*/

    // first test loop. 
    // Measure overhead = the test count produced by the test program itself
    for (repi = 0; repi < OVERHEAD_REPETITIONS; repi++) {

        Serialize();

#if USE_PERFORMANCE_COUNTERS
        // Read counters
        for (i = 0; i < MAXCOUNTERS; i++) {
            ThreadData[thread].CountTemp[i+1] = (int)Readpmc(Counters[i]);
        }
#endif

        Serialize();
        ThreadData[thread].CountTemp[0] = (int)Readtsc();
        Serialize();

        // no test code here

        Serialize();
        ThreadData[thread].CountTemp[0] -= (int)Readtsc();
        Serialize();

#if USE_PERFORMANCE_COUNTERS
        // Read counters
        for (i = 0; i < MAXCOUNTERS; i++) {
            ThreadData[thread].CountTemp[i+1] -= (int)Readpmc(Counters[i]);
        }
#endif
        Serialize();

        // find minimum counts
        for (i = 0; i < MAXCOUNTERS+1; i++) {
            if (-ThreadData[thread].CountTemp[i] < ThreadData[thread].CountOverhead[i]) {
                ThreadData[thread].CountOverhead[i] = -ThreadData[thread].CountTemp[i];
            }
        }
    }


    // Second test loop. Includes code to test.
    // This must be identical to first test loop, except for the test code
    for (repi = 0; repi < REPETITIONS; repi++) {

        Serialize();

#if USE_PERFORMANCE_COUNTERS
        // Read counters
        for (i = 0; i < MAXCOUNTERS; i++) {
            ThreadData[thread].CountTemp[i+1] = (int)Readpmc(Counters[i]);
        }
#endif

        Serialize();
        ThreadData[thread].CountTemp[0] = (int)Readtsc();
        Serialize();


        /*############################################################################
        #
        #        Test code start
        #
        ############################################################################*/

        // Put the code to test here,
        // or a call to a function defined in a separate module
        //��


        for (auto n=0; n<N; n++) {

            V_Outputs[n] = ZIC.NT_ZIC(V_Inputs[n]);
            // V_Outputs[n] = ICC.NT_ICC(V_Inputs[n]);
            // Outputs[n] = ZIC.T_ZIC(Inputs[n]);
            // Outputs[n] = ICC.T_ICC(Inputs[n]);
            // Outputs[n] = ICC.T_ICC2(Inputs[n]);
            // Outputs[n] = _permuteV(Inputs[n]);
            // V_Outputs[n] = I.Option_1(V_Inputs[n]);
            // Outputs[n] = I.Option_2(Inputs[n]);
            // Outputs[n] = I.Option_3(Inputs[n]);
            // Outputs[n] = I.Option_3_2(Inputs[n]);
            // Outputs[n] = I2.End_Option_2(I1.Front_Option_3(Inputs[n]));
            // Outputs[n] = I2.Option_2(I1.Option_2(Inputs[n]));
            // Outputs[n] = I2.End_Option_3(I1.Front_Option_3(Inputs[n]));



            // for (auto m=0; m<M; m++) Outputs[n][m] = ICC.NT_ICC(Inputs[n][m]);
            // V_Outputs[n] = I3.Option_1(I1.Option_1(V_Inputs[n]));

        }

        /*############################################################################
        #
        #        Test code end
        #
        ############################################################################*/

        Serialize();
        ThreadData[thread].CountTemp[0] -= (int)Readtsc();
        Serialize();

#if USE_PERFORMANCE_COUNTERS
        // Read counters
        for (i = 0; i < MAXCOUNTERS; i++) {
            ThreadData[thread].CountTemp[i+1] -= (int)Readpmc(Counters[i]);
        }
#endif
        Serialize();

        // subtract overhead
        ThreadData[thread].ClockResults[repi] = -ThreadData[thread].CountTemp[0] - ThreadData[thread].CountOverhead[0];
        for (i = 0; i < MAXCOUNTERS; i++) {
            ThreadData[thread].PMCResults[repi+i*REPETITIONS] = -ThreadData[thread].CountTemp[i+1] - ThreadData[thread].CountOverhead[i+1];
        }
    }

    // return
    return REPETITIONS;
}
