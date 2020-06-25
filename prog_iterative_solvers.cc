//Maximilian Köper
//Erik Steinkamp



#include <iostream>    // notwendig zur Ausgabe
#include <vector>
#include "hdnum.hh"    // hdnum header
#include <math.h>       /* pow */
#include <time.h>

//#define MULTIPLE_RUNS
#define OUT
#define UI

namespace hdnum {

  template<typename REAL>
  class SparseMatrix {

    struct MatrixEntry {
      int i;
      int j;
      REAL value;
    };

  public:

    void AddEntry (int i, int j, REAL value) {
      assert(i >= 0);
      assert(j >= 0);
      if (value != .0)
        entries.push_back(MatrixEntry{.i=i, .j=j, .value=value});
    }

    template<typename V>
    void mv_sparse (Vector<V>& y, const Vector<V>& x) {

      zero(y);

      for (MatrixEntry& matrix_entry : entries) {
        assert(y.size() > matrix_entry.i);
        assert(x.size() > matrix_entry.j);
        y[matrix_entry.i] += matrix_entry.value * x[matrix_entry.j];
      }
    }

  private:
    std::vector<MatrixEntry> entries;
  };

}

void iterative_Solver_Dense_Richardson(const hdnum::DenseMatrix<double> A, const hdnum::Vector<double> b, hdnum::Vector<double> x)
{
  float w = 1 / std::min(A.norm_infty(), A.norm_1());
  hdnum::Vector<double> vector_tmp1;
  hdnum::Vector<double> vector_tmp2;
  hdnum::Vector<double> vector_tmp3;
  vector_tmp1.resize(b.size()); 
  vector_tmp2.resize(b.size()); 
  vector_tmp3.resize(b.size());  
  bool first = true;
  float conv_iteration, conv_0;

  do
  {
    A.mv(vector_tmp1, x);
    vector_tmp1 *= w;
    vector_tmp2 = b;
    vector_tmp2 *= w;
    vector_tmp2 -= vector_tmp1;
    x -= vector_tmp2;

    vector_tmp3 -= x;
    conv_iteration = norm(vector_tmp3);
    if(first == true)
    {
      conv_0 = conv_iteration;
      first = false;
    }
    vector_tmp3 = x;
  }while(conv_iteration > (conv_0/10000));

  #ifdef OUT
  std::cout << x << std::endl;
  #endif
}

void iterative_Solver_Dense_Jacobi(const hdnum::DenseMatrix<double> A, const hdnum::Vector<double> b, hdnum::Vector<double> x)
{
  hdnum::DenseMatrix<double> D(A);
  hdnum::identity(D);
  D *= (-0.5);
  //std::cout << D << std::endl;
  hdnum::Vector<double> vector_tmp1;
  hdnum::Vector<double> vector_tmp2;
  hdnum::Vector<double> vector_tmp3;
  vector_tmp1.resize(b.size()); 
  vector_tmp2.resize(b.size()); 
  vector_tmp3.resize(b.size());  
  bool first = true;
  float conv_iteration, conv_0;

  do
  {
    A.mv(vector_tmp1, x);
    vector_tmp2 = b;
    vector_tmp2 -= vector_tmp1;
    D.mv(vector_tmp3, vector_tmp2);
    x += vector_tmp3;


    vector_tmp2 = b;
    vector_tmp2 -= vector_tmp1;
    conv_iteration = norm(vector_tmp2);
    if(first == true)
    {
      conv_0 = conv_iteration;
      first = false;
    }
  }while(conv_iteration > (conv_0/10000));

  
  #ifdef OUT
  std::cout << x << std::endl;
  #endif

}

void iterative_Solver_Dense_GaussSeidel(const hdnum::DenseMatrix<double> A, const hdnum::Vector<double> b, hdnum::Vector<double> x)
{
  hdnum::DenseMatrix<double> D(A.rowsize(), A.colsize());
  for (size_t i = 0; i < D.colsize(); i++)
  {
    for (size_t j = i; j < D.colsize(); j++)
    {
      D[j][i] = -(1.0 / pow(2, j-i+1)); 
    }
  }
  
  //std::cout << D << std::endl;
  hdnum::Vector<double> vector_tmp1;
  hdnum::Vector<double> vector_tmp2;
  hdnum::Vector<double> vector_tmp3;
  vector_tmp1.resize(b.size()); 
  vector_tmp2.resize(b.size()); 
  vector_tmp3.resize(b.size());  
  bool first = true;
  float conv_iteration, conv_0;

  do
  {
    A.mv(vector_tmp1, x);
    vector_tmp2 = b;
    vector_tmp2 -= vector_tmp1;
    D.mv(vector_tmp3, vector_tmp2);
    x += vector_tmp3;


    vector_tmp2 = b;
    vector_tmp2 -= vector_tmp1;
    conv_iteration = norm(vector_tmp2);
    if(first == true)
    {
      conv_0 = conv_iteration;
      first = false;
    }
  }while(conv_iteration > (conv_0/10000));

  
  #ifdef OUT
  std::cout << x << std::endl;
  #endif
}


void iterative_Solver_Sparse_Richardson(hdnum::SparseMatrix<double> B, const hdnum::Vector<double> b, hdnum::Vector<double> x)
{
  float w = 0.5;
  hdnum::Vector<double> vector_tmp1;
  hdnum::Vector<double> vector_tmp2;
  hdnum::Vector<double> vector_tmp3;
  vector_tmp1.resize(b.size()); 
  vector_tmp2.resize(b.size()); 
  vector_tmp3.resize(b.size());  
  bool first = true;
  float conv_iteration, conv_0;

  do
  {
    B.mv_sparse(vector_tmp1, x);
    vector_tmp3 = vector_tmp1;
    vector_tmp1 *= w;
    vector_tmp2 = b;
    vector_tmp2 *= w;
    vector_tmp2 -= vector_tmp1;
    x -= vector_tmp2;


    vector_tmp1 = b;
    vector_tmp1 -= vector_tmp3;
    conv_iteration = norm(vector_tmp1);
    if(first == true)
    {
      conv_0 = conv_iteration;
      first = false;
    }
  }while(conv_iteration > (conv_0/10000));

  
  #ifdef OUT
  std::cout << x << std::endl;
  #endif
  
}


void iterative_Solver_Sparse_Jacobi(hdnum::SparseMatrix<double> B, const hdnum::Vector<double> b, hdnum::Vector<double> x, const int N=16)
{
  hdnum::SparseMatrix<double> D;
  for (size_t i = 0; i < N; i++)
  {
      D.AddEntry(i,i,-0.5);
  }
  //std::cout << D << std::endl;
  hdnum::Vector<double> vector_tmp1;
  hdnum::Vector<double> vector_tmp2;
  hdnum::Vector<double> vector_tmp3;
  vector_tmp1.resize(b.size()); 
  vector_tmp2.resize(b.size()); 
  vector_tmp3.resize(b.size()); 
  bool first = true;
  float conv_iteration, conv_0;

  do
  {
    B.mv_sparse(vector_tmp1, x);
    vector_tmp2 = b;
    vector_tmp2 -= vector_tmp1;
    D.mv_sparse(vector_tmp3, vector_tmp2);
    x += vector_tmp3;


    vector_tmp2 = b;
    vector_tmp2 -= vector_tmp1;
    conv_iteration = norm(vector_tmp2);
    if(first == true)
    {
      conv_0 = conv_iteration;
      first = false;
    }
  }while(conv_iteration > (conv_0/10000));

  
  #ifdef OUT
  std::cout << x << std::endl;
  #endif

}

void iterative_Solver_Sparse_GaussSeidel(hdnum::SparseMatrix<double> B, const hdnum::Vector<double> b, hdnum::Vector<double> x, const int N=16)
{
  hdnum::SparseMatrix<double> D;
  for (size_t i = 0; i < N; i++)
  {
    for (size_t j = i; j < N; j++)
    {
      D.AddEntry(j,i,-(1.0 / pow(2, j-i+1)));
    }
  }
  
  hdnum::Vector<double> vector_tmp1;
  hdnum::Vector<double> vector_tmp2;
  hdnum::Vector<double> vector_tmp3;
  vector_tmp1.resize(b.size()); 
  vector_tmp2.resize(b.size()); 
  vector_tmp3.resize(b.size()); 
  bool first = true;
  float conv_iteration, conv_0;

  do
  {
    B.mv_sparse(vector_tmp1, x);
    vector_tmp2 = b;
    vector_tmp2 -= vector_tmp1;
    D.mv_sparse(vector_tmp3, vector_tmp2);
    x += vector_tmp3;


    vector_tmp2 = b;
    vector_tmp2 -= vector_tmp1;
    conv_iteration = norm(vector_tmp2);
    if(first == true)
    {
      conv_0 = conv_iteration;
      first = false;
    }
  }while(conv_iteration > (conv_0/10000));

  
  #ifdef OUT
  std::cout << x << std::endl;
  #endif
}


int main ()
{
  int N = 16;

  #ifdef MULTIPLE_RUNS
  for (N = 10; N < 1000; N=N+10)
  {
  #endif


  // Testmatrix aufsetzen
  hdnum::DenseMatrix<double> A(N,N,.0);
  hdnum::SparseMatrix<double> B;
  for (typename hdnum::DenseMatrix<double>::size_type i=0; i<A.rowsize(); ++i)
  {
    if (i > 0) {
      A[i][i-1] = 1.0;
      B.AddEntry(i,i-1, 1.0);
    }
    if (i + 1 < A.colsize()) {
      A[i][i+1] = 1.0;
      B.AddEntry(i,i+1, 1.0);
    }
    A[i][i] -= 2.0;
    B.AddEntry(i,i, -2.0);
  }
  //std::cout << B << std::endl;

  // Rechte Seite und Lösungsvektor
  hdnum::Vector<double> x(N, 0.0);
  hdnum::Vector<double> b(N, 1.0);

  // Lösen Sie nun A*x=b iterativ

  double time1=0.0, tstart;
  std::cout << std::fixed << std::setprecision(9);
  

  tstart = clock();
  iterative_Solver_Dense_Richardson(A,b,x);
  time1 = clock() - tstart;
  #ifdef UI
  std::cout << "Richardson mit Dense Matrix\n"<< time1/CLOCKS_PER_SEC << std::endl;
  #endif
  #ifndef UI
  std::cout << time1/CLOCKS_PER_SEC << " ";
  #endif
  tstart = clock();
  iterative_Solver_Dense_Jacobi(A,b,x);
  time1 = clock() - tstart;
  #ifdef UI
  std::cout << "Jacobi mit Dense Matrix\n"<< time1/CLOCKS_PER_SEC << std::endl;
  #endif
  #ifndef UI
  std::cout << time1/CLOCKS_PER_SEC << " ";
  #endif
  tstart = clock();
  iterative_Solver_Dense_GaussSeidel(A,b,x);
  time1 = clock() - tstart;
  #ifdef UI
  std::cout << "Gauß-Seidel mit Dense Matrix\n"<< time1/CLOCKS_PER_SEC << std::endl;
  #endif
  #ifndef UI
  std::cout << time1/CLOCKS_PER_SEC << " ";
  #endif


  tstart = clock();
  iterative_Solver_Sparse_Richardson(B,b,x);
  time1 = clock() - tstart;
  #ifdef UI
  std::cout << "Richardson mit Sparse Matrix\n"<< time1/CLOCKS_PER_SEC << std::endl;
  #endif
  #ifndef UI
  std::cout << time1/CLOCKS_PER_SEC << " ";
  #endif
  tstart = clock();
  iterative_Solver_Sparse_Jacobi(B,b,x,N);
  time1 = clock() - tstart;
  #ifdef UI
  std::cout << "Jacobi mit Dense Matrix\n"<< time1/CLOCKS_PER_SEC << std::endl;
  #endif
  #ifndef UI
  std::cout << time1/CLOCKS_PER_SEC << " ";
  #endif
  tstart = clock();
  iterative_Solver_Sparse_GaussSeidel(B,b,x,N);
  time1 = clock() - tstart;
  #ifdef UI
  std::cout << "Gauß-Seidel mit Dense Matrix\n"<< time1/CLOCKS_PER_SEC << std::endl;
  #endif
  #ifndef UI
  std::cout << time1/CLOCKS_PER_SEC << " ";
  #endif


  tstart = clock();
  hdnum::linsolve(A,x,b);
  time1 = clock() - tstart;
  #ifdef OUT
  std::cout << x << std::endl;
  #endif
  #ifdef UI
  std::cout << "Linsolve\n"<< time1/CLOCKS_PER_SEC << std::endl;
  #endif
  #ifndef UI
  std::cout << time1/CLOCKS_PER_SEC << " " << std::endl;
  #endif


  #ifdef MULTIPLE_RUNS
  }
  #endif

}
