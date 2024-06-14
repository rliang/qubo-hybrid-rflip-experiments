#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "json.hpp"

using nlohmann::json;
using std::async;
using std::cout;
using std::endl;
using std::function;
using std::future;
using std::ifstream;
using std::iota;
using std::lock_guard;
using std::make_shared;
using std::max;
using std::move;
using std::mt19937;
using std::mutex;
using std::numeric_limits;
using std::ofstream;
using std::pair;
using std::setw;
using std::string;
using std::swap;
using std::to_string;
using std::unique_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;
using std::chrono::steady_clock;

/**
 * A UBQP problem instance.
 */
struct ubqp {
  /** The number of variables in a solution vector. */
  size_t n;

  /** The matrix, stored in row-major order. */
  unique_ptr<long[]> Q;

  /**
   * Constructs an UBQP object.
   */
  ubqp(size_t n) : n(n), Q(new long[n * n]()) {}

  /**
   * Obtains the pointer to the beginning of a row of the matrix.
   */
  long* operator[](size_t i) const { return &Q[n * i]; }

  /**
   * Constructs a UBQP problem instance from a file.
   */
  static ubqp load(string label) {
    size_t n, number;
    if (sscanf(label.c_str(), "bqp%zu.%zu", &n, &number) == 2) return from_bqp(n, number);
    if (sscanf(label.c_str(), "G%zu", &number) == 1) return from_maxcut(number);
    assert(false);
    exit(1);
  }

  /**
   * Constructs a UBQP problem instance from an OR-Library file.
   */
  static ubqp from_bqp(size_t n, size_t number) {
    ifstream file("instances/bqp" + to_string(n) + string(".txt"), ifstream::in);
    assert(file);
    size_t count;
    file >> count;
    for (size_t k = 1; k <= count; k++) {
      size_t nonzeros, i, j;
      long q;
      if (k == number) {
        file >> n >> nonzeros;
        ubqp Q(n);
        for (size_t l = 0; l < nonzeros; l++) {
          file >> i >> j >> q;
          i -= 1;
          j -= 1;
          Q[i][j] = Q[j][i] = i == j ? -q : 2 * -q;
        }
        return Q;
      } else {
        file >> n >> nonzeros;
        for (size_t l = 0; l < nonzeros; l++) file >> i >> j >> q;
      }
    }
    assert(false);
    exit(1);
  }

  /**
   * Constructs a UBQP problem instance from a Max-Cut problem file.
   */
  static ubqp from_maxcut(size_t number) {
    ifstream file("instances/G" + to_string(number), ifstream::in);
    assert(file);
    size_t n, nonzeros;
    file >> n >> nonzeros;
    ubqp Q(n);
    for (size_t l = 0; l < nonzeros; l++) {
      size_t i, j;
      long q;
      file >> i >> j >> q;
      i -= 1;
      j -= 1;
      Q[i][i] -= q;
      Q[j][j] -= q;
      Q[i][j] += 2 * q;
      Q[j][i] += 2 * q;
    }
    return Q;
  }
};

/**
 * Represents an evaluation algorithm, or hybrid strategy, to use when evaluating UBQP solutions.
 */
enum struct evaluation { basic, rflip_rv, s, a, c, m, ac, am, cm, acm };

/**
 * Represents an incumbent solution in a UBQP search process.
 */
template <evaluation eval>
struct incumbent_solution {
  /** The incumbent's objective function value. */
  long fy;
  /** The incumbent's solution vector. */
  unique_ptr<bool[]> y;
  /** The previous solution vector with an updated reevaluation vector. */
  unique_ptr<bool[]> x;
  /** The number of non-zero components in x. */
  size_t nx1;
  /** The number of components which have changed from 0 to 1 between x and y. */
  size_t wx01;
  /** The number of components which have changed from 1 to 0 between x and y. */
  size_t wx10;
  /** The components which have changed from 0 to 1 between x and y. */
  unique_ptr<size_t[]> WX01;
  /** The components which have changed from 1 to 0 between x and y. */
  unique_ptr<size_t[]> WX10;
  /** The reevaluation vector of x. */
  unique_ptr<long[]> dx;

  /**
   * Constructs an incumbent solution.
   */
  incumbent_solution(const ubqp& Q)
      : fy(0),
        y(new bool[Q.n]()),
        x(new bool[Q.n]()),
        nx1(0),
        wx01(0),
        wx10(0),
        WX01(new size_t[Q.n]()),
        WX10(new size_t[Q.n]()),
        dx(new long[Q.n]()) {}
};

/**
 * Specialization of the incumbent solution for the basic evaluation algorithm.
 */
template <>
struct incumbent_solution<evaluation::basic> {
  /** The incumbent's objective function value. */
  long fy;
  /** The incumbent's solution vector. */
  unique_ptr<bool[]> y;

  /**
   * Constructs an incumbent solution.
   */
  incumbent_solution(const ubqp& Q) : fy(0), y(new bool[Q.n]()) {}
};

/**
 * Represents a neighbor solution obtained from an r-flip move.
 */
struct neighbor_solution {
  /** The neighbor's objective function value. */
  long fz;
  /** The number of non-zero components in the neighbor. */
  size_t n1z;
  /** The number of components which changed from 0 to 1 between the incumbent and the neighbor. */
  size_t r01;
  /** The number of components which changed from 1 to 0 between the incumbent and the neighbor. */
  size_t r10;
  /** The non-zero components in the neighbor. */
  unique_ptr<size_t[]> N1z;
  /** The components which changed from 0 to 1 between the incumbent and the neighbor. */
  unique_ptr<size_t[]> R01;
  /** The components which changed from 1 to 0 between the incumbent and the neighbor. */
  unique_ptr<size_t[]> R10;

  /**
   * Constructs a neighbor solution.
   */
  neighbor_solution(size_t n)
      : fz(0),
        n1z(0),
        r01(0),
        r10(0),
        N1z(new size_t[n]()),
        R01(new size_t[n]()),
        R10(new size_t[n]()) {}
};

/**
 * Checks whether a neighbor solution has the correct objective function value.
 */
template <evaluation eval>
bool check_evaluation(const ubqp& Q, const incumbent_solution<eval>& y,
                      const neighbor_solution& z) {
  unique_ptr<bool[]> z_tmp(new bool[Q.n]());
  for (size_t i = 0; i < Q.n; i++) z_tmp[i] = y.y[i];
  for (size_t i = 0; i < z.r01; i++) z_tmp[z.R01[i]] = 1;
  for (size_t i = 0; i < z.r10; i++) z_tmp[z.R10[i]] = 0;
  long fz_tmp = 0;
  for (size_t i = 0; i < Q.n; i++)
    for (size_t j = 0; j <= i; j++)
      if (z_tmp[i] && z_tmp[j]) fz_tmp += Q[i][j];
  return z.fz == fz_tmp;
}

/**
 * Evaluates a neighbor solution with the basic algorithm.
 *
 * The N vector contains the changed components in the first r components,
 * and the unchanged components in the (n-r) last components.
 */
template <evaluation eval>
void evaluate_basic(const ubqp& Q, const incumbent_solution<eval>& y, neighbor_solution& z,
                    const unique_ptr<size_t[]>& N) {
  z.n1z = 0;
  for (size_t k = 0; k < z.r01; k++) z.N1z[z.n1z++] = z.R01[k];
  for (size_t k = z.r01 + z.r10; k < Q.n; k++) {
    size_t i = N[k];
    if (y.y[i]) z.N1z[z.n1z++] = i;
  }
  z.fz = 0;
  for (size_t m = 0; m < z.n1z; m++) {
    size_t i = z.N1z[m];
    z.fz += Q[i][i];
    for (size_t l = 0; l < m; l++) {
      size_t j = z.N1z[l];
      z.fz += Q[i][j];
    }
  }
  assert(check_evaluation(Q, y, z));
}

/**
 * Checks whether the reevaluation vector of an incumbent solution has the correct value.
 */
template <evaluation eval>
bool check_rv(const ubqp& Q, const incumbent_solution<eval>& y) {
  static_assert(eval != evaluation::basic);
  for (size_t i = 0; i < Q.n; i++) {
    long dxi_tmp = 0;
    for (size_t j = 0; j < Q.n; j++)
      if (y.y[j]) dxi_tmp += Q[i][j];
    if (y.dx[i] != dxi_tmp) return 0;
  }
  return 1;
}

/**
 * Checks whether the sets of components which changed between x and y are correct.
 */
template <evaluation eval>
bool check_sets(const incumbent_solution<eval>& y) {
  static_assert(eval != evaluation::basic);
  for (size_t k = 0; k < y.wx01; k++) {
    size_t i = y.WX01[k];
    if (y.x[i] != 0) return 0;
    if (y.y[i] != 1) return 0;
  }
  for (size_t k = 0; k < y.wx10; k++) {
    size_t i = y.WX10[k];
    if (y.x[i] != 1) return 0;
    if (y.y[i] != 0) return 0;
  }
  return 1;
}

/**
 * Evaluates a neighbor solution with the r-flip-rv algorithm.
 */
template <evaluation eval>
void evaluate_rfliprv(const ubqp& Q, const incumbent_solution<eval>& y, neighbor_solution& z) {
  static_assert(eval != evaluation::basic);
  z.fz = y.fy;
  for (size_t m = 0; m < z.r10; m++) {
    size_t i = z.R10[m];
    z.fz -= y.dx[i];
    for (size_t l = m + 1; l < z.r10; l++) {
      size_t j = z.R10[l];
      z.fz += Q[j][i];
    }
  }
  for (size_t m = 0; m < z.r01; m++) {
    size_t i = z.R01[m];
    z.fz += y.dx[i];
    z.fz += Q[i][i];
    for (size_t l = 0; l < m; l++) {
      size_t j = z.R01[l];
      z.fz += Q[i][j];
    }
    for (size_t l = 0; l < z.r10; l++) {
      size_t j = z.R10[l];
      z.fz -= j <= i ? Q[i][j] : Q[j][i];
    }
  }
  assert(check_evaluation(Q, y, z));
}

/**
 * Rebuilds an incumbent's reevaluation vector from scratch.
 */
template <evaluation eval>
void build_rv(const ubqp& Q, incumbent_solution<eval>& y) {
  static_assert(eval != evaluation::basic);
  for (size_t i = 0; i < Q.n; i++) y.dx[i] = 0;
  for (size_t j = 0; j < Q.n; j++)
    if (y.y[j])
      for (size_t i = 0; i < Q.n; i++) y.dx[i] += Q[i][j];
  assert(check_rv(Q, y));
}

/**
 * Updates an incumbent's reevaluation vector with information from the previous solution.
 */
template <evaluation eval>
void update_rv(const ubqp& Q, incumbent_solution<eval>& y) {
  static_assert(eval != evaluation::basic);
  for (size_t i = 0; i < Q.n; i++) {
    for (size_t k = 0; k < y.wx01; k++) {
      size_t j = y.WX01[k];
      y.dx[i] += Q[i][j];
    }
    for (size_t k = 0; k < y.wx10; k++) {
      size_t j = y.WX10[k];
      y.dx[i] -= Q[i][j];
    }
  }
  assert(check_rv(Q, y));
}

/**
 * Rebuilds or updates an incumbent's reevaluation vector, depending on the evaluation strategy.
 */
template <evaluation eval>
void update_rv_hybrid(const ubqp& Q, incumbent_solution<eval>& y) {
  static_assert(eval != evaluation::basic);
  size_t r = y.wx01 + y.wx10;
  if (!r) return;
  size_t basic_ops = 0, delta_ops = 0;
  if constexpr (eval == evaluation::s) {
    basic_ops = Q.n * y.nx1;
    delta_ops = Q.n * r;
  } else if constexpr (eval == evaluation::a) {
    basic_ops = Q.n + y.nx1 + Q.n + Q.n * y.nx1 + Q.n * y.nx1;
    delta_ops = Q.n * (2 * r + 1);
  } else if constexpr (eval == evaluation::c) {
    basic_ops = Q.n + 1 + Q.n + Q.n + 1 + Q.n * (y.nx1 + 1);
    delta_ops = 1 + Q.n * (r + 3);
  } else if constexpr (eval == evaluation::m) {
    basic_ops = 3 + Q.n + y.nx1 + Q.n + Q.n + Q.n * y.nx1 + 3 * Q.n * y.nx1;
    delta_ops = 1 + Q.n * (4 * r + 3);
  } else if constexpr (eval == evaluation::ac) {
    basic_ops = Q.n + y.nx1 + Q.n + Q.n * y.nx1 + Q.n * y.nx1;
    basic_ops += Q.n + 1 + Q.n + Q.n + 1 + Q.n * (y.nx1 + 1);
    delta_ops = Q.n * (2 * r + 1);
    delta_ops += 1 + Q.n * (r + 3);
  } else if constexpr (eval == evaluation::am) {
    basic_ops = Q.n + y.nx1 + Q.n + Q.n * y.nx1 + Q.n * y.nx1;
    basic_ops += 3 + Q.n + y.nx1 + Q.n + Q.n + Q.n * y.nx1 + 3 * Q.n * y.nx1;
    delta_ops = Q.n * (2 * r + 1);
    delta_ops += 1 + Q.n * (4 * r + 3);
  } else if constexpr (eval == evaluation::cm) {
    basic_ops = Q.n + 1 + Q.n + Q.n + 1 + Q.n * (y.nx1 + 1);
    basic_ops += 3 + Q.n + y.nx1 + Q.n + Q.n + Q.n * y.nx1 + 3 * Q.n * y.nx1;
    delta_ops = 1 + Q.n * (r + 3);
    delta_ops += 1 + Q.n * (4 * r + 3);
  } else if constexpr (eval == evaluation::acm) {
    basic_ops = Q.n + y.nx1 + Q.n + Q.n * y.nx1 + Q.n * y.nx1;
    basic_ops += Q.n + 1 + Q.n + Q.n + 1 + Q.n * (y.nx1 + 1);
    basic_ops += 3 + Q.n + y.nx1 + Q.n + Q.n + Q.n * y.nx1 + 3 * Q.n * y.nx1;
    delta_ops = Q.n * (2 * r + 1);
    delta_ops += 1 + Q.n * (r + 3);
    delta_ops += 1 + Q.n * (4 * r + 3);
  }
  if constexpr (eval == evaluation::rflip_rv) {
    update_rv(Q, y);
  } else {
    if (basic_ops < delta_ops)
      build_rv(Q, y);
    else
      update_rv(Q, y);
  }
  y.wx10 = y.wx01 = 0;
  for (size_t i = 0; i < Q.n; i++) y.x[i] = y.y[i];
}

/**
 * Evaluates a neighbor solution with the basic or the r-flip-rv algorithm,
 * depending on the evaluation strategy.
 *
 * The count template parameter specifies whether to increment the basics and deltas counter
 * when one algorithm is chosen.
 */
template <evaluation eval, bool count>
void evaluate_hybrid(const ubqp& Q, incumbent_solution<eval>& y, neighbor_solution& z,
                     const unique_ptr<size_t[]>& N, size_t& basics, size_t& deltas) {
  size_t r = z.r01 + z.r10;
  size_t basic_ops = 0, delta_ops = 0;
  if constexpr (eval == evaluation::s) {
    size_t ny1 = (y.nx1 + z.r01) - z.r10;
    basic_ops = (Q.n - z.r10) + ny1 * ny1;
    delta_ops = r * r;
  } else if constexpr (eval == evaluation::a) {
    size_t ny1 = (y.nx1 + z.r01) - z.r10;
    basic_ops = (Q.n - z.r10) + ny1 * (ny1 + 3);
    delta_ops = 2 + z.r01 + r * (r + 2);
  } else if constexpr (eval == evaluation::c) {
    size_t ny1 = (y.nx1 + z.r01) - z.r10;
    basic_ops = 3 + z.r01 + 2 * (Q.n - r) + ny1 * ((ny1 + 3) / 2);
    delta_ops = 2 + z.r01 + ((r * (r + 3)) / 2);
  } else if constexpr (eval == evaluation::m) {
    size_t ny1 = (y.nx1 + z.r01) - z.r10;
    basic_ops = 5 + (Q.n - r) + ny1 * (ny1 + 3);
    delta_ops = 3 + 2 * z.r01 + r * (r + 2);
  } else if constexpr (eval == evaluation::ac) {
    size_t ny1 = (y.nx1 + z.r01) - z.r10;
    basic_ops = (Q.n - z.r10) + ny1 * (ny1 + 3) + 3 + z.r01 + 2 * (Q.n - r) + ny1 * ((ny1 + 3) / 2);
    delta_ops = 2 + z.r01 + r * (r + 2) + 2 + z.r01 + ((r * (r + 3)) / 2);
  } else if constexpr (eval == evaluation::am) {
    size_t ny1 = (y.nx1 + z.r01) - z.r10;
    basic_ops = (Q.n - z.r10) + ny1 * (ny1 + 3) + 5 + (Q.n - r) + ny1 * (ny1 + 3);
    delta_ops = 2 + z.r01 + r * (r + 2) + 3 + 2 * z.r01 + r * (r + 2);
  } else if constexpr (eval == evaluation::cm) {
    size_t ny1 = (y.nx1 + z.r01) - z.r10;
    basic_ops = 3 + z.r01 + 2 * (Q.n - r) + ny1 * ((ny1 + 3) / 2) + 5 + (Q.n - r) + ny1 * (ny1 + 3);
    delta_ops = 2 + z.r01 + z.r01 + ((r * (r + 3)) / 2) + 3 + 2 * z.r01 + r * (r + 2);
  } else if constexpr (eval == evaluation::acm) {
    size_t ny1 = (y.nx1 + z.r01) - z.r10;
    basic_ops = (Q.n - z.r10) + ny1 * (ny1 + 3) + 3 + z.r01 + 2 * (Q.n - r) +
                ny1 * ((ny1 + 3) / 2) + 5 + (Q.n - r) + ny1 * (ny1 + 3);
    delta_ops =
        2 + z.r01 + r * (r + 2) + 2 + z.r01 + ((r * (r + 3)) / 2) + 3 + 2 * z.r01 + r * (r + 2);
  }
  if constexpr (eval == evaluation::basic) {
    if constexpr (count) basics++;
    evaluate_basic(Q, y, z, N);
  } else if constexpr (eval == evaluation::rflip_rv) {
    if constexpr (count) deltas++;
    update_rv_hybrid(Q, y);
    evaluate_rfliprv(Q, y, z);
  } else if (delta_ops < basic_ops) {
    if constexpr (count) deltas++;
    update_rv_hybrid(Q, y);
    evaluate_rfliprv(Q, y, z);
  } else {
    if constexpr (count) basics++;
    evaluate_basic(Q, y, z, N);
  }
}

/**
 * Randomizes an incumbent solution, and also computes its reevaluation vector.
 */
template <evaluation eval>
void randomize(const ubqp& Q, incumbent_solution<eval>& y, mt19937& rng) {
  for (size_t i = 0; i < Q.n; i++) {
    if (rng() % 2) {
      y.y[i] = 1;
      if constexpr (eval != evaluation::basic) {
        y.x[i] = 1;
        y.nx1++;
      }
    }
  }
  for (size_t i = 0; i < Q.n; i++)
    for (size_t j = 0; j <= i; j++)
      if (y.y[i] && y.y[j]) y.fy += Q[i][j];
  if constexpr (eval != evaluation::basic) {
    for (size_t i = 0; i < Q.n; i++)
      for (size_t j = 0; j < Q.n; j++)
        if (y.y[j]) y.dx[i] += Q[i][j];
  }
}

/**
 * Performs a random r-flip move on an incumbent, resulting in a neighbor solution.
 */
template <evaluation eval>
void random_neighbor_solution(const ubqp& Q, const incumbent_solution<eval>& y,
                              neighbor_solution& z, size_t r, unique_ptr<size_t[]>& N,
                              mt19937& rng) {
  z.r01 = z.r10 = 0;
  if constexpr (eval != evaluation::basic) z.n1z = y.nx1;
  for (size_t m = 0; m < r; m++) {
    swap(N[m], N[m + (rng() % (Q.n - m))]);
    if (!y.y[N[m]]) {
      z.R01[z.r01++] = N[m];
      if constexpr (eval != evaluation::basic) z.n1z++;
    } else {
      z.R10[z.r10++] = N[m];
      if constexpr (eval != evaluation::basic) z.n1z--;
    }
  }
}

/**
 * Replaces an incumbent with a neighbor.
 */
template <evaluation eval>
void replace_incumbent(incumbent_solution<eval>& y, const neighbor_solution& z) {
  y.fy = z.fz;
  for (size_t k = 0; k < z.r01; k++) y.y[z.R01[k]] = 1;
  for (size_t k = 0; k < z.r10; k++) y.y[z.R10[k]] = 0;
  if constexpr (eval != evaluation::basic) {
    y.nx1 = z.n1z;
    size_t k = 0;
    while (k < y.wx01) {
      size_t i = y.WX01[k];
      if (y.y[i] == 0)
        swap(y.WX01[k], y.WX01[--y.wx01]);
      else
        k++;
    }
    k = 0;
    while (k < y.wx10) {
      size_t i = y.WX10[k];
      if (y.y[i] == 1)
        swap(y.WX10[k], y.WX10[--y.wx10]);
      else
        k++;
    }
    for (size_t k = 0; k < z.r01; k++) {
      size_t i = z.R01[k];
      if (y.x[i] == 0) y.WX01[y.wx01++] = i;
    }
    for (size_t k = 0; k < z.r10; k++) {
      size_t i = z.R10[k];
      if (y.x[i] == 1) y.WX10[y.wx10++] = i;
    }
    assert(check_sets(y));
  }
}

/**
 * Performs a local search in an incumbent solution.
 *
 * The count template parameter specifies whether to increment the basics and deltas counter
 * when one algorithm is chosen.
 */
template <evaluation eval, bool count>
void ls(const ubqp& Q, incumbent_solution<eval>& y, neighbor_solution& z, size_t r, size_t iters,
        unique_ptr<size_t[]>& N, mt19937& rng, size_t& basics, size_t& deltas) {
  for (size_t l = 1; l <= iters; l++) {
    random_neighbor_solution(Q, y, z, r, N, rng);
    evaluate_hybrid<eval, count>(Q, y, z, N, basics, deltas);
    if (z.fz < y.fy) {
      replace_incumbent(y, z);
      l = 0;
    }
  }
}

/**
 * Performs a variable neighborhood search in an incumbent solution.
 *
 * The count template parameter specifies whether to increment the basics and deltas counter
 * when one algorithm is chosen.
 */
template <evaluation eval, bool count>
void vns(const ubqp& Q, incumbent_solution<eval>& y_inc, incumbent_solution<eval>& y,
         neighbor_solution& z, neighbor_solution& z2, size_t r_max, size_t r_step, size_t iters,
         size_t ls_iters, unique_ptr<size_t[]>& N, mt19937& rng, size_t& basics, size_t& deltas) {
  for (size_t l = 1; l <= iters; l++) {
    for (size_t r = 1; r <= r_max; r += r_step) {
      y.fy = y_inc.fy;
      for (size_t i = 0; i < Q.n; i++) y.y[i] = y_inc.y[i];
      if constexpr (eval != evaluation::basic) {
        y.nx1 = y_inc.nx1;
        y.wx01 = y_inc.wx01;
        y.wx10 = y_inc.wx10;
        for (size_t i = 0; i < Q.n; i++) y.x[i] = y_inc.x[i];
        for (size_t i = 0; i < Q.n; i++) y.dx[i] = y_inc.dx[i];
        for (size_t i = 0; i < y.wx01; i++) y.WX01[i] = y_inc.WX01[i];
        for (size_t i = 0; i < y.wx10; i++) y.WX10[i] = y_inc.WX10[i];
      }
      random_neighbor_solution(Q, y, z, r, N, rng);
      evaluate_hybrid<eval, count>(Q, y, z, N, basics, deltas);
      replace_incumbent(y, z);
      ls<eval, count>(Q, y, z2, r, ls_iters, N, rng, basics, deltas);
      if (y.fy < y_inc.fy) {
        swap(y_inc, y);
        r = 0;
        l = 1;
      }
    }
  }
}

/**
 * Measures the processing time of a given function.
 */
size_t measure(function<void(void)> f) {
  auto t1 = steady_clock::now();
  f();
  auto t2 = steady_clock::now();
  return duration_cast<nanoseconds>(t2 - t1).count();
}

/**
 * Performs an experiment which measures the processing time of an evaluation strategy.
 */
template <evaluation eval>
json eval_experiment(const ubqp& Q, const json params) {
  mt19937 rng;
  size_t r = params["r"];
  size_t n1 = params["n1"];
  size_t iters = params["iters"];
  size_t basics, deltas;
  long maximum = 0, minimum = numeric_limits<long>::max(), sum = 0;
  unique_ptr<size_t[]> N(new size_t[Q.n]());
  iota(&N[0], &N[Q.n], 0);
  incumbent_solution<eval> x(Q);
  randomize(Q, x, rng);
  neighbor_solution y(Q.n);
  for (size_t _ = 0; _ < iters; _++) {
    for (size_t m = 0; m < n1; m++) {
      swap(N[m], N[m + (rng() % (Q.n - m))]);
      x.y[N[m]] = 1;
    }
    for (size_t m = n1; m < Q.n; m++) x.y[N[m]] = 0;
    x.fy = 0;
    for (size_t i = 0; i < Q.n; i++)
      for (size_t j = 0; j <= i; j++)
        if (x.y[i] && x.y[j]) x.fy += Q[i][j];
    if constexpr (eval != evaluation::basic) {
      build_rv(Q, x);
      x.wx01 = x.wx10 = 0;
    }
    random_neighbor_solution(Q, x, y, r, N, rng);
    evaluate_hybrid<eval, false>(Q, x, y, N, basics, deltas);
    replace_incumbent(x, y);
    swap(y.R01, y.R10);
    swap(y.r01, y.r10);
    if constexpr (eval != evaluation::basic) {
      build_rv(Q, x);
      x.wx01 = x.wx10 = 0;
    }
    long dt = measure([&]() { evaluate_hybrid<eval, false>(Q, x, y, N, basics, deltas); });
    sum += dt;
    if (dt > maximum) maximum = dt;
    if (dt < minimum) minimum = dt;
  }
  return {{"max", maximum}, {"min", minimum}, {"avg", sum / iters}};
}

/**
 * Performs an experiment which measures the processing time of a local search.
 *
 * The count template parameter specifies whether to increment the basics and deltas counter
 * when one algorithm is chosen.
 */
template <evaluation eval, bool count>
json ls_experiment(const ubqp& Q, const json params) {
  mt19937 rng;
  size_t r = params["r"];
  size_t iters = params["iters"];
  size_t basics = 0, deltas = 0;
  incumbent_solution<eval> y(Q);
  randomize(Q, y, rng);
  neighbor_solution z(Q.n);
  unique_ptr<size_t[]> N(new size_t[Q.n]());
  iota(&N[0], &N[Q.n], 0);
  if constexpr (count) {
    ls<eval, count>(Q, y, z, r, iters, N, rng, basics, deltas);
    return {{"basics", basics}, {"deltas", deltas}};
  } else {
    long dt = measure([&]() { ls<eval, count>(Q, y, z, r, iters, N, rng, basics, deltas); });
    return {{"dt", dt}, {"fx", y.fy}};
  }
}

/**
 * Performs an experiment which measures the processing time of a variable neighborhood search.
 *
 * The count template parameter specifies whether to increment the basics and deltas counter
 * when one algorithm is chosen.
 */
template <evaluation eval, bool count>
json vns_experiment(const ubqp& Q, const json params) {
  mt19937 rng;
  size_t iters = params["iters"];
  size_t ls_iters = params["ls_iters"];
  size_t r_max = params["r_max"];
  size_t r_step = params["r_step"];
  size_t basics = 0, deltas = 0;
  incumbent_solution<eval> y_inc(Q), y(Q);
  randomize(Q, y_inc, rng);
  neighbor_solution z(Q.n), z2(Q.n);
  unique_ptr<size_t[]> N(new size_t[Q.n]());
  iota(&N[0], &N[Q.n], 0);
  if constexpr (count) {
    vns<eval, count>(Q, y_inc, y, z, z2, r_max, r_step, iters, ls_iters, N, rng, basics, deltas);
    return {{"basics", basics}, {"deltas", deltas}};
  } else {
    long dt = measure([&]() {
      vns<eval, count>(Q, y_inc, y, z, z2, r_max, r_step, iters, ls_iters, N, rng, basics, deltas);
    });
    return {{"dt", dt}, {"fx", y_inc.fy}};
  }
}

int main(int argc, const char* argv[]) {
  if (argc < 3) return 1;

  const string results_filename(argv[1]);
  json results;
  mutex results_mutex;
  {
    ifstream file(results_filename, ifstream::in);
    if (file) file >> results;
  }

  /** Runs an experiment if needed, and stores the results in a JSON file. */
  auto run = [&](function<json(const ubqp&, json)> exp, const ubqp& Q, json params) {
    {
      lock_guard<mutex> lock(results_mutex);
      for (auto& j : results)
        if (j["params"] == params) return;
      cout << "<<< " << params << endl;
    }
    json result = exp(Q, params);
    {
      lock_guard<mutex> lock(results_mutex);
      cout << ">>> " << result << endl;
      results.push_back({{"params", params}, {"result", result}});
      ofstream file(results_filename, ifstream::out);
      file << setw(1) << results;
    }
  };

  const string experiment(argv[2]);

  if (experiment == "eval")
    for (string instance : {"G55"}) {
      ubqp Q = ubqp::load(instance);
      size_t n1_step = max(Q.n / 7, size_t(1));
      size_t r_step = max(Q.n / 100, size_t(1));
      for (size_t n1 = n1_step; n1 <= n1_step * 6 + 1; n1 += n1_step)
        for (auto& [eval, experiment] : {
                 pair{evaluation::basic, eval_experiment<evaluation::basic>},
                 pair{evaluation::rflip_rv, eval_experiment<evaluation::rflip_rv>},
                 pair{evaluation::s, eval_experiment<evaluation::s>},
                 pair{evaluation::a, eval_experiment<evaluation::a>},
                 pair{evaluation::c, eval_experiment<evaluation::c>},
                 pair{evaluation::m, eval_experiment<evaluation::m>},
                 pair{evaluation::ac, eval_experiment<evaluation::ac>},
                 pair{evaluation::am, eval_experiment<evaluation::am>},
                 pair{evaluation::cm, eval_experiment<evaluation::cm>},
                 pair{evaluation::acm, eval_experiment<evaluation::acm>},
             })
          for (size_t r = r_step; r <= Q.n; r += r_step)
            run(experiment, Q,
                {{"exp", "eval"},
                 {"instance", instance},
                 {"n", Q.n},
                 {"eval", static_cast<int>(eval)},
                 {"n1", n1},
                 {"r", r},
                 {"iters", 100}});
    }

  if (experiment == "ls")
    for (string instance : {"bqp250.1", "bqp500.1", "G43", "G22"}) {
      ubqp Q = ubqp::load(instance);
      size_t r_step = max(Q.n / 100, size_t(1));
      for (auto& [eval, experiment] : {
               pair{evaluation::basic, ls_experiment<evaluation::basic, false>},
               pair{evaluation::rflip_rv, ls_experiment<evaluation::rflip_rv, false>},
               pair{evaluation::s, ls_experiment<evaluation::s, false>},
               pair{evaluation::a, ls_experiment<evaluation::a, false>},
               pair{evaluation::c, ls_experiment<evaluation::c, false>},
               pair{evaluation::m, ls_experiment<evaluation::m, false>},
               pair{evaluation::ac, ls_experiment<evaluation::ac, false>},
               pair{evaluation::am, ls_experiment<evaluation::am, false>},
               pair{evaluation::cm, ls_experiment<evaluation::cm, false>},
               pair{evaluation::acm, ls_experiment<evaluation::acm, false>},
           })
        for (size_t r = r_step; r <= Q.n; r += r_step)
          run(experiment, Q,
              {{"exp", "ls"},
               {"instance", instance},
               {"n", Q.n},
               {"eval", static_cast<int>(eval)},
               {"r", r},
               {"iters", Q.n}});
    }

  if (experiment == "ls_count") {
    vector<future<void>> futures;
    for (string instance : {"bqp250.1", "bqp500.1", "G43", "G22"}) {
      auto Q = make_shared<ubqp>(move(ubqp::load(instance)));
      size_t r_step = max(Q->n / 100, size_t(1));
      for (auto pair : {
               pair{evaluation::s, ls_experiment<evaluation::s, true>},
               pair{evaluation::a, ls_experiment<evaluation::a, true>},
               pair{evaluation::c, ls_experiment<evaluation::c, true>},
               pair{evaluation::m, ls_experiment<evaluation::m, true>},
               pair{evaluation::ac, ls_experiment<evaluation::ac, true>},
               pair{evaluation::am, ls_experiment<evaluation::am, true>},
               pair{evaluation::cm, ls_experiment<evaluation::cm, true>},
               pair{evaluation::acm, ls_experiment<evaluation::acm, true>},
           })
        for (size_t r = r_step; r <= Q->n; r += r_step)
          futures.emplace_back(async([=]() {
            run(pair.second, *Q,
                {{"exp", "ls_count"},
                 {"instance", instance},
                 {"n", Q->n},
                 {"eval", static_cast<int>(pair.first)},
                 {"r", r},
                 {"iters", Q->n}});
          }));
    }
    for (auto& f : futures) f.wait();
  }

  if (experiment == "vns_figures")
    for (string instance : {"bqp100.1", "bqp250.1", "bqp500.1", "G1"}) {
      ubqp Q = ubqp::load(instance);
      size_t r_step = max(Q.n / 100, size_t(1));
      for (auto& [eval, experiment] : {
               pair{evaluation::basic, vns_experiment<evaluation::basic, false>},
               pair{evaluation::rflip_rv, vns_experiment<evaluation::rflip_rv, false>},
               pair{evaluation::s, vns_experiment<evaluation::s, false>},
               pair{evaluation::a, vns_experiment<evaluation::a, false>},
               pair{evaluation::c, vns_experiment<evaluation::c, false>},
               pair{evaluation::m, vns_experiment<evaluation::m, false>},
               pair{evaluation::ac, vns_experiment<evaluation::ac, false>},
               pair{evaluation::am, vns_experiment<evaluation::am, false>},
               pair{evaluation::cm, vns_experiment<evaluation::cm, false>},
               pair{evaluation::acm, vns_experiment<evaluation::acm, false>},
           })
        for (size_t r = r_step; r <= Q.n; r += r_step)
          run(experiment, Q,
              {{"exp", "vns_figures"},
               {"instance", instance},
               {"n", Q.n},
               {"eval", static_cast<int>(eval)},
               {"r_max", r},
               {"r_step", 1},
               {"iters", 10},
               {"ls_iters", Q.n}});
    }

  if (experiment == "vns_figures_count") {
    vector<future<void>> futures;
    for (string instance : {"bqp100.1", "bqp250.1", "bqp500.1", "G1"}) {
      auto Q = make_shared<ubqp>(move(ubqp::load(instance)));
      size_t r_step = max(Q->n / 100, size_t(1));
      for (auto pair : {
               pair{evaluation::s, vns_experiment<evaluation::s, true>},
               pair{evaluation::a, vns_experiment<evaluation::a, true>},
               pair{evaluation::c, vns_experiment<evaluation::c, true>},
               pair{evaluation::m, vns_experiment<evaluation::m, true>},
               pair{evaluation::ac, vns_experiment<evaluation::ac, true>},
               pair{evaluation::am, vns_experiment<evaluation::am, true>},
               pair{evaluation::cm, vns_experiment<evaluation::cm, true>},
               pair{evaluation::acm, vns_experiment<evaluation::acm, true>},
           })
        for (size_t r = r_step; r <= Q->n; r += r_step)
          futures.emplace_back(async([=]() {
            run(pair.second, *Q,
                {{"exp", "vns_figures_count"},
                 {"instance", instance},
                 {"n", Q->n},
                 {"eval", static_cast<int>(pair.first)},
                 {"r_max", r},
                 {"r_step", 1},
                 {"iters", 10},
                 {"ls_iters", Q->n}});
          }));
    }
    for (auto& f : futures) f.wait();
  }

  // if (experiment == "vns_tables")
  //   for (string instance : {
  //            "bqp50.1",   "bqp50.2",   "bqp50.3",   "bqp50.4",    "bqp50.5",    "bqp50.6",
  //            "bqp50.7",   "bqp50.8",   "bqp50.9",   "bqp50.10",   "bqp100.1",   "bqp100.2",
  //            "bqp100.3",  "bqp100.4",  "bqp100.5",  "bqp100.6",   "bqp100.7",   "bqp100.8",
  //            "bqp100.9",  "bqp100.10", "bqp250.1",  "bqp250.2",   "bqp250.3",   "bqp250.4",
  //            "bqp250.5",  "bqp250.6",  "bqp250.7",  "bqp250.8",   "bqp250.9",   "bqp250.10",
  //            "bqp500.1",  "bqp500.2",  "bqp500.3",  "bqp500.4",   "bqp500.5",   "bqp500.6",
  //            "bqp500.7",  "bqp500.8",  "bqp500.9",  "bqp500.10",  "G1",         "G2",
  //            "G3",        "G4",        "G5",        "G6",         "G7",         "G8",
  //            "G9",        "G10",       "G11",       "G12",        "G13",        "G14",
  //            "G15",       "G16",       "G17",       "G18",        "G19",        "G20",
  //            "G21",       "bqp1000.1", "bqp1000.2", "bqp1000.3",  "bqp1000.4",  "bqp1000.5",
  //            "bqp1000.6", "bqp1000.7", "bqp1000.8", "bqp1000.9",  "bqp1000.10", "G43",
  //            "G44",       "G45",       "G46",       "G47",        "G51",        "G52",
  //            "G53",       "G54",       "G22",       "G23",        "G24",        "G25",
  //            "G25",       "G26",       "G27",       "G28",        "G29",        "G30",
  //            "G31",       "G32",       "G33",       "G34",        "G35",        "G36",
  //            "G37",       "G38",       "G39",       "G40",        "G41",        "G42",
  //            "bqp2500.1", "bqp2500.2", "bqp2500.3", "bqp2500.4",  "bqp2500.5",  "bqp2500.6",
  //            "bqp2500.7", "bqp2500.8", "bqp2500.9", "bqp2500.10",
  //        }) {
  //     ubqp Q = ubqp::load(instance);
  //     for (auto& [eval, experiment] : {
  //              pair{evaluation::basic, vns_experiment<evaluation::basic, false>},
  //              pair{evaluation::rflip_rv, vns_experiment<evaluation::rflip_rv, false>},
  //              pair{evaluation::s, vns_experiment<evaluation::s, false>},
  //          })
  //       for (size_t r_div : {30, 60, 90})
  //         run(experiment, Q,
  //             {{"exp", "vns_tables"},
  //              {"instance", instance},
  //              {"n", Q.n},
  //              {"eval", static_cast<int>(eval)},
  //              {"r_max", (Q.n * r_div) / 100},
  //              {"r_step", 1},
  //              {"iters", 10},
  //              {"ls_iters", Q.n}});
  //   }

  if (experiment == "vns_tables")
    for (string instance : {
             "bqp50.1",   "bqp50.2",   "bqp50.3",   "bqp50.4",    "bqp50.5",    "bqp50.6",
             "bqp50.7",   "bqp50.8",   "bqp50.9",   "bqp50.10",   "bqp100.1",   "bqp100.2",
             "bqp100.3",  "bqp100.4",  "bqp100.5",  "bqp100.6",   "bqp100.7",   "bqp100.8",
             "bqp100.9",  "bqp100.10", "bqp250.1",  "bqp250.2",   "bqp250.3",   "bqp250.4",
             "bqp250.5",  "bqp250.6",  "bqp250.7",  "bqp250.8",   "bqp250.9",   "bqp250.10",
             "bqp500.1",  "bqp500.2",  "bqp500.3",  "bqp500.4",   "bqp500.5",   "bqp500.6",
             "bqp500.7",  "bqp500.8",  "bqp500.9",  "bqp500.10",  "G1",         "G2",
             "G3",        "G4",        "G5",        "G6",         "G7",         "G8",
             "G9",        "G10",       "G11",       "G12",        "G13",        "G14",
             "G15",       "G16",       "G17",       "G18",        "G19",        "G20",
             "G21",       "bqp1000.1", "bqp1000.2", "bqp1000.3",  "bqp1000.4",  "bqp1000.5",
             "bqp1000.6", "bqp1000.7", "bqp1000.8", "bqp1000.9",  "bqp1000.10", "G43",
             "G44",       "G45",       "G46",       "G47",        "G51",        "G52",
             "G53",       "G54",       "G22",       "G23",        "G24",        "G25",
             "G25",       "G26",       "G27",       "G28",        "G29",        "G30",
             "G31",       "G32",       "G33",       "G34",        "G35",        "G36",
             "G37",       "G38",       "G39",       "G40",        "G41",        "G42",
             "bqp2500.1", "bqp2500.2", "bqp2500.3", "bqp2500.4",  "bqp2500.5",  "bqp2500.6",
             "bqp2500.7", "bqp2500.8", "bqp2500.9", "bqp2500.10",
         })
      for (size_t r_div : {30, 60, 90}) {
        vector<future<void>> futures;
        auto Q = make_shared<ubqp>(move(ubqp::load(instance)));
        for (auto& pair : {
                 pair{evaluation::basic, vns_experiment<evaluation::basic, false>},
                 pair{evaluation::rflip_rv, vns_experiment<evaluation::rflip_rv, false>},
                 pair{evaluation::s, vns_experiment<evaluation::s, false>},
             })
          futures.emplace_back(async([=]() {
            run(pair.second, *Q,
                {{"exp", "vns_tables"},
                 {"instance", instance},
                 {"n", Q->n},
                 {"eval", static_cast<int>(pair.first)},
                 {"r_max", (Q->n * r_div) / 100},
                 {"r_step", 1},
                 {"iters", 10},
                 {"ls_iters", Q->n}});
          }));
        for (auto& f : futures) f.wait();
      }

  return 0;
}