# NUR_A Notes

Notes for NUR_A

## Lecture 1

### General info

- staff
  - Dr. Marcel van Daalen
  - TA: Maria Marinichenko
  - TA: Luc Voorend

- structure
  - menti quizes during lectures
  - tutorials for algo problems -> less course load
  - individual exercises: 3 total -> account for 50% grade
  - exam: another 50% grade

- note
  - no vibe coding
  - prefer numpy var

### intro

- algo
  - steps to get from initial conditions to solution
  - examples
    - long division
    - sieve of something -> greek stuff
    - sorting stuff
    -
  - simple things in math might be difficult in computers
    - computers -> imperfect math machine

- libs
  - examples
    - scipy -> py
    - gsl -> c
  - why build from ground up
    - for more specific tasks and such

- stability, accuracy, efficiency, and convergence
  - stability
    - reliability of algo
  - accuracy
    - how close the algo result is
  - efficiency
    - how quick the algo get to the result
    - reduce steps needed for the algo to reach results
  - convergence

### numbers and arithmetics in computers

- bits, bytes, and ints
  - binary representation
    - bits: 0, 1
    - bytes: 8 bits -> 2\*\*8 int values
      - cover 0, 255 or something like -128, 127
      - 32 bit and 64 bit, and so on -> more val possible
  - int addition and multiplication
    - bin addition
      - examples
        - 00000010 + 00000001 = 00000011
        - 10000000 + 00000001 = 00000000 -> rollover -> it is a feature not a bug
      - transcribe to dec, hex, etc
      - rollover if bits limited for val
    - bin multiplication
      - examples
        - 00010110 x 00000101 = 01101110
      - like long division/multiplication

- operator
  - bit-wise
    - bit-shift
      - x << y -> x \* 2\*\*y
      - x >> y -> x // 2\*\*y
    - logic
      - and & -> 1 if both are 1, 0 otherwise
      - or | -> 1 if either one is 1
      - xor ^ -> exclusive or, 1 if only one is 1, otherwise 0
      - not ~ -> means no
      - floor div %
    - bit-wise examples
      - (7 >> 2) << 2 = (7 // 2\*\*2 ) \* 2\*\*2 = 1 \* 4 = 4.dec
      - (4 ^ 13) | 10 = (0100 xor 1101) | 1010 = 1001 | 1010 = 1011 = 11.dec
    - neg num
      - -x = ~(x-1) = ~x + 1
    - subtractions and divisions
      - subtraction <-> addition
        - y - x = y + (-x) = y + ~x
        - 3 - 1 = 0011 - 0001 = 0010 = 2.dec
      - division
        - 6 / 3 = 0110 / 0011 = 2.dec -> difficult and inefficient

- representing floats
  - sci notation
    - for dec: a E b = a \* 10 \*\* b
    - for bin: a E b = a \* 2 \*\* b
  - floats
    - bits to represent fractions
    - 0001 for float -> 1 \* 2 \*\* -1 for example
    - fraction bits
      - 1101 -> 2 \*\* -1 + 2 \*\* -2 + 2 \*\* -4
    - exponent bits
      - 0101 -> 2 \*\* 2 + 2 \*\* 0
    - precision
      - IEEE754
        - 32 bit -> single
          - sign bit: s -> 1 bit
          - exponent bit: e -> 8 bits -> 2 \*\* 8 -> -126, 127
          - fraction bit: f -> 23 bits
        - 64 bit -> double
          - sign bit: s -> 1 bit
          - exponent bit: e -> 11 bits -> 2 \*\* 11 -> 2 \*\* -1022, 2 \*\* 1023
          - fraction bit: f -> 52 bits
        - etc

- flops and reducing operations
  - FLOP -> floating point operations
    - 1 FLOP: addition, subtraction, multiplication
    - 4-16 FLOP: division
    - 6-32 FLOP: sqrt
  - why a range of FLOPs -> hardware, precision, etc
  - how to get around this
    - convert division => multiplication => a / b = c => def var k = (1/b) => a \* k = c
    - make sqrt => squared version -> sqrt(a+b) = c => a+b = c \*\* 2

- sources of numerical error
  - overflow and underflow
    - overflow
      - too large to represent
        - int: rollover -> cannot show 256 in 8 bit int -> become 0
        - float: NaN -> cannot use the appropriate exponent bit for something like 10 \*\* ? -> NaN
    - underflow
      - too small to represent
        - float: rounded to 0
        - 1 bit fraction -> min 2 \*\* -1 -> cannot show 1/4 as 1/4 < 1/2 -> 1/4 becomes 0
    - workaround
      - log space
      - factorial to summation
  - machine precision
    - inf bytes/bits not possible
    - fractional machine accuracy: epsilon_m
      - smallest number we can add to 1.0 and not get 1.0
        - example
          - 1.0 + 10 \*\* -100 -> 1.0
          - 1.0 = 1.0002 \pm epsilon_m
        - single: 10 \*\* -7
        - double: 10 \*\* -16
      - numbers cannot be represented exactly -> assume this for most cases
        - example
          - int 5 can be represented exactly, but 4.9 and 0.1 cannot
          - consequence: if (4.9 + 0.1) == 5 -> false
  - rounding, truncation, stability
    - round-off
      - accumulation of machine error
        - for N operations -> fractional error sqrt(N) \* epsilon_m
        - in practice -> N \* epsilon_m
      - fewer steps -> smaller error
    - stability
      - error magnification
      - example
        - 1.0 - 1.0 -> (1.0001 \pm epsilon_m) - (1.0004 \pm epsilon_m) = 10 \*\* -4 \pm epsilon_m
        - nearly equal number subtractions magnifies error
    - truncation
      - controllable
      - example
        - continuous summation fn -> discrete summation fn in computers

- efficiency and accuracy
  - truncation error
    - smaller number of operations -> high efficiency, low accuracy
    - larger number of operations -> low efficiency, high accuracy
  - round-off
    - more operations -> lower efficiency, lower accuracy
    - example
      - sum(a/n) vs sum(a)/n -> the latter is in general better
  - vectorization in py
    - vectorized array manipulation reduces number of operations
    - favoured over loops
    - example
      - array(a) \* n is better than sum(loop over a_i \* n)

## Lecture 2

### interpolation

- in general
  - start with some unknown f(x)
    - f(x) is only known at N points x_i -> sample points/knots
    - y = f(x)
  - assume
    - f(x) is smooth and well-behaved
    - x_i should be monotonic
    - that sample points are absolute truths -> no noise/outliers
  - use
    - M out of N points for each (x, y) -> order of interpolation order -> M-1
    - many interpolation function can fit to the sample points -> choose wisely
  - what it does
    - we have some samples points (x_i, y_i), what is (x_n, y_n) at some x = n

- bisection algo
  - start with N samples points labeled monotonically as N_i with (x_i, y_i)
  - start from the edges, assume M = 9 out of N points -> M_i with i range from 0 to 8 -> starting with edge 0 and 8
  - split by 2 -> edge_1 0 and 4, edge_2 5 and 8
  - repeat till we locate the closest edge for some N_i -> for example edge 4 and 5
  - therefore we find the idx for the M data point we want to locate between sample points idx

- linear interpolation
  - linear -> piecewise
  - say we have 2 points
    - draw a line between -> y = ax+b
    - interpolate points in between
  - failure
    - slope at inf when points are not monotonic -> x*i = x*(i+1)

- lagrange polynomial
  - unique polynomial that goes through exactly M points
  - why not use a really high order polynomial to fit
    - increased operation -> increased error
    - expensive

- neville's algo
  - general
    - def H(x) as linear transformation of some F(x) to G(x)
    - H(x) = ((x_j - x) \* F_i(x) + (x - x_i) \* G_j(x)) / (x_j - x_i)
  - how it works
    - start from M = 4 for example -> x_i = P(x_i)
    - what's the thing between x_1 and x_2 -> def as P(12) -> from M = 4, we have 3 P -> P(01) P(12) P(13)
    - keeps on going to get 2 P -> P(012) P(123)
    - and 1 P -> P(0123)
  - cavieta
    - not the fastest

- spline
  - what is
    - piecewise polynomial
    - derivative constraints -> continuous derivatives
  - example -> cubic spline
    - for x_i -> y(x_i) = a_i + b_i \* (x-x_i) + c_i \* (x-x_i)\*\*2 + d_i \* (x-x_i)\*\*3
    - for N points
      - N-1 derivative constraints
      - order \* (N-1) unknowns -> 3(N-1) unknowns for cubic
  - natural cubic spline
    - natural -> second derivative set to zero at endpoints
    - see slides for details
    - effectively solving system of equations -> more efficient
    - choice of boundary conditions can be different -> example: setting 1st order derivative to some known val
  - higher order splines -> may be problematic

- akima sub-spline
  - no continuous 2nd order derivatives
  - let the data take lead
  - add weights based off of data points
  - failure
    - if 3 consecutive points on straight line
    - mitigate by averaging -> see slides -> getting the weight W*(i-1) = W*(i+1) = 0 around point i

- comparisons
  - check slides for comparisons

- caution
  - log log scale plot/data fitting
  - investigate data first before passing into interpolation routines

- higher dim interpolation
  - bilinear interpolation
    - assume some grid to do calculations on edges
    - values inside grid interpolated from grid edges
    - linear along grid axis, but not linear in the whole operation
  - alternatives to solving for system of equations
    - transform to u-v plane from x-y plane -> scaled axis
      - u = (x-x*i) / (x*(i+1)-x_i)
      - v = (y-y*j) / (y*(j+1)-y_j)
    - interpolate along individual axis at a time
      - interpolate along x axis for some y -> x_i for the interpolated i
      - interpolate along y axis for some x -> y_j on some interpolated x_i

- beyond 2d -> even higher dim
  - linear -> grid -> cube -> hypercube etc

- gridless approach
  - bunch of points -> triangulate first
  - interpolate between the triangles

### extrapolation
