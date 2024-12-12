# Examples from/based on Fundamentals of Numerical Computation, Julia Edition. Tobin A. Driscoll and Richard J. Braun

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra

#TAKEAWAYS (SLIDES 16 - 22)
 
#κ := condition number >= 1 := relative change in the solution divided by relative change in data (input)
#κ <= κ(A) := cond(A) of an invertible square matrix A
#IN THE CODE κ(A) DEFINED AS κ = cond(A); 
#cond(A)*eps() = upper bound of error due to rounding (because machine epsilon =/= 0)
#condition number of 10^k means losing k digits of precision
#cond(A) = 1 for identity matrix, large for a collinear matrix and Inf for a non-invertible one
#cond(A) = 10^8 is bad (even A\b of Linear Algebra can have it) -> USE better algorithm or more bits to store precision (128, 256 etc.)
#when at terminal, ctrl+l gets you to the bottom of the terminal


### ------------------------------
### norms 

# vector norms - slide 16
x = [1.0,2.0,3.0]
norm(x,1) # L1 norm: sum of absolute values
norm(x,2) # L2 norm: length of a vector
norm(x,Inf) # infinity norm: max entry of a vector

y = [5.0,6.0,7.0]
norm(x+y,2) #(6^2 + 8^2 + 10^2)^0.5 = 14.14
norm(x,2) + norm(y,2) #14.22 greater (or equal) than above: triangular inequality 


#(Frobenius) matrix norms - slide 17

A  = [1.0 2.0; 3.0 4.0]
norm(A) # frobenius norm (2nd by default): stack entries of A into one vector and calculate its (L2) length 
norm(A,1) #1st frobenius norm: |1| + |2| + |3| + |4| = 10

#induced matrix norms - slide 17
opnorm(A) # default operator norm (2)
opnorm(A,1) # operator norm (1)
opnorm(A,Inf) # operator norm (inf)


### condition number - slide 20
A = float(I(4))
#dots instead of 0s: sparse matrix - allows to store matrices using less data!
κ = cond(A) #1 for an identity matrix = the best result we can get
rounding_bound = κ*eps() # upper bound from rounding errors - small number - good

A = [1.0 2.5; 3.25 4.125]
κ = cond(A) #~1 digit of precision lost: 8 < 10^1 
rounding_bound = κ*eps()

A = [ 1/(i+j) for i in 1:2, j in 1:2 ]
κ = cond(A) #good: ~1 digit of precision lost: 10^1 < 38.47 < 10^2
rounding_bound = κ*eps() #low number - good

A = repeat([1 2 3],3,1)
κ = cond(A) #Inf, because non-invertible matrix
rounding_bound = κ*eps() #Inf


# an example of how things are not always as they seem
A = [ 1/(i+j) for i in 1:6, j in 1:6 ]
κ = cond(A) # very large! Matrix still invertible, but difficult (around 10^8 -> around 8 digits loss of precision)
det(A) #large collinearity -> determinant very small -> Julia will be confused


#slide 22 - residual r
#let's reverse-engineer: (we KNOW x is vector 1 to 6)
x = 1:6
b = A*x #A above - very collinear

x_sol = A\b #close to true x

resid = A*x_sol - b #r; things look fine...? - be careful of the intepretation! - it's not exactly 0.0
difference = x - x_sol # things look fine...? - close to 0
relative_error = norm(difference) / norm(x) #left hand side in slide 22, small
rounding_bound = κ*eps() # small number, upper bound due to rounding errors


# perturb  the right hand side by a tiny number
Δb = randn(size(b))  #random draws from N(0,1)
Δb = 1e-10*normalize(Δb) #normalizing Δb so it has norm = 1 (default norm of order 2: length of a vector)

new_x = ((A) \ (b+Δb)) #finding new solution with perturbed b using backslash operator
Δx = new_x - x #not that small
relative_error = norm(Δx) / norm(x) #not that small, LHS of slide 22
#answers might be all over the place
println("Upper bound from κ: $(κ*norm(Δb)/norm(b))")
#so with just small permutations, relative error of x increases substantially


## another example 

A = [ 1/(i+j) for i in 1:15, j in 1:15 ]
κ = cond(A) # very large!
x = 1:15 #we know the solutions - reverse-engineering
b = A*x
x_sol = A\b
resid = A*x_sol - b # close to 0 = LOOKS fine
difference = x - x_sol # BIG DIFFERENCES - BAD

relative_error = norm(difference) / norm(x) #1.55 HUGE
rounding_bound = κ*eps() # upper bound: 60 = huge potential error

# perturb  the right hand side
Δb = randn(size(b));  Δb = 1e-10*normalize(Δb);

new_x = ((A) \ (b+Δb))
Δx = new_x - x #VERY BIG
relative_error = norm(Δx) / norm(x) #VERY BIG

println("Upper bound from κ: $(κ*norm(Δb)/norm(b))") #HUGE; upper bound of relative error of solution: RHS of slide 22