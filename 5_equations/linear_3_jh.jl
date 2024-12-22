# Examples from/based on Fundamentals of Numerical Computation, Julia Edition. Tobin A. Driscoll and Richard J. Braun

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, IterativeSolvers

#TAKEAWAYS (SLIDES 23 - 28)

#A\b [Linear Algebra package] doesn't do iterative methods by default - you need package IterativeSolvers
#Iterative methods [jacobi(A,b) or gauss_seidel(A,b) etc.] only guaranteed to converge if square A is strictly diagonally dominant - slide 28 
#for big square strictly diagonally dominant A jacobi(A,b) or gauss_seidel(A,b) [IterativeSolvers] should be quicker than A\b [LinearAlgebra] (Oh 2 vs Oh 3 - slide 23)
#when choosing method to find x, compare norm(A*x - b) of the residual (x := found with a given method)

### calling iterative solvers

# create an easy example 
A = I(1000) + 0.0001*randn(1000,1000) #adding to I(1000) matrix with numbers of very low magnitude
b = randn(1000)

# check if square A is strictly diagonally dominant (approx. of the equ on slide 28 working well for above A)
all(sum(abs.(A),dims=2) .<= 2abs.(diag(A))) #true
#dims = 2 -> summing entries of a row (over columns) - see data_analysis_jh.jl line ca. 39
#all: has to be true for every row; 2abs.(diag(A))) :multiplying diagonal entry by 2


#check residual - see linear_2_jh.jl line ca. 75
x = jacobi(A, b) #finds solution x vector by iterative Jacobi method (slide 24)
norm(A*x - b) #norm of the residual, IMO small, because A is strictly diagonally dominant, so good

x = gauss_seidel(A, b) #finds solution x vector by iterative Gauss-Seidel method (slide 25)
norm(A*x - b) #also very small - good


# create a bad example 
A = I(1000) + 5 * randn(1000,1000) #square A will be very likely not a strictly dominant matrix 
all(sum(abs.(A),dims=2) .<= 2abs.(diag(A))) #false - A not strictly diagonally dominant

#check residual
x = jacobi(A, b)
norm(A*x - b) #VERY BIG - BAD

x = A\b #finding solution vector x by backslash operator of Linear Algebra
norm(A*x - b) #VERY SMALL - A\b better than iterative methods then

### time 


n = 1000:1000:10000
t_operator = []
t_jacobi = []

for n in n 
    A = I(n) + 0.0001*randn(n,n) #square A very likely strictly dominant - slide 28
    b = randn(n)
    time_operator = @elapsed for j in 1:10 
        A\b #backslash operator of LinearAlgebra
    end
    time_jacobi = @elapsed for j in 1:10 
        jacobi(A,b) #IterativeSolvers
    end
    push!(t_operator,time_operator)
    push!(t_jacobi,time_jacobi)
end

data = hcat(n,t_operator,t_jacobi)
header = (["size","time operator","time jacobi"],["n","seconds","seconds"])

pretty_table(data;
header=header,
header_crayon=crayon"yellow bold" ,
formatters = ft_printf("%5.2f",2),
display_size =  (-1,-1))

plt = plot(n,t_operator,label="operator",seriestype=:scatter)
plot!(plt,n,t_jacobi,label="jacobi",seriestype=:scatter,
xaxis=(:log10,L"n"),yaxis = (:log10,"elapsed time (s)"),
title = "Time",)

#for big square strictly diagonally dominant A jacobi(A,b) or gauss_seidel(A,b) [IterativeSolvers] should be quicker than A\b [LinearAlgebra] (Oh 2 vs Oh 3 - slide 23)