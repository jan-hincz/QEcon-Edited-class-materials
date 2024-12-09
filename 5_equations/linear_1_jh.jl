# Examples from/based on Fundamentals of Numerical Computation, Julia Edition. Tobin A. Driscoll and Richard J. Braun

using PrettyTables, Plots, LaTeXStrings, LinearAlgebra

#TAKEAWAYS (SLIDES 1 - 15)

#USE PACKAGES AND FUNCTIONS, PREFERABLY WITH PIVOTING (like LinearAlgebra and its A\b)
#use A\b := backslash operator of LinearAlgebra package to solve systems of linear equations
#? lu -> more info; check documentation of functions you don't know




### let's solve a simple system of equations 
A = [1.0 2.5; 3.25 4.125]
b = [5.5,6.75]

x_sol = A\b #backslash operator of LinearAlgebra package

# confirm the solution
A*x_sol - b #good

# let's solve the same system using the inverse - BAD IDEA (slide 4)
A_inv = inv(A)
x_sol2 = A_inv * b 

# confirm the solution
A*x_sol2 - b #almost 0: worse than A\b

# are the solutions the same?
x_sol == x_sol2 #very small difference -> false
x_sol ≈ x_sol2 #approximately equal

#NEVER USE inv(A)*b TO GET SOLUTIONS OF LINEAR EQIATONS, USE A\b INSTEAD


### let's count flops!

n = 50:50:500
t_operator = []
t_inv = []

for n in n 
    A = randn(n,n)
    b = randn(n)
    time = @elapsed for j in 1:100 # do it many times to be able to measure time
        A\b
    end
    push!(t_operator,time)
end

for n in n 
    A = randn(n,n)
    b = randn(n)
    time = @elapsed for j in 1:100 # do it many times to be able to measure time
        inv(A)*b
    end
    push!(t_inv,time)
end

data = hcat(n,t_operator,t_inv)
header = (["size","time operator","time inv"],["n","seconds","seconds"])

pretty_table(data;
header=header,
header_crayon=crayon"yellow bold" ,
formatters = ft_printf("%5.2f",2),
display_size =  (-1,-1))
#A\b (operator) much quicker than inv(A)*b

plt = plot(n,t_operator,label="operator",seriestype=:scatter)
plot!(plt,n,t_inv,label="inv(A)*b",seriestype=:scatter,
xaxis=(:log10,L"n"),yaxis = (:log10,"elapsed time (s)"),
title = "Time of matrix-matrix multiplication")


#method of Elementary operations - still worse than operator, partially hinging on them (slide 5-6)

function forwardsub(L,b) #slide 8; finding solutions for a lower triangular matrix

    n = size(L,1) #L is a lower triangular matrix, n = # of vectors (dimension nr 1) of L
    x = zeros(n)
    x[1] = b[1]/L[1,1] #slide 8 - top equation is very easy to solve
    for i in 2:n
        s = sum( L[i,j]*x[j] for j in 1:i-1 )
        x[i] = ( b[i] - s ) / L[i,i]
    end
    return x
end

function backsub(U,b) #slide 9; finding solutions for an upper triangular matrix

    n = size(U,1) #U is an upper triangular matrix
    x = zeros(n)
    x[n] = b[n]/U[n,n] #slide 9 - bottom equation is very easy to solve
    for i in n-1:-1:1
        s = sum( U[i,j]*x[j] for j in i+1:n )
        x[i] = ( b[i] - s ) / U[i,i]
    end
    return x
end


# let's test our functions 
A = rand(1.:9.,5,5) #5 by 5 matrix filled with INTEGERS from 1 to 9
L = tril(A) #to get lower triangular matrix by extracting elements of A
U = triu(A) #to get upper triangular matrix by extracting elements of A
b = rand(1.:9.,5) #5-element vector filled with INTEGERS from 1 to 9

x_L = forwardsub(L,b)
x_U = backsub(U,b)

resid_L = L*x_L - b #0: good
resid_U = U*x_U - b #0: good


# let's count flops!
n = 500:500:10000
t_1 = []
t_2 = []
for n in n 
    A = randn(n,n)
    L = tril(A)
    b = randn(n)
    time_1 = @elapsed for j in 1:10 # do it many times to be able to measure time
        forwardsub(L,b)
    end
    time_2 = @elapsed for j in 1:10 
        L\b #backslash operator
    end
    push!(t_1,time_1)
    push!(t_2,time_2)
end


plt = scatter(n,t_1,label="forwardsub",legend=false,
xaxis=(:log10,L"n"),yaxis = (:log10,"elapsed time (s)"),
title = "Time of forward elimination")

plot!(plt,n,t_2,label="operator",seriestype=:scatter)

plot!(n,t_1[end]*(n/n[end]).^2,label=L"O(n^3)",lw=2,ls=:dash,lc=:red,legend = :topleft)
#slide 7: forwardsub is O(n^3) 
#operator A\b still quicker

### ----------------
### LU factorization without pivoting (slide 13) - BEGINNING OF LESS IMPORTANT  #################################

A₁ = [
     2    0    4     3 
    -4    5   -7   -10 
     1   15    2   -4.5
    -2    0    2   -13
    ]
L = diagm(ones(4)) #diagonal matrix with 1s on the diagonal (I4)
U = zeros(4,4)

# first step 
U[1,:] = A₁[1,:]
U

L[:,1] = A₁[:,1]/U[1,1]
L

# second step 
A₂ = A₁ - L[:,1]*U[1,:]'
U[2,:] = A₂[2,:]
L[:,2] = A₂[:,2]/U[2,2]
L

# third step
A₃ = A₂ - L[:,2]*U[2,:]'
U[3,:] = A₃[3,:]
L[:,3] = A₃[:,3]/U[3,3]
L

# fourth step
A₄ = A₃ - L[:,3]*U[3,:]'
U[4,:] = A₄[4,:]
L[:,4] = A₄[:,4]/U[4,4]
L

# let's check if the factorization is correct
L*U - A₁ #correct, only zeroes

# write LU factorization without pivoting (1st part) as a function (here we write it with loops)
function my_lu_fact(A)
    n = size(A,1)
    A_ret = float(copy(A))
    for j in 1:n 
        for i in j+1:n
            A_ret[i,j] = A_ret[i,j]/A_ret[j,j]
            for k in j+1:n
                A_ret[i,k] = A_ret[i,k] - A_ret[i,j]*A_ret[j,k]
            end
        end
    end

    return A_ret
end

A = rand(1.:9.,5,5)

A_ret = my_lu_fact(A) #1st part of LU factorization without pivoting 

#additional parts of LU factorization without pivoting
L = tril(A_ret,-1) + I #tril(A,-1): takes lower triangular of A; -1 -> sets the diagonal entries to 0
#I is identity matrix, here it knows what dimension, thx to LinearAlgebra package
tril(A_ret,-1)
tril(A_ret,-2) #vs. tril(A_ret,-1): additionally sets the entries just below the diagonal to 0
U = triu(A_ret) #takes upper  triangular of any matrix

# let's check if the factorization is correct
L*U - A #close, but not quite

# let's compare (our code) with LinearAlgebra's LU factorization (WITH PIVOTING) - SLIDE 15
L_1,U_1,P = lu(A) #lu(A) := LinearAlgebra's function (LU factorization of A)

L_1 - L #not 0
U_1 - U #not 0
L_1 * U_1 - A #not 0: BUT LinearAlgebra is not wrong
P #for a specific random A we had: 4th, 1st, 3rd, 5th, 2nd row - it was permutated (PIVOTING)

#LinearAlgebra returns decomposition A~ (with pivoting), not A (avoids problems of LU without pivoting - slide 13)


#we can turn off pivoting in LinearAlgebra's lu(A) function 
L_2,U_2,P = lu(A,NoPivot())
L_2 - L #0 as expected
U_2 - U #0 as expected
L_2 * U_2 - A #0 as expected
P #1 2 3 4 5 (no pivoting)


### ----------------
#END OF LESS IMPORTANT ###############################################

### LU solve - slides 10 - 11

# combine forward and backward substitution and LU factorization

function my_lu_solve(A,b)
    A_ret = my_lu_fact(A) #our function: lina ca. 186 (LU factorization without pivoting - slide 13)
    L = tril(A_ret,-1) + I #LinearAlgebra's: tril(A,-1): takes lower triangular of A; -1 -> sets the diagonal entries to 0
    #I is identity matrix, here it knows what dimension, thx to LinearAlgebra package
    U = triu(A_ret) #LinearAlgebra
    y = forwardsub(L,b) #our function: line ca. 78
    x = backsub(U,y) #our function: line ca. 90
    return x
end

# let's test our function
A = rand(1.:9.,5,5)
b = rand(1.:9.,5)
x = my_lu_solve(A,b)
resid = A*x - b #close to 0, but not quite


# another example... 
ϵ = 1e-15; A = [1 1 2;2 2+ϵ 0; 4 14 4]; b = [-5;10;0.0]
x = my_lu_solve(A,b) #1 set of numbers
resid = A*x - b #40 as one of entries! because we divided by very small number (epsilon) -> USE PIVOTING
x = A\b #2nd set of numbers 
resid = A*x - b #zeroes as it should, A\b still better! 

L,U   = my_lu_fact(A) #very big numbers e16
L,U,P = lu(A)

##### TAKEAWAY - USE PACKAGES AND FUNCTIONS, PREFERABLY WITH PIVOTING (like LinearAlgebra and its A\b)


# let's count flops!

n = 100:100:1500
t_1 = []
t_2 = []
for n in n 
    A = randn(n,n)
    b = randn(n)
    time_1 = @elapsed for j in 1:10 # do it many times to be able to measure time
        my_lu_solve(A,b)
    end
    time_2 = @elapsed for j in 1:10 
        A\b
    end
    push!(t_1,time_1)
    push!(t_2,time_2)
end

plt = plot(n,t_1,label="my LU",seriestype=:scatter)
plot!(plt,n,t_2,label="operator",seriestype=:scatter,
xaxis=(:log10,L"n"),yaxis = (:log10,"elapsed time (s)"),title = "Time",)

#A\b STILL QUICKER