# Examples from/based on Fundamentals of Numerical Computation, Julia Edition. Tobin A. Driscoll and Richard J. Braun

using PrettyTables, Plots, LaTeXStrings #for part 2 only (around line XXX)

### PRELIMINARIES 
x_int  =  1
x_float  = 1.0

# use typeof to check the type of a variable
typeof(x_int)
typeof(x_float)



# convert integer to float
x_int_float = Float64(x_int)
typeof(x_int_float)

# can we convert back? 
x_float_int = Int64(x_float)
typeof(x_float_int)

x_float_int2 = Int64(1.02)
typeof(x_float_int2)


# representation of a number 
bitstring(1.0)
bitstring(-1.0)
x = 3.0
@show sign(x),exponent(x),significand(x);
x = -3.0
@show sign(x),exponent(x),significand(x);

big_numer = 100000000000000000000000000000000000000000000.0
typeof(big_numer)

bitstring(big_numer)
bitstring(big_numer + 1.0)




# define type of variable 
x_fl32::Float32 = 1.0
typeof(x_fl32)
x_myint::Int64 = 1.0
typeof(x_myint)
# machine epsilon 
eps(Float64)

# are Float64 equally spaced? 
eps(1.0)
eps(1000000000.0)

nextfloat(1.0)
nextfloat(1000000000.0)



# what is there between 1.0 and nextfloat(1.0)
@show (nextfloat(1.0) - 1.0)/2

@show eps()/2

@show 1.0 + eps()/2

@show (1.0 + eps()/2) - 1.0

@show 1.0 + (eps()/2 - 1.0) 

# beware! the result is off by eps()/2 --- which is the exact value  itself



# smallest and largest floating point number
@show floatmin(),floatmax();

nextfloat(floatmax())
nextfloat(-Inf)

# a mystery...
a::Float16 = 0.1
b::Float16 = 0.2
c::Float16 = 0.3 

result_1 = a + b + c
result_2 = c + b + a
@assert result_1 == result_2




### QUADRATIC EQUATION EXAMPLE
ϵ = 1e-6   
ax,bx,cx = 1/3, (-2-ϵ)/3, (1+ϵ)/3  

dx = sqrt(bx^2-4ax*cx)
r1 = (-bx-dx)/(2ax)  
r2 = (-bx+dx)/(2ax)
(r1,r2)

# what is the true r2? 
abs_error = abs(r2 - (1.0 + ϵ))
rel_error = abs_error / (1.0 + ϵ)



### STABILITY
a = 1.0
b = - (10^6 + 10^(-6))
c = 1.0 

# what are the true roots?
r1_true = 10^(-6)
r2_true = 10^6

# use quadratic formula to compute the roots
d = sqrt(b^2-4a*c)
r1 = (-b-d)/(2a)
r2 = (-b+d)/(2a)


# what are the errors?
abs_error_r1 = abs(r1 - r1_true)
rel_error_r1 = abs_error_r1 / r1_true

# condition number? 

step1 = b^2 #see slides 
cond_step_1 = abs( b * (2*b) / step1) #2.0 is not much for a condition number = good

step2 = step1 - 4*a*c
cond_step_2 = abs(  1.0 * step1  / step2) #1.00004 is fine for a condition number

step3 = sqrt(step2)
cond_step_3 = abs( 0.5 * step2 ^ (-0.5) * step2 / step3) #0.5 no problem

step4 = -b + step3 #calculating greater root (before dividing by 2a)
cond_step_4 = abs( -1.0 * step3 / step4) #0.5 fine

step5 = step4 / (2*a)
cond_step_5 = abs( 1.0 / (2*a) * step4 / step5) #1.0 good

cond_good_root = cond_step_1 * cond_step_2 * cond_step_3 * cond_step_4 * cond_step_5 #overall condition number

# what happens if we use the other root?

#hitherto steps with equivalent condition numbers as the other root
step4 = -b - step3
cond_step_4 = abs( -1.0 * step3 / step4) #very big number, NOT GOOD

step5 = step4 / (2*a)
cond_step_5 = abs( 1.0 / (2*a) * step4 / step5)

cond_bad_root = cond_step_1 * cond_step_2 * cond_step_3 * cond_step_4 * cond_step_5


# better method: exploit the fact that r1 * r2 = c/a
r1_new = c / r2 #we got a = 1; r1_new will be calculated well
#we know the first root was calculated well, so we'll find the previously ill-condition by the above formula

#that's why you should use packages first and your own functions second, because the packages were created in a way to avoid these and other problems

### SCALING
# we often have expressions like exp(a) / (exp(a) + exp(b))

a = 3999.0
b = 4000.0

bad_solution = exp(a) / (exp(a) + exp(b)) # NaN why? 
#computer cannot handle e^3999, which is bigger than 10^308 - see slides about max number a computer can handle in double precision limit

exp(a) #Inf

# better solution: use the fact that exp(a) / (exp(a) + exp(b)) = 1 / (1 + exp(b-a))
good_solution = 1.0 / (1.0 + exp(b-a))



################ PART 2

# let's count flops!

### matrix-vector multiplication
n = 1000:500:5000 #range of numbers
t = []
for n in n 
    A = randn(n,n)
    x = randn(n)
    time = @elapsed for j in 1:100 # do it many times to be able to measure time
        A*x
    end
    push!(t,time)
end

#t will be the time of operations (not visible yet)

data = hcat(n,t) #2 vectors stacked right to left; below: pretty table
header = (["size","time"],["n","seconds"]) #columns; subtitles
 pretty_table(data;
    header=header,
    header_crayon=crayon"yellow bold" ,
    formatters = ft_printf("%5.2f",2), #2 digits shown
    display_size =  (-1,-1)) #to get entire table

scatter(n,t,label="data",legend=false,
xaxis=(:log10,L"n"),yaxis = (:log10,"elapsed time (s)"),
title = "Time of matrix-vector multiplication",);

plot!(n,t[end]*(n/n[end]).^2,label=L"O(n^2)",lw=2,ls=:dash,lc=:red,legend = :topleft)

#(n_1/n_2)^2 = time_1/time_2 roughly as expected - see slides

### matrix-matrix multiplication
n = 100:100:1000
t = []
for n in n 
    A = randn(n,n)
    B = randn(n,n)
    time = @elapsed for j in 1:20 # do it many times to be able to measure time
        A*B
    end
    push!(t,time)
end

data = hcat(n,t)
header = (["size","time"],["n","seconds"])
pretty_table(data;
    header=header,
    header_crayon=crayon"yellow bold" ,
    formatters = ft_printf("%5.2f",2))

scatter(n,t,label="data",legend=false,
xaxis=(:log10,L"n"),yaxis = (:log10,"elapsed time (s)"),
title = "Time of matrix-matrix multiplication",);

plot!(n,t[end]*(n/n[end]).^3,label=L"O(n^3)",lw=2,ls=:dash,lc=:red,legend = :topleft)