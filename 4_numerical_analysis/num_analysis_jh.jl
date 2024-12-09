# ] Examples from/based on Fundamentals of Numerical Computation, Julia Edition. Tobin A. Driscoll and Richard J. Braun

TAKEAWAYS
#by representing numbers as floats with rounding errors order of operations might matter
#Float64 has 15 or 16 digits of precision
# Float64 numbers are not equally spaced!!!
# condition number = 10^k means you lose k digits of precision
#you should use packages first and your own functions second, because the packages were created to avoid problems (e.g. of ill condition)
#make sure Julia can handle the magnitude of the numbers (machine epsilon, floatmax(), floatmin() etc.)


using PrettyTables, Plots, LaTeXStrings #for part 2 only (around line 192)

### PRELIMINARIES 
x_int  =  1
x_float  = 1.0

# use typeof to check the type of a variable
typeof(x_int) #Int64
typeof(x_float) #Float64

# convert integer to float
x_int_float = Float64(x_int) #1.0
typeof(x_int_float) #Float64

#we can convert back
x_float_int = Int64(x_float) #1
typeof(x_float_int) #Int64

x_float_int2 = Int64(1.02) #Error
typeof(x_float_int2) #Error

# representation of a number 
bitstring(1.0) #0011111111110000000000000000000000000000000000000000000000000000
bitstring(-1.0) #like above, just 1 at the beginning - the sign bit
x = 3.0
@show sign(x),exponent(x),significand(x) #3 = 1.0 * 2^1 * 1.5 
x = -3.0
@show sign(x),exponent(x),significand(x); #3 = 1.0 * 2^1 * -1.5

big_numer = 100000000000000000000000000000000000000000000.0 #1.0e44 = 1.0 * 10^44
typeof(big_numer) #Float64

bitstring(big_numer)
bitstring(big_numer + 1.0) #Float64 has 15 or 16 digits of precision
#big number and big number + 1 both with over 40 digits will be represented the same


# define type of variable 
x_fl32::Float32 = 1.0
typeof(x_fl32) #Float32 - single precision
x_myint::Int64 = 1.0
typeof(x_myint) #Float64

# machine epsilon - slide 17 
eps(Float64) #2.220446049250313e-16

# Float64 are not equally spaced!!!

eps(1.0) #2.220446049250313e-16
eps(1000000000.0) #1.1920928955078125e-7
nextfloat(1.0) #1.0000000000000002
nextfloat(1000000000.0) #1.0000000000000001e9


# what is there between 1.0 and nextfloat(1.0)
@show (nextfloat(1.0) - 1.0)/2 #1.1102230246251565e-16
@show eps()/2 #1.1102230246251565e-16

@show 1.0 + eps()/2 #1.0!!!!!!!

@show (1.0 + eps()/2) - 1.0 #0.0!!!!!!!
# beware! the result is off by eps()/2 --- which is the exact value itself

@show 1.0 + (eps()/2 - 1.0) #1.1102230246251565e-16!!!!
#this time ok, even though it's the same equ as above!

#TAKEAWAY
#by representing numbers as floats with rounding errors order of operations might matter



# smallest and largest positive floating point numbers
@show floatmin(),floatmax(); #(2.2250738585072014e-308, 1.7976931348623157e308)

nextfloat(floatmax()) #Inf
nextfloat(-Inf) #-1.7976931348623157e308 = - max positive number


#################################################### NOT IMPORTANT ####################################
# a mystery...
f::Float16 = 0.1
g::Float16 = 0.2
h::Float16 = 0.3 

result_1 = f + g + h #Float16(0.5996)
result_2 = h + g + f #Float16(0.6)
@assert result_1 == result_2 #Error
#################################################### NOT IMPORTANT ####################################


### QUADRATIC EQUATION EXAMPLE
ϵ = 1e-6
ax,bx,cx = 1/3, (-2-ϵ)/3, (1+ϵ)/3 #a,b,c of a quadratic function

dx = sqrt(bx^2-4ax*cx) #delta
r1 = (-bx-dx)/(2ax) #root 1
r2 = (-bx+dx)/(2ax) #root 2
(r1,r2)

# what is the true r2? 
abs_error = abs(r2 - (1.0 + ϵ)) #1.7485035641584545e-10
rel_error = abs_error / (1.0 + ϵ) #1.748501815656639e-10


### STABILITY - quadratic equation example - slide 28
a = 1.0
b = - (10^6 + 10^(-6))
c = 1.0 

# what are the true roots?
r1_true = 10^(-6) #smaller root
r2_true = 10^6 #greater root

# use quadratic formula to compute the roots
d = sqrt(b^2-4a*c) #999999.999999
r1 = (-b-d)/(2a) #1.00000761449337e-6 - close to r1_true
r2 = (-b+d)/(2a) #1.0e6 = same as r2_true, good!

# what are the errors? - see slide 19
abs_error_r1 = abs(r1 - r1_true) #7.614493370101404e-12
rel_error_r1 = abs_error_r1 / r1_true #7.614493370101404e-6



# CONDITION NUMBERS - notes after slide 28 (quadratic roots example)
#recall: condition number = 10^k means you lose k digits of precision
#SHOWCASING PROBLEM (HERE, WITH SMALLER ROOT X2)

step1 = b^2 
cond_step_1 = abs( b * (2*b) / step1) #2.0 is small for a condition number = good

step2 = step1 - 4*a*c
cond_step_2 = abs(  1.0 * step1  / step2) #1.000000000004: small = good

step3 = sqrt(step2)
cond_step_3 = abs( 0.5 * step2 ^ (-0.5) * step2 / step3) #0.5: small = good

step4 = -b + step3 #calculating greater root (before dividing by 2a)
cond_step_4 = abs( -1.0 * step3 / step4) #0.5: small = good

step5 = step4 / (2*a)
cond_step_5 = abs( 1.0 / (2*a) * step4 / step5) #1.0: small = good

cond_good_root = cond_step_1 * cond_step_2 * cond_step_3 * cond_step_4 * cond_step_5 #overall condition number = 0.5: SMALL = GOOD

# what happens if we use the other smaller root x2?

#hitherto steps with equivalent condition numbers as the other root
step4 = -b - step3
cond_step_4 = abs( -1.0 * step3 / step4) #4.99996192781805e11: very big number = NOT GOOD

step5 = step4 / (2*a)
cond_step_5 = abs( 1.0 / (2*a) * step4 / step5)

cond_bad_root = cond_step_1 * cond_step_2 * cond_step_3 * cond_step_4 * cond_step_5 #4.9999619278380505e11: VERY BIG NUMBER: NOT GOOD
#SMALLER ROOT CALCULATION ILL-CONDITIONED HERE

# better method: exploit the fact that r1 * r2 = c/a; r2: greater root - in notes as x1
r1_new = c / r2 #we got a = 1; r1_new calculation will be well-conditioned now, r2 was already well calculated  as shown above

#TAKEAWAY
#that's why you should use packages first and your own functions second, because the packages were created in a way to avoid these and other problems



### SCALING
# we often have expressions like exp(a) / (exp(a) + exp(b))

a = 3999.0
b = 4000.0

bad_solution = exp(a) / (exp(a) + exp(b)) # NaN 
#computer cannot handle e^3999, which is bigger than 10^308 - see slide 16 about floatmax() in Float64 or line ca. 84
exp(a) #Inf

# better solution: use the fact that exp(a) / (exp(a) + exp(b)) = 1 / (1 + exp(b-a))
good_solution = 1.0 / (1.0 + exp(b-a))



#################### PART 2 - SLIDES 31 AND BEYOND #######################################################

# let's count flops!

### matrix-vector multiplication - slides 34-35
n = 1000:500:5000 #range of numbers
t = [] # Any[]: empty vector that can contain any kind of data
for n in n 
    A = randn(n,n) #filled with random numbers from N(0,1)
    x = randn(n)
    time = @elapsed for j in 1:100 # do it many times (e.g. 100) to be able to measure time
        A*x
    end
    push!(t,time) #values from time will be pushed into t (before the loop t was empty: t = [])
end
#t will be the time of operations (not visible yet)

data = hcat(n,t) #2 vectors stacked left to right; below: pretty table

header = (["size","time"],["n","seconds"]) #[columns], [subtitles]

pretty_table(data;
header=header,
header_crayon=crayon"yellow bold" ,
formatters = ft_printf("%5.2f",2), #2 decimals shown
display_size =  (-1,-1)) #to get the entire table

scatter(n,t,label="data",legend=false, #the actual t of flops, not O(n^2) approximation 
xaxis=(:log10,L"n"),yaxis = (:log10,"elapsed time (s)"), #:log10 -> scale will be in powers of 10
title = "Time of matrix-vector multiplication",); #look at the plot below

plot!(n,t[end]*(n/n[end]).^2,label=L"O(n^2)",lw=2,ls=:dash,lc=:red,legend = :topleft)
#adding the red line of O(n^2); lw: line width, ls: line style, lc: line color
#n[end] = 5000 (end of range), t[end] =+- 1.7 s (time of 5000 flops), 

#slide 35: O(n^2) -> time_2/time_1 =+- (n_2/n_1)^2  
#dots converge to red line as expected -> t[end]/t =+- (n_end/n)^2 -> t =+- t[end]*(n/n[end]).^2 


### matrix-matrix multiplication - slide 36
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
formatters = ft_printf("%5.2f",2),
display_size =  (-1,-1))

scatter(n,t,label="data",legend=false,
xaxis=(:log10,L"n"),yaxis = (:log10,"elapsed time (s)"),
title = "Time of matrix-matrix multiplication",);

plot!(n,t[end]*(n/n[end]).^3,label=L"O(n^3)",lw=2,ls=:dash,lc=:red,legend = :topleft)
#slide 36: dots converge to red line given by O(n^3): as expected