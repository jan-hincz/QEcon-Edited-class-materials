##ctrl+enter to run a line of code

##alt+j alt+o to open Julia REPL terminal
##cd("path") := change directory, follow syntax from pwd() := print working directory (you can also look up whether you are in a good folder)

##Julia Package Manager
#"]" to open Julia package manager; CHECK IT EVERY DAY: blue package manager should be named after the current folder (working in local environment)
#if not, remember to activate your local environment ["activate ."] in the current directory and, if needed, add the package Plots ["add Plots"]
#if you want to clone a repo and download all its required packages, use "instantiate" command
# "status" shows environment dependencies and packages of a project
#"remove XYZ" to remove XYZ package from the environment
#"help" gives all the commands with short description
# backspace to come back from Julia package manager to Julia REPL terminal

##If some code is in the function you can always take it out of the function, test it for yourself!


using Plots #necessary to use an external package

# 1.Defining variables: a variable is a name that is bound to a value
x = 1
y = [1, 2, 3] #rows separated by , -> so this is a 3x1 column vector
println("y: ",y)


# 2. Beware! Binding vs. copying values. "=" performs only binding of values to variables
z       = y #we define z by y (NOT VICE VERSA), now z and y are the same vectors [1,2,3] and changes to either z or y, apply to both!
z[2]    = 3 #now 2nd element of z is 3, and because we coded z = y above, z = y = [1,3,3] (works both ways)
z # [1,3,3]
y # [1,3,3]


y       = [1, 2, 3]
z       = copy(y) #changes applied to z won't apply to y anymore
z[2]    = 3
z # [1,3,3] as before
y #intact: [1,2,3]
z = similar(y) #y = [1,2,3] is a 3x1 column vector with only integers {Int64}, so z will also be a 3x1 column vector with random integers
# 3. Types: The value 64 in Int64 implies that it take up 64 bits of memory
typeof(1) #Int64
typeof(true) #Bool
typeof("Hello world!") #String
typeof(0.1) #Float64


# 3a. Types: Vectors are 1-dimensional arrays with parameters referring to the type of elements
#Matrices are 2-dimensional arrays with parameters referring to the type of elements
# arrays store the dimensions of an array
typeof([1.0, 2.0, 3.0]) #Vector (1-dim) with floats
typeof(y) #y = [1,2,3] : Vector (1-dim) with integers
typeof([1.0 2.0 3.0; 2.0 3.0 4.0]) #2x3 matrix (2-dim) with Floats
[1.0 2.0 3.0; 2.0 3.0 4.0]

# 3b. Types: Julia is a dynamically typed language, so it does not need to know the types bound to variables during compile time
#    NOTE:  It is possible, though not recommended to bind values of different types to the same variable name!
typeof(x) #x = 1 -> Int
x = 1.0
typeof(x) #float


# 4. arithmetic operations
x = 5
y = 3

x+y
println("x + y = ",x + y)
println("x - y = ",x - y)
println("x * y = ",x * y)
println("x / y = ",x / y) #this is how standard division is done
println("x ^ y = ",x ^ y) #5^3 = 125
println("x ÷ y = ",x ÷ y) #truncate to an integer: x/y = 5/3 = 1.666 -> 1 IMPORTANT: YOU CAN COPY THIS SIGN FROM HERE
println("x % y = ",x % y) #modulo operator - returns the reminder of a division 5/3 -> 5 = 1*3 + 2 -> modulo is 2


# 5. Conditional evaluation: In Julia, conditional expressions can be written using the if-elseif-else-end syntax
#every "if" has to have its own "end"

x = 5
y = 1
if x % y == 0 # ==, not = when a logical statement
    println("no reminder")
elseif  x % y > 0
    println("some reminder")
else
    println("unexpected condition") #modulo is > 0 for x,y > 0, so that would indicate a bug
end

if x > 0
    sqrt(x)
else
    sqrt(-x)
end

#1-line syntax for if-else-end
x % y==0 ? println("no reminder") : println("some reminder")
x > 0 ? sqrt(x) : sqrt(-x)

if x < 0 
    println("negative x: ", x)
elseif  x  > 100
    println("Large, positive x: ", x)
else
    println("Positive, but not large x: ", x)
end


# 5. Boolean operators: 
typeof(!(x==4)) #Bool : logical statement: x not equal to 4
typeof(true) #Bool
x = 4
!(x==4) #false, because x equals 4
x = 3
!(x==4) #true
x > 0 && x < 10 #conjunction (intersection), 3 = x e (0,10), so true
0<x<10 ##this and the above statement are the same; true
x < 0 || x > 10 ##alternative; false


#############################        CONCEPT CHECK:         ############################# 
# a. Make two variables, var_a and var_b. Put any numeric types in these variables.
# b. Print out "It is easy!" if var_a is greater than 1 and var_b is NOT less than 2. Do it using nested if conditions
# c. Now write only one if condition, use boolean operators.
####################################################################################### 
#a.
var_a = 21
var_b = 37

#.b
if var_a > 1
    if var_b >= 2
        println("It is easy!")
    end
end

#c.
if var_a > 1 && !(var_b < 2)
    println("It is easy!")
end

# Julia evaluates only as many conditions (starting from the leftmost) as are needed to determine the logical value of the whole expression
x = -7
# the first condition x < 0 is true, so Julia never checks the second condition (because after || it knows the alternative is true)
x < 0 || log(x) > 10
x = 3
#the second part of the expression does not have to produce a Bool value
iseven(x) || println("x is odd") #3 is not even, so it checks and does the second part (which is not a logical statement)
iseven(x) || !iseven(x) #both are logical statements -> Bool is given (true)

if !iseven(x)
    println("x is odd")
end

x = 4
iseven(x) || println("x is odd") #first part fulfills the alternative -> shows true


isdir("some_folders") || mkpath("some_folders") #either there is a "some folders" folder at our current directory [then "true"] or Julia creates one

#using an expression that does not produce a Bool value in a normal if-end condition is not allowed -> POSSIBLE ERROR 
#x = 3
#if iseven(x) || println("x is odd")
#    println("It either works or not...")
#end


### Functions ###
### (x,y) in the function below are arguments
function first_functionA(x,y)
    return x*y #generally: put "return" at the end of the function, just before "end"
end
first_functionA(3,2)
# You can also define a simple function this way:
first_functionB(x,y) = x*y 
first_functionB(3,2)

# You can define a new function that will use another function but ALWAYS evaluate it at a particular value
second_function(x) = first_functionB(x,3)

# 2 becomes a DEFAULT value for y, though you can still change it when calling the function
function third_function(x,y=2)
    return x*y
end
third_function(3)
third_function(3,2)
third_function(3,3)

# Define a function for a line
function fourth_function(x,a,b)
    return a*x+b
end

fourth_function(2,2,1)

# Sometimes it is useful to have parameters defined as KEYWORD arguments (the ones after ";")
# you only need one ";", you can write it at the beginning too, making all arguments keyword arguments

function fifth_function(x; a, b) #x is not keyword, a and b are keyword
    return a*x+b
end

fifth_function(2,2,1) # ERROR, you need to provide the names of keyword arguments!

fifth_function(2, a=2, b=1) #now it works - keyword arguments were defined. You don't need to use ';" here

# You can combine both keyword arguments and default values:
function sixth_function(x; a=2, b=1)
    return a*x+b
end

sixth_function(2, a=2, b=0) 
sixth_function(2) #2*2+1 = 5; it works (no error), because keyword arguments were defined within the function code

# You can define a "squared" function in either of these ways:
function squared(x)
    return x^2
end
squared(x) = x^2
plot(squared,-10:10) #less smooth than one below, because default value of linearly interpolated intervals is 1
#just squared, not squared(x), because endpoint x are given (-10 and 10)

plot(squared,-10:0.1:10) #calculates endpoints of 0.1-length intervals and linearly interpolates them 

# Short syntax for defining simple functions
times_two(x) = 2 * x
plot!(times_two,-10:0.1:10) #! -> former plot (y1) was added to the current (y2) 
plot(times_two,-10:0.1:10)
plot!(times_two,-10:0.1:10) #now it is 2 coinciding 2x graphs - order matters!

#############################        QUICK TASK:         ############################# 
# Consider the following polynomial function g(x, α, β, γ, δ) = α*x^3 + β*x^2 + γ*x + δ
# a. Write it in Julia such that α, β, γ, δ are keyword arguments. Test for arbitrary values of all parameters. 
# b. Now write a function h(x) that accepts only a value x but evaluates g at the coefficients 4, -3, 2, and 10.
# c. Plot function h at the interval (-100, 100), using a 0.1 step size
####################################################################################### 
# NOTE: the way to write Greek letters is to start typing: \alpha, \beta etc.

#a.
g(x; α, β, γ, δ) = α*x^3 + β*x^2 + γ*x + δ

g(3, α = 4, β = 5, γ = 6, δ = 7)
g(1, α = 2, β = 3, γ = 4, δ = 5)
#b.
h(x) = g(x; α = 4, β = -3, γ = 2, δ = 10) #h(x) is a transformation of g(x), which uses keyword arguments, so we have to write α = 4 [not just 4], because otherwise ERROR
h(2)
#c.
plot(h,-100:0.1:100)


function seventh_function(x)
    a = x^2
    b = 2 * a
    return  a, b #"return" just before the "end"
end

seventh_function(2) #(4,8)

solution_7 = seventh_function(2)
solution_7.a #ERROR
a #ERROR

function eighth_function(x)
    a = x^2
    b = 2 * a
    return (; a, b)  #both a and b are keyword; brackets needed
end

eighth_function(2) # (a = 4, b = 8): now more explicit form

solution_8 = eighth_function(2)
solution_8.a #4; now with keyword arguments it's possible to extract a single number from the tuple

a #still ERROR
(; a, b) = eighth_function(2) #with keyword arguments we can separate tuple into standalone variables
a #4; finally works

# Often you will see an exclamation mark (!) at the end of the function name
x = [5, 1, 3, 2,1000]
sort(x) #sorted
x #not sorted
sort!(x) # Julia recommends that developers add ! at the end of functions they create if those functions modify their arguments [here: x]
x #now sorted


### Loops ###
for i in [1,2,3,4,5]
    println(i)
end

for i in 1:5
    println(i)
end

sum = 0 #initializing sum before loop
for i in 1:5
    sum = sum + i
    println("sum: ", sum)
end
println("1+2+3+4+5=", sum)

for i in 1:2:5 #from 1 to 5 with step(s) of 2 - so also 3
    println(i)
end

for i in 5:-1:1
    println(i)
end

for i in [1,2,3,4,5]
    if !iseven(i) # is (not) even
        println(i, " is odd")
    end
end

i = 1 #initialising an interesting loop - SEE THAT IT IS OUTSIDE THE LOOP
while i <= 5
    println(i)    
    global i += 1 #global -> it increases i by 1 globally (outside the loop)
end

i = 1
while i <= 5 
    global i += 1 #it adds 1 to 1 (2) before printing, so we don't get "1 is odd"
    if !iseven(i)
        println(i, " is odd")
    end
end

i = 1
while i <= 5 
    println(i)    
end #just 1 produced indefinitely, because there is no adding to i


#############################         CONCEPT CHECK:           ############################# 
# Write a function that takes a number (n) as an argument and returns the mean of the values 1, 2, 3, ..., n.
# 1. Define a function my_mean(n)
# 2. Define a variable my_sum=0
# 3. Use a for loop to get the sum of numbers from 1 to n
# 4. Then use the calculated sum to get the mean
# 5. Return the mean
# 6. Test the function

function my_mean(n)
    my_sum = 0 #initialising the sum
    for i in 1:n
        my_sum += i #or my_sum = my_sum + i [adding i to my_sum] 
    end
    return my_sum/n #just before the final "end", outside the internal loop, because you want to get final my_sum/n
end

my_mean(0) #NaN
my_mean(1) #1
my_mean(2) #1.5
my_mean(3) #2


### Arrays and matrices ###
# We have already seen a Julia array and array indexing in action
y = [1, 2, 3]
y = [1.0, 2.0, 3.0]

y[3]    # third element of an array
y[end]  # the last element of an array

# Other ways to initialize a vector
n = 10

vec = Vector{Float64}(undef, n) #initilising a vector with n random numbers (IF YOU SET n, THE BELOW LINES WILL STILL WORK)

vec = zeros(n) #n zeroes
vec = ones(n) #n ones
vec = rand(n) #n random uniform values [default: between 0 and 1]
vec = rand(1:10,n) #n random integers from 1 to 10 [with that syntax, the default step is 1]
vec = randn(n) #n random values from a standard Normal
vec = collect(1:n) #integers from 1 to n (default step is 1)

typeof(vec) #vector is just an alias for a 1-dimensional array
typeof(vec) == Array{Float64, 1} #true 
size(vec) # The syntax (10,) displays a tuple containing one element – the size along the one dimension that exists

vec = collect(1:0.5:n) #from 1 to n with steps of 0.5
size(vec) #(19,)

y = [1 2 3 ; 4 5 6]

ndims(y) #matrices are 2-dimensional arrays
y_size = size(y) #(2,3)
y[2,3] #6
y[end,end] #6

#Extracting columns and rows!!!
y[1,:]   # only the first row: in a (vertical) vector form, even though it reads [1 2 3] as a matrix
y[:,end] # only the last column

z = [1 2 3]
typeof(z) #Julia: 2-dimensional matrix

f = [1;2;3]
typeof(f) #Julia: 1-dimensional vector

n = 10
mat = Matrix{Float64}(undef, n,n) ##initilising n by n matrix with random numbers (IF YOU SET n, THE BELOW LINES WILL STILL WORK)

mat = zeros(n,n) #n by n zeroes
mat = ones(n,n) #n by n ones
mat = rand(n,n) # random standard uniform [between 0 and 1] values
mat = randn(n,n) # random values from a standard Normal

fill(5, 2, 3) #2 by 3 matrix is filled with 5s

typeof(mat) == Array{Float64, 2} #true

#############################        CONCEPT CHECK:       ############################# 
# Let's do a multiplication table
# Write a function that does the following:
#   Accepts n, which is the maximum value of a times table.
#   Returns a matrix of size n by n, where the entries of the matrix are the product of the indices of that array.    
# I.e. for n=5, I want to see a matrix where the [3,2] entry is 3 * 2 = 6

# HINTS: 
# 1. Initialize the Matrix with one of the commands we've just discussed
# 2. Use two nested for loops
# 3. M[i,j] will give you element from the i-th row and j-th column
####################################################################################### 

function multiplication_table(n)
    M = Matrix{Float64}(undef, n,n)
    for i in 1:n
        for j in 1:n
            M[i,j] = i*j
        end
    end
    return M
end

multiplication_table(4)
multiplication_table(4)[3,2] #6; entry from 3rd row and 2nd column 


# Broadcasting
# In Julia, definitions of functions follow the rules of mathematics
x = [1 2 3]
size(x) # matrix (1,3)
y = [1, 2, 3] #3-element vector 
x*y # 14; the "*" follows matrix multiplication rules (1,3)*(3,1) --> (1,1)

x'
transpose(x)
x*x' #14 as before
x'*x #still works; 3 by 1 matrix x 1 by 3 matrix -> 3 by 3 matrix

# How should we multiply two vectors element-wise?
y = [1,2,3]
x = [2,2,2]

# x*y <- You get an error, as multiplication of a vector by a vector is not a valid mathematical operation.

# Instead, we need to broadcast the multiplication. In Julia, adding broadcasting to an operator is easy. You just prefix it with a dot (.), like this:
y .* x #[1*2,2*2,3*2]

# NOTE: the dimensions of the passed objects must match [the exception under line 491]:
y = [1,2,3]
x = [2,2]
y.*x # error

y = [1,2,3]
x = [2,2,2]

# we can get the same result with a simple loop - NOTE: in Julia loops are fast
z = similar(y) #creates the object with the same size and type as y (3-element Vector{Int64}) filling it with random integers)
for i in eachindex(y, x)
    z[i] = y[i] * x[i] 
end
z # [1*2,2*2,3*2] = [2,4,6] as before

# using map - result as above
map(*, x, y)  #The passed function (in this case: * := multiplying) is applied iteratively element-wise to those collections (x and y I guess) until one of them is exhausted

# Broadcasting Functions: [I want to apply a function to a vector] - slide 2 of 2.broadcasting
times_two(x) = 2 * x
times_two(5)
times_two.(y) # [1*2,2*2,3*2] = [2,4,6] as before; dot to get a function for each element of a vector (y = [1,2,3])
# those can be built-in functions too
log.(y) #ln for [1,2,3]

# or using map function
map(x -> 2*x, y) #1st argument of map() is a function (mapping): from some x to 2x [it disregards that we defined x before]; this mapping is applied to y we defined
# [1*2,2*2,3*2] = [2,4,6] as before


# Expanding length-1 dimensions in broadcasting - slide 3 of 2.broadcasting

# There is one exception to the rule that dimensions of all collections taking part in
# broadcasting must match. This exception states that single-element dimensions get
# expanded to match the size of the other collection by repeating the value stored in
# this single element:

[1, 2, 3] .- 1 #subtracting 1 element-wise
[1, 2, 3] .- 2

mat_ones = ones(3,3) #slide 4 of 2.broadcasting: matrix vs vector
vec_horizontal = [0.0 1.0 2.0]
mat_ones .+ vec_horizontal # dot allows it once again; values added to ones horizontally

vec_vertical = [0.0, 1.0, 2.0] #slide 5 of 2.broadcasting: vector vs vector [1 column vs. 1 row]
vec_vertical .+ vec_horizontal

mat_ones .+ vec_vertical # dot allows it once again; values added to ones vertically

#via analogous multiplication you can get multiplication table
#############################        CONCEPT CHECK:         ############################# 
# Multiplication table returns!
# Write a function that does the following:
#   Accepts n, which is the maximum value of a times table.
#   Returns a matrix of size n by n, where the entries of the matrix are the product of the indices of that array.    
# I.e. for n=5, I want to see a matrix where the [3,2] entry is 3 * 2 = 6

# BUT!
# The body of the function can contain only two lines of code:
# 1. Initialize the array containing values 1 to N (see around line 375 for hints)
# 2. Use vector operations (transpose) & broadcasting to get the multiplication table. Do not use any loops.
####################################################################################### 

function multi_table(n)
    vec_vert = collect(1:n)
    return vec_vert .* vec_vert'
end

multi_table_1 = multi_table(4)
multi_table_1[3,2]


# Conditional extraction
a = [10, 20, 30]
b = [-100, 0, 100]

a .> 0 # 1 if this particular element of vector a is greater than 0, 0 otherwise
b .> 0

sum(b .> 0) # How many of the elements in the array b that are greater than 0? This will sum 1s and 0s.
sum(a .> 0) #3
b[b .> 0] # Extract only those elements of b which are greater than 0
a[a.>0]

a = [10, 20, 30]
b = [10, 0, 100]
a .== b # which element of a is equal to the corresponding element of b?
# Now we extract only the elements of an array that satisfy a condition
a[a .== b]

a = randn(100)
a[a .> 0]


#### Tuples
# Note in Julia it matters whether we use "()" or "[]"
# good for storing different datatypes
my_tuple_1 = (10, 20, 30) #() -> tuple
my_tuple_1[2] #20
my_tuple_1[2] = 4 #ERROR -> they are not editible
my_tuple_2 = (12, "hello?", "Bernanke, Diamond, Dybvig")

my_tuple_2[2] #"hello?"
my_tuple_2[3] #"Bernanke, Diamond, Dybvig"
my_named_tuple = (α = 0.33, β = 0.9, r = 0.05)
my_named_tuple.α #0.33