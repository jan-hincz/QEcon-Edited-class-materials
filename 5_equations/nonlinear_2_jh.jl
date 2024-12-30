using PrettyTables, Plots, LaTeXStrings, LinearAlgebra, NLsolve

#TAKEAWAYS (SLIDES 46 - 49) - solving methods for nonlinear multivariate functions   

#packages: Roots for a univariate function, NLsolve more general - use it for more complicated tasks
#1. GENERAL LESSON: NO GUARANTEE A NON-LINEAR JULIA FUNCTION WILL FIND ANY ROOTS
#2. ALWAYS VERIFY THAT THIS IS THE CASE: "Convergence: true"/"Converged to .." etc.
#3. NO GUARANTEE JULIA WILL FIND ALL ROOTS
#4. TRY MANY STARTING POINTS

#TO TRY FINDING ALL SOLUTIONS LOOK AT CONTOUR AND START WITH INITIAL GUESSES CLOSE TO A GIVEN INTERSECTION
#r = nlsolve(f, dfdx, initial_x,method=:newton,store_trace=true,extended_trace=true)

#lines ca. 57-75: 
#sts you have to use the Julia panel on the left to dig where a given object you'rea searching for is stored



# Define a function f: R^2 -> R^2 
#2 inputs x[1], x[2]; 2 outputs: calculation of each separated by ";"
#write square brackets in this convention to ensure multivariate function code has beginning and end
f(x) = [
    (x[1]^2 + x[2]^2)^2 - 2 * (x[1]^2 - x[2]^2);
    (x[1]^2 + x[2]^2 - 1)^3 - x[1]^2 * x[2]^3
]

# Animation of the surfaces
n = 100
animation = @animate for i in range(0, stop=2π, length=n)
    surface(-1:0.1:1, -1:0.1:1, (x, y) -> f([x, y])[1], st=:surface, c=:blues, legend=false)
    surface!(-1:0.1:1, -1:0.1:1, (x, y) -> f([x, y])[2], st=:surface, c=:reds, legend=false, alpha=0.5, camera=(30 * (1 + cos(i)), 40)) 
end
gif(animation, fps=50)

# Contour plots
contourplot = contour(-2:0.01:2, -2:0.01:2, (x, y) -> f([x, y])[1], c=:blues, levels=[0.0], clabels=true, cbar=false, lw=1)
contour!(-2:0.01:2, -2:0.01:2, (x, y) -> f([x, y])[2], c=:reds, levels=[0.0], clabels=true, cbar=false, lw=1)


# Derivative of f - Jacobian matrix (slide 46; rows (separated by ;): f' of functions
#columns separated by space; 2nd column: with respect to x[2])
dfdx(x) = [
    4 * x[1] * (x[1]^2 + x[2]^2 - 1)       4 * x[2] * (x[1]^2 + x[2]^2 + 1); #1st row of f' - of 1st function
    6 * x[1] * (x[1]^2 + x[2]^2 - 1)^2 - 2 * x[1] * x[2]^3    6 * x[2] * (x[1]^2 + x[2]^2 - 1)^2 - 3 * x[1]^2 * x[2]^2
]

# find one of the roots (look at the contour and pick initial_x close to one of intersections) 
initial_x = [0.5,0.5] #IMO looking at contour, initial guess could be chosen much better


r = nlsolve(f, dfdx, initial_x,method=:newton,store_trace=true,extended_trace=true) #we found top-left intersection root [-1.13.., 0.439..]
#3rd argument: initial guess, storing more memory = true
#"convergence: true" root [-1.13.., 0.439..] found after 13 iterations and evaluating function and Jacobian 14 times
#look at countourplot: to find 3 remaining roots change at r initial_x to a close one to a different given intersection


#sts you have to use the panel on the left to dig where a given object you'rea searching for is stored

#TO CONDUCT visualising consecutive iterations for each solution close to a given initial_x (look at contour)
#look at left corner -> Julia
#object r" - you can access it (defined above, stores info regarding the conducted optimization)
#"r NLsolve.SolverResults...
#we'll try to find r subobject storing consecutive iteration x (potential solution for a given initial_x)
r.method #"Newton with line-search"
r.zero #vector of solutions [-1.13.., 0.439..]
r.trace #r = nlsolve(f, dfdx, initial_x,...)
#output: f(x) values of function at subsuequent iterations;
#g(x) values of derivative at subsuequent iterations
#x: value of potential root at subsuequent iterations
#at the bottom we have last iteration and found solution x
r.trace.states #returns same thing as before
r.trace.states[1] #we can get guesses of different iterations; here: initial guess  x: [0.5, 0.5]
r.trace.states[8] #8-1 = 7th iteration (1st guess is initial: 0th iteration)
r.trace.states[8].metadata
r.trace.states[8].metadata["x"] #finally I got to x of the 7th iteration as an object


#####BEGINNING OF LESS IMPORTANT - visualisation of consecutive iterations x guesses ###################################################

x_interations = vcat([(r.trace.states[i].metadata["x"])' for i in 1:r.iterations]...)
#Julia panel on the left -> r.iterations = 13

function plot_iterations(f,x_interations)
    contourplot = contour(-2:0.01:2, -2:0.01:2, (x,y) -> f([x,y])[1], c=:blues,  levels=[0.0], clabels=true, cbar=false, lw=1);
    contour!(-2:0.01:2, -2:0.01:2, (x,y) -> f([x,y])[2], c=:reds, levels=[0.0], clabels=true, cbar=false, lw=1)

    animation = @animate for i in 1:size(x_interations,1)
        scatter!(contourplot,[x_interations[i,1]], [x_interations[i,2]], color = :blue, lab="iterations", legend=false)
    end
    
    return animation
    
end


animation = plot_iterations(f,x_interations) #f defined at the top, x_iterations: line ca. 80
gif(animation,fps = 5) #shows consecutive iterations getting closer to solution
root_1 = r.zero #saving the vector of solutions [-1.13.., 0.439..]


#FINDING REMAINING ROOTS and their respective iteration x guesses (with animation)

initial_x = [1.2,0.43] #close to top-right root intersection
r = nlsolve(f,initial_x,store_trace=true,extended_trace=true) #Convergence: true; [1.13.., 0.439..]
x_interations = vcat([(r.trace.states[i].metadata["x"])' for i in 1:r.iterations]...)
animation = plot_iterations(f,x_interations)
gif(animation,fps = 5)
root_2 = r.zero


initial_x = [-0.6,-0.43] #close to bottom-left root intersection
r = nlsolve(f,initial_x,method=:newton,store_trace=true,extended_trace=true) #Convergence: true; [-0.65.., -0.467..]
x_interations = vcat([(r.trace.states[i].metadata["x"])' for i in 1:r.iterations]...)
animation = plot_iterations(f,x_interations)
gif(animation,fps = 5)
root_3 = r.zero


initial_x = [0.6,-0.43] #close to bottom-right root intersection
r = nlsolve(f,initial_x,method=:newton,store_trace=true,extended_trace=true) #convergence: true; [0.65.., -0.467..]
x_interations = vcat([(r.trace.states[i].metadata["x"])' for i in 1:r.iterations]...)
animation = plot_iterations(f,x_interations)
gif(animation,fps = 5)
root_4 = r.zero

#END OF LESS IMPORTANT #####################################

# plot all four roots
contourplot = contour(-2:0.01:2, -2:0.01:2, (x,y) -> f([x,y])[1], c=:blues,  levels=[0.0], clabels=true, cbar=false, lw=1);
contour!(-2:0.01:2, -2:0.01:2, (x,y) -> f([x,y])[2], c=:reds, levels=[0.0], clabels=true, cbar=false, lw=1)
scatter!(contourplot,[root_1[1],root_2[1],root_3[1],root_4[1]], [root_1[2],root_2[2],root_3[2],root_4[2]], color = [:blue,:blue,:blue,:blue], lab="roots", legend=true)
#root_i defined in the less important section with respective close initial guess
#r = nlsolve(f,initial_x,method=:newton,store_trace=true,extended_trace=true)


# Consumer's 3 first-order conditions
function consumers_first_order_conditions(x, p, w, α) #3-entry vector function
    return [
        α * x[1]^(-α) * x[2]^(1 - α) - x[3] * p[1];
        (1 - α) * x[1]^(1 - α) * x[2]^(-α) - x[3] * p[2];
        p[1] * x[1] + p[2] * x[2] - w
    ]
end

# derive demand: finds x[1], x[2], x[3] for which above 3 F.O.C.s are 0
r = nlsolve( x -> consumers_first_order_conditions(x,[1.0,1.0],2.0,0.25),[1.0,1.0,0.5],method=:newton,store_trace=true,extended_trace=true)
#inputs: p vector = [1.0,1.0], w = 2.0, α = 0.25; initial guess: x vector = [1.0,1.0,0.5]
r #converged:true solution vector x = [0.5000000000000001, 1.5, 0.40296372443393863]


# suppose we want to do it many times for different prices 
function demand(p,w,α)
    r = nlsolve( x -> consumers_first_order_conditions(x,p,w,α),[1.0,1.0,0.5],method=:newton,store_trace=true,extended_trace=true)
    @assert r.f_converged == true
    return r.zero[1:3] #returns all 3 x demand entries
end

demand([1,1],2,0.25) #above as before

# we can use broadcasting to calculate the demand for many prices
#e.g. set p2 = 1.0 and find x for various p1
x=[]
p1_vec = 0.1:0.1:10.0 #grid of possible p1 prices
for p1 in p1_vec
    push!(x,demand([p1,1.0],2.0,0.25)) #p2 = 1, w = 2.0, α = 0.25
end

x = hcat(x...) #each row: vector solution for a given p1

# Plot demand curves for x[1] (p1 on y-axis, x[1] on x-axis]) 
plot(x[1,:],p1_vec,label="good 1",xlabel="quantity",ylabel="price",legend=:topleft)

#Different convention (x[1] on y-axis, p1 on x-axis)
plot(p1_vec,x[1,:],label="good 1",xlabel="price",ylabel="quantity",legend=:topleft)