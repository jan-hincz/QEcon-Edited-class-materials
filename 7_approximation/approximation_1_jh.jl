using ApproxFun, Plots, LinearAlgebra, PrettyTables, FastGaussQuadrature


### Monomial basis example
# test function
f1(x) = log(x^2+x + 1) + 0.3*x + 0.1
# plot it to see what happens
plot(f1, -5, 5, label="f(x)", legend=:topleft, title = "our amazing function")

J = 20 #approximating from a point (^0) to (function at point)^20 (from x^0 to x^20)
nodes = LinRange(-3,3,J+1) # J+1 nodes between -3 and 3
Phi = Array{Float64}(undef,J+1,J+1) #1st column: x^0, 21st column: x^20

for j = 0:J
    Phi[:,j+1] = nodes[:].^j
end
betas = Phi\f1.(nodes) # solve the linear system: coefficients

f_hat(x, betas ,J) = sum([betas[j+1]*x^j for j = 0:J])

plot(f1, -3, 3, label="f(x)", legend=:topleft,color = :blue)
plot!(x->f_hat(x,betas,J),  -3, 3, label="approximation", legend=:topleft,color = :red)
scatter!(nodes,zeros(J+1),label="nodes", legend=:topleft,color = :black)
#pretty good approximation with exception of near the interval ends - WE'LL DEAL WITH THIS LATER
#manipulating J (now 20) doesn't really help that much - it pushes weird behaviour somewhere else)
#this is called Runge phenomenon

## let's see if the Phi matrix is nice...
vec_cond = []
for J = 1:15 
    
    nodes = LinRange(-3,3,J+1) # J+1 nodes between -3 and 3
    Phi = Array{Float64}(undef,J+1,J+1)
    for j = 0:J
        Phi[:,j+1] = nodes[:].^j
    end
    push!(vec_cond,cond(Phi))
end
pretty_table([collect(1:1:15) vec_cond],header = (["Order","Condition number"]),formatters = (ft_printf("%5.0f"),ft_printf("%12.6f")))


### Runge phenomenon
# evaluate
runge(x) = 1/(1+25x^2)
my_plot = plot(runge, -1, 1, label="f(x)", legend=:topleft,color = :blue,linewidth = 4,title = "Runge function")

function get_betas(f,a,b,J)

    nodes = LinRange(a,b,J+1) # J+1 nodes, equidistant
    Phi = Array{Float64}(undef,J+1,J+1)

    for j = 0:J
        Phi[:,j+1] = nodes[:].^j
    end
    betas = Phi\f.(nodes) # solve the linear system
end

maxJ = 20
Beta_mat = zeros(maxJ,maxJ+1)


for J = 1:maxJ
    Beta_mat[J,1:J+1] = get_betas(runge,-1,1,J)
end


my_plot = plot(runge, -1, 1, label="f(x)", legend=:topleft,color = :blue,linewidth = 4,title = "Equidistant")

for j = 2:2:10
plot!(my_plot, x->f_hat(x,Beta_mat[j,:],j), -1, 1, label="approx with J = $j", legend=:bottomleft, ylim = [-2, 2])
end
@show my_plot #higher J -> better central approximation and weirder ossicilation at the interval ends
#because nodes are equidistant (?) 




function get_betas_chebnodes(f,a,b,J) #using chebyshev nodes to do the above

    nodes =  (b-a)/2* gausschebyshev(J+1)[1] .+ (b+a)/2  # J+1 nodes, equidistant
    Phi = Array{Float64}(undef,J+1,J+1)

    for j = 0:J
        Phi[:,j+1] = nodes[:].^j
    end
    betas = Phi\f.(nodes) # solve the linear system
end

maxJ = 20
Beta_mat = zeros(maxJ,maxJ+1)


for J = 1:maxJ
    Beta_mat[J,1:J+1] = get_betas_chebnodes(runge,-1,1,J)
end


my_plot2 = plot(runge, -1, 1, label="f(x)", legend=:topleft,color = :blue,linewidth = 4,title = "Chebyshev")

for j = 2:2:10
plot!(my_plot2, x->f_hat(x,Beta_mat[j,:],j), -1, 1, label="approx with J = $j", legend=:bottomleft, ylim = [-2, 2])
end
@show my_plot2


plot(my_plot, my_plot2) #comparison - LHS: equidistant, RHS: Chebyshev nodes - much better

### Chebyshev polynomials
# this uses ApproxFun package
# first see an example of how this works

space = Chebyshev(-1..1) #from -1 to 1
runge_cheb_approx = Fun(runge,space) #ApproxFun function creating a function approximation

my_plot3 = plot(runge, -1, 1, label="f(x)", legend=:topleft,color = :blue,linewidth = 4)
plot!(my_plot3,runge_cheb_approx, -1, 1, label="approximation", legend=:topleft,color = :red,linestyle=:dash, linewidth = 4)

# without J? J chosen by package
runge_cheb_approx2 = Fun(runge,space)
ncoefficients(runge_cheb_approx2) #J = 189 = number of coefficients

# what is this one?
cheb0= Fun(space,[1])
cheb1 = Fun(space,[0,1])
cheb2 = Fun(space,[0,0,1])
cheb3 = Fun(space,[0,0,0,1])
cheb4 = Fun(space,[0,0,0,0,1])

plot(cheb0, -1, 1, label="T0", legend=:topleft,color = :blue,linewidth = 4,title = "Chebyshev polynomials")
plot!(cheb1, -1, 1, label="T1", legend=:topleft,color = :red,linewidth = 4)
plot!(cheb2, -1, 1, label="T2", legend=:topleft,color = :green,linewidth = 4)
plot!(cheb3, -1, 1, label="T3", legend=:topleft,color = :yellow,linewidth = 4)
plot!(cheb4, -1, 1, label="T4", legend=:topleft,color = :purple,linewidth = 4)

## let's see if the Phi matrix is nice...
vec_cond = []
for J = 1:15 
    space = Chebyshev(-1..1)
    nodes = points(space,J+1)
    Phi = Array{Float64}(undef,J+1,J+1)
    vec_one = [1]
    
    for j = 0:J
        cheb = Fun(space,vec_one)
        Phi[:,j+1] = cheb.(nodes)
        pushfirst!(vec_one,0)
    end
    push!(vec_cond,cond(Phi))
end
pretty_table([collect(1:1:15) vec_cond],header = (["Order","Condition number"]),formatters = (ft_printf("%5.0f"),ft_printf("%12.6f")))
#condition number = 1 = 10^0 -> losing 0 digits of precision

# investigate
J = 10
space = Chebyshev(-1..1)
nodes = points(space,J)
Phi = Array{Float64}(undef,J,J)
vec_one = [1]

for j = 1:J
    cheb = Fun(space,vec_one)
    Phi[:,j] = cheb.(nodes)
    pushfirst!(vec_one,0)
end

@show Phi'*Phi #10, 5, 5,5.. on diagonal, very close to 0 off- -> quasi-orthogonal

# extrapolation can be bad... 
# we use Chebyshev interpolation to approximate the functionn f1(x) = log(x^2+x + 1) + 0.3*x + 0.1

space = Chebyshev(-1..1)
fun1_cheb_approx = Fun(f1,space)
fun1_cheb_approx_extrapolate(x) = extrapolate(fun1_cheb_approx,x )

plot(f1, -1.5, 1.5, label="log(x^2+x + 1) + 0.3*x + 0.1", legend=:topleft,color = :blue,linewidth = 4)
plot!(fun1_cheb_approx, -1.5, 1.5, label="fun1_cheb_approx", legend=:topleft,color = :green,linewidth = 2)
plot!(fun1_cheb_approx_extrapolate,  -1.5, 1.5, label="fun1_cheb_approx_extrapolate", legend=:topleft,color = :red,linewidth = 2)
#very bad approximation outside -1 to 1