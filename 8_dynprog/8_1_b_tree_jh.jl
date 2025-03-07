
## Tree Cutting 
using Distributions, Plots, Parameters

#TAKEAWAYS (slides 8-15)
#you can compare it to mccall file, many similarities
#now: Parameters package makes unpacking parameters easier
#struct: structures show up often when you optimize your code (e.g. for speed)
#structures once run, cannot be changed -> you would have to Ctrl+D in the terminal (restarting it)
#and changing it before running it the first time

#with p = 0.1 and q = 0.1 I should cut sick tree earlier: when tree is healthy, there is 0.9 prob it will grow -> wait with cutting
#when tree is sick, it is only 0.1 prob it will become healthy and grow -> cut it quicker
#if we set r = 0, there might be a problem with contraction mapping assumptions holding
# if p is very high, a tree will be sick longer on average, smaller differences between v_H and v_S

@with_kw struct TreeCuttingProblem #Parameters package -> @with_kw : "with keywords"
    n=100 # number of possible sizes
    h = 0.1 # increment of size
    s_grid = collect(range(0.0,step = h, length = n)) # possible sizes
    S = maximum(s_grid) #max tree size = 9.9
    α0 = 0.1 # parameter of the reward function
    α1 = 0.25  # parameter of the reward function
    r = 0.05 # interest rate
    f = x -> α0 * x^α1 # reward function; increasing and concave (slides)
    c = 0.0 # cost of cutting down the tree
    p = 0.0 # probability of the tree dying / getting sick 
    q = 0.0 # probability of the recovery from sickness
end


my_tree = TreeCuttingProblem(α0 = 0.1, r=0.05)


### BASIC TREE CUTTING PROBLEM (slides 9-10)

function T(v,model) # Bellman operator
    @unpack n, s_grid, r, S, f, h = model #Parameters package -> unpack -> it will take
    #arguments named this way above from model of choice
    return [max(f(s),  1.0/(1.0+r) * v[min(s_index+1,n)]) for (s_index,s) in enumerate(s_grid)]
    #enumerate -> (s_index,s) pairs: (1,size1 = 0), (2, size2 = 0.1); v[n] = v(max S)
end

function get_policy(v,model) # this will be used after finding the fixed point of T
    @unpack n, s_grid, r, S, f, h = model
    return σ = [f(s) >=   1.0/(1.0+r) * v[min(s_index+1,n)] for (s_index,s) in enumerate(s_grid)]
    #1 if cut (f(s) >= ); 0 if don't cut
end

function vfi(model;maxiter=1000,tol=1e-8) # value function iteration
    @unpack n, s_grid, r, S, f, h = model
    v_init = f.(s_grid); err = tol + 1.0; iter = 1 # initialize with initial guess
    v = v_init
    v_history = [v_init]
    while err > tol && iter < maxiter
        v_new = T(v,model)
        err = maximum(abs.(v_new - v)) 
        push!(v_history,v_new)
        v = v_new
        iter += 1
    end
    σ = get_policy(v, model)

    return v, σ, iter, err, v_history
end

v, σ, iter, err, v_history = vfi(my_tree) #for basic model my_tree


plot_v = plot(my_tree.s_grid,v, label="v(s)",linewidth=4,xlabel = "size",ylabel = "v");
plot_σ = plot(my_tree.s_grid,σ, label="policy: 1 = cut down",xlabel = "size", linestyle=:dash,linewidth=2);
plot(plot_v,plot_σ,layout=(1,2),legend=:topleft)

anim = @animate for i in 1:length(v_history)
    plot(my_tree.s_grid, v_history[i], label="iter = $i", alpha = (i+1)/(iter+1), linewidth=4, xlabel="w", ylabel="v",ylim=[0 ,maximum(v)])
end

gif(anim, "v_history.gif", fps = 5)


### TREE CUTTING WITH CUTTING COST (slide 12)

function T(v,model) # Bellman operator
    @unpack n, s_grid, c, r, S, f, h = model
    return [max(f(s) - c,  1.0/(1.0+r) * v[min(s_index+1,n)]) for (s_index,s) in enumerate(s_grid)]
    #now f(s) - c instead of f(s)
end

function get_policy(v,model) # this will be used after finding the fixed point of T
    @unpack n, s_grid, c, r, S, f, h = model #below: now f(s) - c instead of f(s)
    return σ = [ (f(s) - c)  >=   1.0/(1.0+r) * v[min(s_index+1,n)] for (s_index,s) in enumerate(s_grid)]
end

my_tree_costly = TreeCuttingProblem(α0 = 0.1, r=0.05, c = 0.15) #c not longer 0

v_costly, σ_costly, iter_costly, err_costly, v_history_costly = vfi(my_tree_costly)


plot_v = plot(my_tree_costly.s_grid,v_costly, label="v(s) - costly",linewidth=4,xlabel = "size",ylabel = "v");
plot!(my_tree_costly.s_grid,v, label="v(s) - free",linewidth=4,color = :red,xlabel = "size",ylabel = "v");
plot_σ = plot(my_tree_costly.s_grid,σ_costly, label="policy: 1 = cut down - costly",xlabel = "size", linestyle=:dash,linewidth=2);
plot!(my_tree_costly.s_grid,σ, label="policy: 1 = cut down - free",color = :red,xlabel = "size", linestyle=:dash,linewidth=2); #base case c = 0
plot(plot_v,plot_σ,layout=(1,2),legend=:topleft)


### TREE CUTTING WITH TREE DEATH (AND CUTTING COST) (modified slide 14)

function T(v,model) # Bellman operator
    @unpack n, s_grid, c, r, S, f, h , p  = model
    return [max(f(s) - c,   1.0/(1.0+r) *  ((1 - p) * v[min(s_index+1,n)] + p * 0.5 * (f(s)-c))) for (s_index,s) in enumerate(s_grid)]
end

function get_policy(v,model) # this will be used after finding the fixed point of T
    @unpack n, s_grid, c, r, S, f, h, p = model
    return σ = [ (f(s) - c)  >=   1.0/(1.0+r) *  ((1 - p) * v[min(s_index+1,n)] + p * 0.5 * (f(s)-c)) for (s_index,s) in enumerate(s_grid)]
end

my_tree_death = TreeCuttingProblem(α0 = 0.1, r=0.05, c = 0.15, p = 0.01)

v_death, σ_death, iter_death, err_death, v_history_death = vfi(my_tree_death)
plot_v = plot(my_tree_death.s_grid,v_death, label="v(s)",linewidth=4,xlabel = "size",ylabel = "v");
plot_σ = plot(my_tree_death.s_grid,σ_death, label="policy: 1 = cut down",xlabel = "size", linestyle=:dash,linewidth=2);
plot(plot_v,plot_σ,layout=(1,2),legend=:topleft)


### TREE CUTTING WITH TREE SICKNESS AND RECOVERY (code robust to setting c>0) (modified slide 15)

function T(v,model) # Bellman operator
    # note - now it takes a matrix v (n by 2) as input
    # the first column is the value of being healthy
    # the second column is the value of being sick
    @unpack n, s_grid, c, r, S, f, h , p, q  = model

    v_H = [max(f(s) - c,   1.0/(1.0+r) *  ((1 - p) * v[min(s_index+1,n),1] + p * v[s_index,2])) for (s_index,s) in enumerate(s_grid)]
    v_S = [max(f(s) - c,   1.0/(1.0+r) *  ((1 - q) * v[s_index,2] + q * v[min(s_index+1,n),1])) for (s_index,s) in enumerate(s_grid)]
#v[s_index,2] -> for 2nd column of the input: value of being sick
    return hcat(v_H,v_S)

end

function get_policy(v,model) # this will be used after finding the fixed point of T
    @unpack n, s_grid, c, r, S, f, h, p, q = model

    σ_H  = [ (f(s) - c)  >=   1.0/(1.0+r) *  ((1 - p) * v[min(s_index+1,n),1] + p * v[s_index,2]) for (s_index,s) in enumerate(s_grid)]
    σ_S  = [ (f(s) - c)  >=   1.0/(1.0+r) *  ((1 - q) * v[s_index,2] + q * v[min(s_index+1,n),1]) for (s_index,s) in enumerate(s_grid)]

    return hcat(σ_H,σ_S)
end

function vfi(model;maxiter=1000,tol=1e-8) # value function iteration
    @unpack n, s_grid, r, S, f, h = model
    v_init = [f.(s_grid) f.(s_grid)]; err = tol + 1.0; iter = 1 # initial guess: same in both columns (for both healthy and sick tree)
    v = v_init
    v_history = [v_init]
    while err > tol && iter < maxiter
        v_new = T(v,model)
        err = maximum(abs.(v_new - v)) 
        push!(v_history,v_new)
        v = v_new
        iter += 1
    end
    σ = get_policy(v, model)

    return v, σ, iter, err, v_history
end

my_tree_sick = TreeCuttingProblem(α0 = 100.00, α1 = 2.0, r=0.01, c = 0.0, p = 0.1, q = 0.1)

v_sick, σ_sick, iter_sick, err_sick, v_history_sick = vfi(my_tree_sick)


plot_v = plot(my_tree_sick.s_grid,v_sick, label=["v(s) - healthy" "v(s) - sick"], linewidth=4,xlabel = "size",ylabel = "v");
plot_σ = plot(my_tree_sick.s_grid,σ_sick, label=["σ(s) - healthy" "σ(s) - sick"],xlabel = "size", linestyle=:dash,linewidth=2);
plot(plot_v,plot_σ,layout=(1,2),legend=:topleft)
#with p = 0.1 and q = 0.1 I should cut sick tree earlier: when tree is healthy, there is 0.9 prob it will grow -> wait with cutting
#when tree is sick, it is only 0.1 prob it will become healthy and grow -> cut it quicker
#if we set r = 0, there might be a problem with contraction mapping assumptions holding
# if p is very high, a tree will be sick longer on average, smaller differences between v_H and v_S
