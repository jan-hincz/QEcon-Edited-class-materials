
## Resource Extraction 
using Distributions, Plots, Parameters

#TAKEAWAYS (slides 16-18): NONE

@with_kw struct ResourceExtractionProblem
    n = 5 #number of possible prices 
    p_min = 1.0 # lowest price
    p_max = 10.0 # highest price

    p_vals = collect(LinRange(p_min, p_max, n))
    distribution = Categorical(1/n * ones(n))
    ϕ = pdf(distribution) #pdf of distribution of prices

    S = 50 # size of resource
    s_grid = collect(0:S) # grid for the size of the resource -> S+1 (51) possible values of resource left

    α0 = 0.1 # parameter of the reward function
    α1 = 2.25  # parameter of the reward function
    β = 0.95 # discount factor
    c = x -> α0 * x^α1 # cost function 
end


### BASIC RESOURCE EXTRACTION PROBLEM

function T(v,model) # Bellman operator
    @unpack n, p_vals, ϕ, S, c, β, s_grid = model

    v_new = zeros(S+1,n) #S+1 rows: # of possible resource size left; n columns: for each possible price level
    reward = zeros(S+1,S+1) #square matrix of lifetime profits for all possible s and s', incl. non-optimal; defined below within this function
    σ = zeros(S+1,n) #matrix of optimal x extraction sizes; each entry will correspond to a different pair of resource and price

    for (p_index,p) in enumerate(p_vals) # loop over the prices 
        for (s_index,s) in enumerate(s_grid) # loop over the sizes today 
            for (s_next_index, s_next) in enumerate(s_grid) # loop over the sizes tomorrow

                if s_next <= s # if the size tomorrow is smaller than today, we can extract today (x >=0)
                    x = s - s_next # extraction size; x>=0
                    reward[s_index,s_next_index] = p * x - c(x) + β * ϕ' * v[s_next_index,:] #slide 18; ϕ':= transpose of n x 1 column price vector ϕ
                    #v[s_next_index,:] -> for a given s' for all possible prices
                elseif s_next > s # if the size tomorrow is larger than today, we cannot extract
                    reward[s_index,s_next_index] = - Inf #penalty for violating the constraint
                end

            end 

            v_new[s_index,p_index], s_next_index_opt = findmax(reward[s_index,:]) # for each (s,p) pair, find the maximum reward and the optimal next size s'
            #s_next_index_opt; findmax(reward[s_index,:]) -> for a given size s, it will look across all possible s' to find maximal reward v(s,p) (reiterated for all p values)
            σ[s_index,p_index] = s - s_grid[s_next_index_opt] # optimal extraction size (x = s - s')
        end
    end
        
    return v_new, σ
end


function vfi(model;maxiter=1000,tol=1e-8) # value function iteration
    @unpack n, p_vals, ϕ, S, c, β = model
    v_init = zeros(S+1,n); err = tol + 1.0; iter = 1 # initial guess of v(s,p) matrix
    v = v_init
    v_history = [v_init]
    σ = zeros(S+1,n)
    while err > tol && iter < maxiter
        v_new, σ = T(v,model)
        err = maximum(abs.(v_new - v)) 
        push!(v_history,v_new)
        v = v_new
        iter += 1
    end


    return v, σ, iter, err, v_history
end


my_resource = ResourceExtractionProblem(α0 = 0.1, α1 = 2.25, β = 0.95)

v, σ, iter, err, v_history = vfi(my_resource)


plot_labels = ["p = $p" for p in my_resource.p_vals]
plot_alphas = LinRange(0.1, 1.0, length(my_resource.p_vals)) #below: higher alpha - lower transparency of a line (see plots)
plot_colors = repeat([:green],outer = length(my_resource.p_vals)) #:green repeated n = length(my_resource.p_vals) times 

plot_v = plot();
plot_σ = plot();
for p_index in 1:length(my_resource.p_vals) #drawing line for each p
    plot!(plot_v,my_resource.s_grid,v[:,p_index], label=plot_labels[p_index],linewidth=4,alpha = plot_alphas[p_index],color = plot_colors[p_index],xlabel = "size",ylabel = "v")
    plot!(plot_σ,my_resource.s_grid,σ[:,p_index], label=plot_labels[p_index],linewidth=4,alpha = plot_alphas[p_index],color = plot_colors[p_index],xlabel = "size",ylabel = "σ")
end

plot(plot_v,plot_σ,layout=(1,2),legend=:topleft)


# sample path of the size of the resource (as price changes)

# draw a random path of prices
Time = 50

price_index_path = rand(my_resource.distribution,Time) #vector of random indices (integers) of prices drawn from a distribution given at the top
price_level_path = my_resource.p_vals[price_index_path] #connecting drawn indices to prices [above: p_vals = collect(LinRange(p_min, p_max, n)]
# simulate the size of the resource
s_path = zeros(Time+1) #+1 because we'll do Time = 50 extractions, so there will be path of Time+1 resource sizes, starting from full S
s_path[1] = my_resource.S #first entry: S (full unextracted resource)

x_path = zeros(Time) #Time = 50 extractions
for t in 1:Time
    x_path[t] =  σ[Int(s_path[t]+1),price_index_path[t]] #σ:= S+1 by n matrix for each (s,p) pair
    #Int() because index has to be Integer; 
    #line 51: σ[s_index,p_index] -> line 36: (s_index,s) in enumerate(s_grid) 
    #-> line 17: s_grid = collect(0:S), so S in a S+1th row of σ matrix  
    #-> s_path[1] = S corresponds to  σ[Int(s_path[1]+1),price_index_path[1]] = σ[Int(S+1),price_index_path[1]]
    s_path[t+1] = s_path[t] - x_path[t]
end

plot_s_path = plot(1:Time,s_path[1:Time], label="s(t)",linewidth=4,xlabel = "t",ylabel = "s"); 
#s_path[1:Time] -> ommitting last Time+1th entry of s_path
plot_x_path = plot(1:Time,x_path, label="x(t)",linewidth=4,xlabel = "t",ylabel = "x");
plot_p_path = plot(1:Time,price_level_path, label="p(t)",linewidth=4,xlabel = "t",ylabel = "p");

plot(plot_s_path,plot_x_path,plot_p_path,layout=(1,3),legend=:topleft)