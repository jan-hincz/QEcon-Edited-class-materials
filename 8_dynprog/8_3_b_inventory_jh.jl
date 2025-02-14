## Inventory Management Problem

##QUESTIONS: line ca. 132: provided code: P[i, j] = sum((σ[i, d+1] == j+1) * pdf(ϕ, d) for d in 0:X_max, but I think it should be == j-1 

#TAKEAWAYS:
#under well-defined structure like Markov Chain GO WITH OPI (Optimistic Policy Iteration) INSTEAD OF VFI: SLIDES 17-21, especially 20-21
#OPI is much faster and uses less memory than VFI
#line ca. 128:
#for (i, j) in product(1:X_max+1, 1:X_max+1) 
    #...
#end
#product(...): IterTools function creating all possible i,j indices combos where each i,j runs from 1 to X_max+1; useful for creating matrix P[i,j] entries


# The firm faces stochastic demand D. It sells each demanded unit at a price p (unless inventory is too low to meet entire demand).  
# The firm can sell at most X units, which is the size of its inventory.

# Every period, it decides to place an order on variable F goods (action). It will be able to sell purchased goods starting from the next period.
#The firm incurs a per-unit cost c for each ordered unit and a fixed ordering cost K.  

# The inventory has an upper bound, X̄.  

# The firm maximizes the present discounted value of profits, β ∈ (0,1).  

# The Bellman equation governing this decision is (firm chooses F to max NPV of profits):

# v(X,D) = p ⋅ min{D, X} - c ⋅ F - K ⋅ 𝟙_{F>0} + β * 𝔼[v(X′, D′)] 
#        = p ⋅ min{D, X} - c ⋅ F - K ⋅ 𝟙_{F>0} + β * 𝔼[v( min{ max(X - D, 0) + F, X̄ }, D′ )] 


#current period revenue: p ⋅ min{D, X} 
#you incur cost of F purchase today, but reap benefits tomorrow
#inventory next period X' = min{ max(X - D, 0) + F, X̄ } ->
#either X̄ (max inventory level)
#or max(X - D, 0) + F: F purchased today + whatever inventory left from selling today: max(X-D,0)


# load some packages we will need today
using Distributions, QuantEcon, IterTools, Plots


function create_inventory_model(; 
    p = 4, # price per unit
    d_par = 0.7, # demand distribution parameter
    ϕ = Geometric(d_par), #Demand has discrete geometric distribution
    X_max = 30, # maximum inventory
    K = 3, # cost of placing an order
    c = 1, # cost of unit ordered
    β = 0.99, # discount factor
    X_vec = 0:X_max # vector of possible inventory levels

    )
   
    return (; p, ϕ, X_max, K, c, β,X_vec)
end

function T_operator(v,model)

    (;p, ϕ, X_max, K, c, β, X_vec) = model

    v_new = similar(v)
    σ_ind_new = zeros(Int64,length(X_vec),length(X_vec)) #N by N matrix (possible today's X and D), holds indices (Int numbers)
    σ_new = zeros(length(X_vec),length(X_vec)) #N by N; N = X_vec +1 (# of elements from 0 to X_vec)

    for (d_ind, d) in enumerate(X_vec) #going over all possible levels of demand - exogeneous state variable; X_vec -> indices from 1 to X_max+1 (above X_vec = 0:X_max)
        for (x_ind, x) in enumerate(X_vec) #going over all today's possible levels of inventory X 
            #(future) X is an endogeneous state variable
    
            RHS_vec = zeros(length(X_vec)) # v(X,D) = p ⋅ min{D, X} - c ⋅ F - K ⋅ 𝟙_{F>0} + β 𝔼[v( min{ max(X - D, 0) + F, X̄ }, D′ )] 
            for (x_next_ind, x_next) in enumerate(X_vec) #considering all possible inventory levels tomorrow
                
                sold = min(d,x) #line ca. 15: real current revenue
                revenue = p * sold - K * (x_next > (x - sold)) - c * (x_next - (x - sold)) #(x_next > (x - sold)): logical statement returning 1 if F>0 or 0 otherwise 
                #line 15: rev = p ⋅ min{D, X} - K ⋅ 𝟙_{F>0} - c ⋅ F; F = x_next - (x - sold)
                if x_next >= x - sold #F >= 0 
                    RHS_vec[x_next_ind] = revenue + β * sum( v[x_next_ind,d_next_ind] * pdf(ϕ,d_next_ind-1) for d_next_ind in 1:X_max+1 )
                else #expected future lifetime profits v(X',D') -> we need pdf of all possible demands (geometric distribution)
                    RHS_vec[x_next_ind] = -Inf #I am not allowing the violation of above in the problem
                end
    
            end
    
            v_new[x_ind,d_ind], σ_ind_new[x_ind,d_ind] = findmax(RHS_vec) #findmax() -> returns value as v_new and index as v
            σ_new[x_ind,d_ind] = X_vec[σ_ind_new[x_ind,d_ind]] #σ_new[x_ind,d_ind] := X'(X,D)
            #policy outcome: future inventory X' as function of current X,D
        end
    end
    return v_new, σ_new, σ_ind_new
end

    
function vfi(model; tol = 1e-6, maxiter = 1500) #code equivalent to former vfi examples, 1000 wasnt enough iter here,too big error

    (;p, ϕ, X_max, K, c, β, X_vec) = model
    
    error = tol + 1.0; iter = 1 #  initialize
    v = zeros(X_max+1,X_max+1); #as X_vec runs from 0 to X_max (X_max + 1 entries)
    while error > tol && iter < maxiter
        v_new = T_operator(v,model)[1] #T_operator() defined above returns v_new, σ_new, σ_ind_new
        #[1] -> will get v_new from T_operator()
        error = maximum(abs.(v_new .- v))
        v = v_new
        iter += 1
    end
    # one more iteration to get the policy function
    v, σ = T_operator(v,model)[1:2]
    return v, σ, iter, error
        
end

model = create_inventory_model()

v, σ, iter, error = vfi(model)
iter #1389 < 1500, good, error lower than our tolerance then (stopping criterion)
    
(;p, ϕ, X_max, K, c, β, X_vec) = model #in our example D is independent from period to period
plot(X_vec, v[:, 3], label="Demand 3") #v(X,D)
plot!(X_vec, v[:, 6], label="Demand 6")
plot!(X_vec, v[:, 9], label="Demand 9", xlabel="Inventory", ylabel="Value Function")

plot(X_vec, σ[:, 3], label="Demand 3") #policy in terms of X'
plot!(X_vec, σ[:, 6], label="Demand 6")
plot!(X_vec, σ[:, 9], label="Demand 9", xlabel="Inventory", ylabel="Inventory next period")
#higher D -> you should incr. your inventory to be able to sell more


#Markov chain approach
P = Matrix{Float64}(undef, X_max+1, X_max+1) #X_vec: from 0 to X_max (X_max + 1 possible entries)
#transition matrix P is X_max+1 by X_max+1
for (i, j) in product(1:X_max+1, 1:X_max+1)
#product(): IterTools function creating all possible i,j indices combo where each i,j runs from 1 to X_max+1; useful for creating matrix P[i,j] entries
    P[i, j] = sum((σ[i, d+1] == j-1) * pdf(ϕ, d) for d in 0:X_max) #or == j+1?: SEE QUESTIONS AT THE TOP
end

# normalize (we truncated the geometric distribution of shocks at X_max)
for i in 1:X_max+1
    P[i,:] = P[i,:] / sum(P[i,:]) #this will be done for each entry of row i;
    #sum(P[i,:]): sum of entries of a row i 
end


mc = MarkovChain(P, X_vec) #simulating markov chain of X
X_ts = simulate(mc, 50, init = 10); #QuantEcon function for Markov chain simulation

plot(X_ts, label="Inventory", xlabel="Time", ylabel="Inventory")

Ψ = stationary_distributions(mc)[1] #stationary distribution of X
plot(X_vec, Ψ, label="stationary distribution", xlabel="Inventory", ylabel="Probability")


## other iterative methods - opi (Optimistic policy iteration) - SLIDES 17-21, especially 20-21

function Tσ_operator(v,σ_ind,model) #compare to T_operator for vfi (line ca. 50) - SLIDES 20-21
#fewer loops, next inventory pinned down using policy, we don't iterate over levels of next inventory

    (;p, ϕ, X_max, K, c, β, X_vec) = model

    v_new = similar(v)
    for (d_ind, d) in enumerate(X_vec)
        for (x_ind, x) in enumerate(X_vec)
    
            x_next_ind = σ_ind[x_ind,d_ind]    
            x_next = X_vec[x_next_ind]

            sold = min(d,x)
            revenue = p * sold - K * (x_next > (x - sold)) - c * (x_next - (x - sold))

            if x_next >= x - sold
                v_new[x_ind,d_ind] = revenue + β * sum( v[x_next_ind,d_next_ind] * pdf(ϕ,d_next_ind-1) for d_next_ind in 1:X_max+1 )
            else
                v_new[x_ind,d_ind] = -Inf
            end
    
        end
    end

    return v_new

end

function opi(model; tol = 1e-6, maxiter = 1000, max_m = 15) #optimistic policy iteration - SLIDES 20-21

    (;p, ϕ, X_max, K, c, β, X_vec) = model
    
    tol = 1e-6; maxiter = 1000
    error = tol + 1.0; iter = 1 #  initialize
    v = zeros(X_max+1,X_max+1); 

    while error > tol && iter < maxiter
        v_new, σ_new, σ_ind_new = T_operator(v,model)

        for m in 1:max_m
            v_new = Tσ_operator(v_new,σ_ind_new,model)
        end


        error = maximum(abs.(v_new .- v))
        v = v_new
        iter += 1
    end
    # one more iteration to get the policy function
    v, σ = T_operator(v,model)[1:2]
    return v, σ, iter, error
        
end

v_opi, σ_opi, iter_opi, error_opi = opi(model)
σ_opi - σ #0

t_vfi = @time vfi(model) #50 sec
t_opi = @time opi(model) #6 sec - much faster and uses less memory
#under well-defined structure like Markov Chain: GO WITH OPI (Optimistic Policy Iteration) INSTEAD OF VFI
#OPI is much faster and uses less memory