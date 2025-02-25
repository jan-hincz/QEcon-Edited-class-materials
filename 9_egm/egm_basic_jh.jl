
using Parameters, Interpolations, Plots, LinearAlgebra, QuantEcon

#TAKEAWAYS: 
#Julia: difference between \prime ′ (used in this file for indicating derivative) and apostrophe ' (transpose)
#ag.prob_trans: 5 x 5 (ih: prod. states), but ag.prob_trans[1,:] Julia treats as column vector (5 x 1), not row vector!
#EGM can be quicker than VFI (see slide 8, section NOT VERY IMPORTANT and onwards)


# Define the setup object
Setup = @with_kw ( #with_kw: Parameters package macro
a_grid_min = 0.0,                                               # smallest grid point (asset holdings size)
a_grid_max = 40,                                                # largest grid point
a_grid_size = 500,                                              # grid size
a_grid = collect(range(0.0, a_grid_max, length = a_grid_size)), # create the grid given the parameters above
prod_grid_size = 5,                                             # grid size to discretize the AR(1) productivity process                   
σ_ϵ = sqrt(0.19),                                               # stdev of the AR(1) process for productivity
ϱ = 0.7,                                                        # persistence of the AR(1) process 
w = 1,                                                          # wage level in the economy                      
r = 0.1                                                         # interest rate in the economy  
)

# Define the agent object
Agent = @with_kw (
β = 0.9,                                    # The discount factor
σ = 1.0,                                    # IES: the utility function parameter
u = σ == 1 ? log : c->(c^(1-σ)-1)/(1-σ),    # CRRA utility function over consumption
u′ = c-> c.^(-σ),                           # marginal utility over consumption
u′_inv = c-> c.^(-1 / σ)                    #slide 10: inverse function of the marginal utility
) 

setup = Setup() #instantiation e.g. Setup(ϱ = 0.3) - and the rest of defaults defined above

# This is how we access an r
setup.r #0.1
# This is how we access the w
setup.w #1
# This is how we access the grid for assets
setup.a_grid
# This is how we access its first element
setup.a_grid[1] #0.0

ag = Agent()
# This is how we access the discount factor
ag.β #0.9

# discretize the AR(1) productivity process
discretized_y = tauchen(setup.prod_grid_size,setup.ϱ, setup.σ_ϵ) #y_{t+1} = ϱ*y_t + ϵ_t; ϵ_t ~ N(0,(σ_ϵ)^2)
#see 8.2 Markov chains slides 14-19 and APPROXIMATION (line ca.102-end) part of 8.2 code

# merge the agent object with the transition probability matrix (.p), and discretized grid (prod_grid_size = 5 elements)
ag = merge(ag,(prob_trans = discretized_y.p,prod_grid = exp.(discretized_y.state_values),init_dist = [0.5,0.5,0,0,0]))
#discretized_y.state_values = -1.83:0.92:1.83, but we want to have positive productivity -> exp function 
# In ag we now have the transition probability
ag.prob_trans #5x5 Matrix; 0.416817 when in the lowest income this is prob. to stay there; see 8.2 MC slide 5
# And the productivity grid
ag.prod_grid #5 possible positive values


# Allocate memory for the relevant arrays
c_EGM           = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2); #consumption policy function
#3-dimensional object: a (500 values), y (productivity, 5 values), age (2: young and old)
a′_EGM          = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2); #policy function for assets saved
a′_EGM          .= 0.0  #make sure it is initially filed with zeros
V_prime_EGM     = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2);
#array to store the derivative of the value function
EV_prime_EGM    = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 1);
#array to store the expected derivative of the value function

# Temporary arrays
c_temp      = Array{Float64,2}(undef, setup.a_grid_size,setup.prod_grid_size); #for each a,y combo
a_endo      = Array{Float64,2}(undef, setup.a_grid_size,setup.prod_grid_size); #for each a,y combo

# For simplicity define a variable income (this is our grid for income)
income = setup.w * ag.prod_grid #w*y

setup.a_grid #500 elements from 0 to 40

################## Quick task for you! - SLIDE 7 ##################
# 1. Use the c_EGM object to define the consumption policy of an old agent for:
#   - particular level of assets (first dimension)
#   - particular level of income (second dimension)
# Hint: the simplest way to fill c_EGM is to loop over each possible level of asset and income (see below):

# 2. Use the V_prime_EGM object to define the derivative of the value function of an old agent for:
#   - particular level of assets (first dimension)
#   - particular level of income (second dimension)
#   - use the c_EGM object defined above!

# You can use this code, just fill in the blanks:
for ia in eachindex(setup.a_grid) #over a (setup.a_grid indices)
  for ih in eachindex(ag.prod_grid) #over y (ag.prod_grid indices)
    c_EGM[ia,ih,2] = (1+setup.r)*setup.a_grid[ia] + income[ih]  #2: 2nd period (old)
    #e.g. c_EGM[1,1,2] = (1+setup.r)*setup.a_grid[1]+income[1] (arguments on LHS are indices)
    V_prime_EGM[ia,ih,2] = (1+setup.r)*ag.u′(c_EGM[ia,ih,2]) #Julia: difference between \prime ′ (used here) and apostrophe '  
    #u′ defined at the top as a function
  end
end


# Define the expected v' from the perspective of the young agent with the realization income[ih] (slide 6)
#EV_prime_EGM[:,ih,1]: 500 (a_grid) x 1 (given ih); V_prime_EGM[:,:,2]: 500 x 5 (from a given ih you can transition to any of 5 ihs)
#ag.prob_trans: 5 x 5 (ih: prod. states), but ag.prob_trans[ih,:] Julia treats as column vector (5 x 1), not row vector!
#ag.prob_trans[ih,:]: transition probs from today's ih to any prod. state tomorrow
#EV_prime_EGM[:,ih,1]: expected value as of young (t=1) of derivative with respect to a of V_2 (slide 6)
for ih in eachindex(ag.prod_grid)
  EV_prime_EGM[:,ih,1] = ( V_prime_EGM[:,:,2] * ag.prob_trans[ih,:] )
end



# The key loop of the EGM algorithm: #slides 11-12
# For each current (young) productivity state y
for ih in eachindex(ag.prod_grid) 
    # For each (optimal) level of assets saved for the future
    for (ia′,a′) in enumerate(setup.a_grid)
        # Calculate the RHS (bottom slide 6) of the Euler equation
        rhs             = ag.β *  (EV_prime_EGM[ia′,ih,1]) 
        # Find a level of consumption which justifies such a' level (slide 11, step 2)
        c_temp[ia′,ih]  = ag.u′_inv.(rhs) #inverse utliity function to get c (slide 10)
        # Infer the initial level of assets while young (slide 11, step 3)
        a_endo[ia′,ih]  =  (a′ + c_temp[ia′,ih] - income[ih] )/(1 + setup.r)
    end
    # Get the level of a_grid for which agents are constrained (slide 13)
    cstr = setup.a_grid .< a_endo[1,ih] #logical statement, slide 13, point 4; EGM for a(a',y)
    # Get the level of a_grid for which agents are unconstrained 
    uncstr = cstr .== false #logical statement for indices of a that lead to no constraint (see below temp)
    # define the interpolated function (slide 12, points 5,6)
    interpolate_endo_g  = LinearInterpolation(a_endo[:,ih],setup.a_grid, extrapolation_bc=Line()) #Interpolations package
    #getting a line of endogeneous a' as function of now EXOGENEOUS a (which came from Endogeneous GM) - slide 12, step 5
    
    temp = @. income[ih] + (1+setup.r) * setup.a_grid[uncstr] - interpolate_endo_g(setup.a_grid[uncstr])
    #slide 11, step 3, solving for unconstr. c; broadcasting -> setup.a_grid[uncstr] -> for each level a on EXOGENEOUS grid,
    #we subtract endogeneous interpolated a' which was found by linear interpolation

    c_EGM[uncstr,ih,1]  = @. max(temp,1e-8) #prevents c from being negative
    c_EGM[cstr,ih,1]    = @. income[ih] + (1+setup.r) * setup.a_grid[cstr] #slide 13, step 5
    a′_EGM[:,ih,1]      = @. income[ih] + (1+setup.r) * setup.a_grid - c_EGM[:,ih,1] #slide 11, step 3, solving for a'
end


######## BEGINNING OF NOT VERY IMPORTANT ###################################################################################

#### A naive algorithm (short description slide 8) ####    
#### SEE HOW MUCH LONGER IT RUNS!!! #### 
# Allocate memory for the relevant arrays
c_vfi           = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2);
a′_vfi          = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2);
a′_vfi          .= 0.0
V_vfi           = Array{Float64,3}(undef, setup.a_grid_size,setup.prod_grid_size, 2);
# Temporary arrays
c_temp = Matrix{Float64}(undef, setup.a_grid_size, 1);
u_temp = Matrix{Float64}(undef, setup.a_grid_size, 1);
v_temp = Matrix{Float64}(undef, setup.a_grid_size, 1);


for ih in eachindex(ag.prod_grid)
    c_vfi[:,ih,2] = @. income[ih] +  (1+setup.r)* setup.a_grid 
end
a′_vfi[:,:,:] .= 0.0
# Just a straight-up value function of old (t=2)
V_vfi[:,:,2] = ag.u.( c_vfi[:,:,2] )
     
# For each current (young) productivity state
for ih in eachindex(ag.prod_grid)
    # For each current (young) level of assets 
    for (ia,a) in enumerate(setup.a_grid)
        # Test each level of assets saved for the future
        for (ia′,a′) in enumerate(setup.a_grid)
            if (1+setup.r) * setup.a_grid[ia] + income[ih]- setup.a_grid[ia′] < 1e-10
                c_temp[ia′] = 1e-10  # if consumption is negative, set to a very small number
                u_temp[ia′] = -Inf    # calculate utility
            else
                c_temp[ia′] = (1+setup.r) * setup.a_grid[ia] + income[ih] - setup.a_grid[ia′]
                u_temp[ia′] = ag.u(c_temp[ia′])  # calculate utility
            end
            # Calculate the expected value function for a' tested, given current ih
            EV_vfi = V_vfi[ia′,:,2]'*ag.prob_trans[ih,:]
            # Calculate the value of such a' choice
            v_temp[ia′] = u_temp[ia′] + ag.β * EV_vfi # calculate the current utility + continuation value
        end
        # Find the best ia' by finding the max v_temp
        V_vfi[ia,ih,1], ia′_opt       = findmax(v_temp[:])     # findmax will return max value and its index
        a′_vfi[ia,ih,1]               = setup.a_grid[ia′_opt]  # record the best level of assets saved
        c_vfi[ia,ih,1]                = c_temp[ia′_opt]        # record the consumption implied
    end
end
   
######## END OF NOT VERY IMPORTANT ###################################################################################


# Let's inspect the results from both methods (close, but VFI much slower)
# Consumption policy functions for young person:
plot(setup.a_grid[1:50],c_EGM[1:50,1,1], label = "EGM lowest productivity",linewidth=2) 
plot!(setup.a_grid[1:50],c_vfi[1:50,1,1], label = "VFI lowest productivity",linewidth=2) #slow method VFI (slide 8), not quick EGM
plot!(setup.a_grid[1:50],c_EGM[1:50,3,1], label = "EGM mid productivity",linewidth=2) 
plot!(setup.a_grid[1:50],c_vfi[1:50,3,1], label = "VFI mid productivity",linewidth=2) #slow method VFI (slide 8), not quick EGM
title!("Consumption policy functions")

# Asset policy functions for young person:
plot(setup.a_grid[1:50],a′_EGM[1:50,1,1], label = "EGM lowest productivity",linewidth=2) 
plot!(setup.a_grid[1:50],a′_vfi[1:50,1,1], label = "VFI lowest productivity",linewidth=2) #slow method VFI (slide 8), not quick EGM
plot!(setup.a_grid[1:50],a′_EGM[1:50,3,1], label = "EGM mid productivity",linewidth=2) 
plot!(setup.a_grid[1:50],a′_vfi[1:50,3,1], label = "VFI mid productivity",linewidth=2) #slow method VFI (slide 8), not quick EGM
title!("a' policy functions")