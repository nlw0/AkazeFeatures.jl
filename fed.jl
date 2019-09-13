using Primes: isprime

################################################################
function fed_tau_by_process_time(T::Float64, M::Int64, tau_max::Float64,
                            reordering::Bool)
    ## All cycles have the same fraction of the stopping time
    fed_tau_by_cycle_time(T/M, tau_max, reordering)
end

################################################################
function fed_tau_by_cycle_time(t::Float64, tau_max::Float64,
                               reordering::Bool)

    n = 0          ## Number of time steps
    scale = 0.0    ## Ratio of t we search to maximal t

    ## Compute necessary number of time steps
    n = ceil(Int64, sqrt(3.0*t/tau_max+0.25)-0.5-1.0e-8.0)
    scale = 3.0*t/(tau_max*Float64(n*(n+1)))

    ## Call internal FED time step creation routine
    fed_tau_internal(n,scale,tau_max,reordering)
end

################################################################
function fed_tau_internal(n::Int64,  scale::Float64, tau_max::Float64,
                          reordering::Bool) ::Int64

    c = 0.0
    d = 0.0             ## Time savers
    tauh = Float64[]    ## Helper vector for unsorted taus

    if (n <= 0)
        return 0, nothing
    end

    #### Allocate memory for the time step size
    tau = zeros(n)

    if (reordering)
        tauh = zeros(n)
    end

    ## Compute time saver
    c = 1.0 / (4.0 * n + 2.0)
    d = scale * tau_max / 2.0

    ## Set up originally ordered tau vector
    for k in 1:n
        h = cos(M_PI * (2.0f * k - 1.0) * c)

        if reordering
            tauh[k] = d / (h * h);
        else
            tau[k] = d / (h * h);
        end
    end

    ## Permute list of time steps according to chosen reordering function
    if reordering
        ## Choose kappa cycle with k = n/2
        ## This is a heuristic. We can use Leja ordering instead!!
        kappa = n ÷ 2

        ## Get modulus for permutation
        prime = nextprime(n + 1)

        ## Perform permutation
        k = 0
        l = 0
        while l < n
            index = ((k+1)*kappa) % prime - 1
            while (index >= n)
                index = ((k+1)*kappa) % prime - 1
                k++
            end

            tau[l+1] = tauh[index+1]

            k += 1
            l += 1
        end
    end

    return n, tau
end
