using ITensors, ITensorMPS

let
    # Define your list of system sizes here
    Ns = [5, 10, 20, 30, 35, 40, 50, 57]

    # Physics parameters
    g = 1.01  # Transverse field (X)
    h = 0.5   # Longitudinal field (Z)

    # Optimized DMRG parameters for the chaotic regime
    nsweeps = 20
    maxdim = [10, 20, 40, 80, 100, 200, 400, 400, 800, 800]
    cutoff = [1E-10]
    # Noise MUST drop to 0.0 at the end to get a tight upper bound
    noise = [1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 0.0]

    println("Starting DMRG for Mixed-Field Ising Model...")
    println("Parameters: J=1.0, g=", g, ", h=", h)
#     println("-"^40)

    # Loop over every system size in the list
    for N in Ns
        sites = siteinds("S=1/2", N)

        # Build the Hamiltonian for the current N
        os = OpSum()

        # -J * ZZ interaction
        for j=1:N-1
            os -= 4,"Sz",j,"Sz",j+1
        end

        # Periodic boundary term
        os -= 4,"Sz",1,"Sz",N

        # -g * X (Transverse field)
        for j=1:N
            os -= 2*g,"Sx",j
        end

        # -h * Z (Longitudinal field) - THIS IS THE NEW TERM
        for j=1:N
            os -= 2*h,"Sz",j
        end

        H = MPO(os, sites)

        # Initialize and solve
        psi0_init = random_mps(sites; linkdims=2)

        # We keep outputlevel=0 so it doesn't spam the sweep history
        energy0, psi0 = dmrg(H, psi0_init; nsweeps, maxdim, cutoff, noise, outputlevel=0)

        # Print the result immediately so it logs to your file
        println("N: ", N)
        println("GS upper bound: ", energy0)
    end

    return
end