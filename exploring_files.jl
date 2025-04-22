using Pkg
using Printf
using JLD2
using Oceananigans
using Oceananigans.Units: minute, minutes, hours

fld_file="outputs/langmuir_turbulence_fields_0.jld2"
averages_file="outputs/langmuir_turbulence_averages_0.jld2"

f = jldopen(fld_file)
fa = jldopen(averages_file)


# Extract grid info from field file
print("")
print("Field file that is causing error: \n \n")
@show keys(f)
Nx = f["grid/Nx"]
Ny = f["grid/Ny"]
Nz = f["grid/Nz"]
@show Nx, Ny, Nz

Lx = f["grid/Lx"]
Ly = f["grid/Ly"]
Lz = f["grid/Lz"]
@show Lx, Ly, Lz

Hx = f["grid/Hx"]
Hy = f["grid/Hy"]
Hz = f["grid/Hz"]
@show Hx, Hy, Hz

@show f["grid/architecture/local_rank"]
@show f["grid/architecture/mpi_requests"]
@show keys(f["grid/architecture/partition"])
@show keys(f["grid/architecture/ranks"])
@show keys(f["grid/architecture/local_index"])
@show f["grid/architecture/connectivity"]
@show f["grid/architecture/communicator/val"]
@show f["grid/architecture/mpi_tag/x"]

print(f["serialized/grid"])

# Same for averages file
print("\n Average file that is reading in correctly: \n \n")
@show keys(fa)
Nx_avg = fa["grid/Nx"]
Ny_avg = fa["grid/Ny"]
Nz_avg = fa["grid/Nz"]
@show Nx_avg, Ny_avg, Nz_avg

Lx_avg = fa["grid/Lx"]
Ly_avg = fa["grid/Ly"]
Lz_avg = fa["grid/Lz"]
@show Lx_avg, Ly_avg, Lz_avg

Hx_avg = fa["grid/Hx"]
Hy_avg = fa["grid/Hy"]
Hz_avg = fa["grid/Hz"]
@show Hx_avg, Hy_avg, Hz_avg

@show fa["grid/architecture/local_rank"]
@show fa["grid/architecture/mpi_requests"]
@show keys(fa["grid/architecture/partition"])
@show keys(fa["grid/architecture/ranks"])
@show keys(fa["grid/architecture/local_index"])
@show fa["grid/architecture/connectivity"]
@show fa["grid/architecture/communicator/val"]
@show fa["grid/architecture/mpi_tag/x"]

print(fa["serialized/grid"])