using Pkg
using Plots
using Printf
using JLD2
using Oceananigans
using Measures

# opening oceananigans output file
fld_file="outputs/box_model.jld2"

CO₂_oc = FieldTimeSeries(fld_file, "CO₂")
CO₃_oc = FieldTimeSeries(fld_file, "CO₃")
HCO₃_oc = FieldTimeSeries(fld_file, "HCO₃")
H_oc = FieldTimeSeries(fld_file, "H")
OH_oc = FieldTimeSeries(fld_file, "OH")
BOH₃_oc = FieldTimeSeries(fld_file, "BOH₃")
BOH₄_oc = FieldTimeSeries(fld_file, "BOH₄")
t = CO₂_oc.times
dt = Float64[t.step][1]

CO₂_oc = vec(CO₂_oc.data)
CO₃_oc = vec(CO₃_oc.data)
HCO₃_oc = vec(HCO₃_oc.data)
H_oc = vec(H_oc.data)
OH_oc = vec(OH_oc.data)
BOH₃_oc = vec(BOH₃_oc.data)
BOH₄_oc = vec(BOH₄_oc.data)

# opening fortran output file
fortran_file = "outputs/cc.hst"
f = open(fortran_file)

CO₂_f = Float64[]
CO₃_f = Float64[]
HCO₃_f = Float64[]
OH_f = Float64[]
BOH₃_f = Float64[]
BOH₄_f = Float64[]

CO₂_d = Float64[]
CO₃_d = Float64[]
HCO₃_d = Float64[]
OH_d = Float64[]
BOH₃_d = Float64[]
BOH₄_d = Float64[]

line_count = 0              

for lines in readlines(f)
     # Split the line into string tokens and parse to Float64
     values = parse.(Float64, split(lines))
     #@show round(values[1]*24*60*60)
     if rem(round(values[1]*24*60*60, digits = 3), dt) == 0.0 || line_count == 0
          # increment line_count
          global line_count += 1 
          # Store in respective arrays (assuming correct ordering in file)
          push!(CO₂_f,   values[2]/(1e6))
          push!(HCO₃_f,  values[3]/(1e6))
          push!(CO₃_f,   values[4]/(1e6))
          push!(BOH₃_f,  values[5]/(1e6))
          push!(BOH₄_f,  values[6]/(1e6))
          push!(OH_f,    values[7]/(1e6))
          push!(CO₂_d,  (CO₂_oc[line_count] - CO₂_f[line_count]) / CO₂_oc[line_count] * 100)
          push!(CO₃_d,  (CO₃_oc[line_count] - CO₃_f[line_count]) / CO₃_oc[line_count] * 100)
          push!(HCO₃_d, (HCO₃_oc[line_count] - HCO₃_f[line_count]) / HCO₃_oc[line_count] * 100)
          push!(OH_d,   (OH_oc[line_count] - OH_f[line_count]) / OH_oc[line_count] * 100) 
          push!(BOH₃_d, (BOH₃_oc[line_count] - BOH₃_f[line_count]) / BOH₃_oc[line_count] * 100)
          push!(BOH₄_d, (BOH₄_oc[line_count] - BOH₄_f[line_count]) / BOH₄_oc[line_count] * 100)
     end 

end

# plotting results

# CO₂
CO₂p = plot(t, [CO₂_oc CO₂_f],
     xlabel = "t (s)", ylabel = "CO₂ (mol m⁻³)",
     title = "CO₂")
# CO₃
CO₃p = plot(t, [CO₃_oc CO₃_f],
     xlabel = "t (s)", ylabel = "CO₃ (mol m⁻³)",
     title = "CO₃")
# HCO₃
HCO₃p = plot(t, [HCO₃_oc HCO₃_f],
     xlabel = "t (s)", ylabel = "HCO₃ (mol m⁻³)",
     title = "HCO₃")
# OH
OHp = plot(t, [OH_oc OH_f],
     xlabel = "t (s)", ylabel = "OH (mol m⁻³)",
     title = "OH")
# BOH₃
BOH₃p = plot(t, [BOH₃_oc BOH₃_f],
     xlabel = "t (s)", ylabel = "BOH₃ (mol m⁻³)",
     title = "BOH₃")
# BOH₄
BOH₄p = plot(t, [BOH₄_oc BOH₄_f],
     xlabel = "t (s)", ylabel = "BOH₄ (mol m⁻³)",
     title = "BOH₄")
# layout 
gr()
pf = plot(CO₂p, CO₃p, HCO₃p, OHp, BOH₃p, BOH₄p, layout = (3, 2), label = ["Oceananigans" "Fortran Miniapp"], legend = :outertop,         # << shared legend at the top
           legendcolumns = 2,
           size = (1600, 1200),
           left_margin = 30mm,
           bottom_margin = 10mm,
           plot_margin = 5mm, 
           titlefont = font(12), 
           tickfont = font(9), 
           labelfont = font(10), 
           xlabelfont = font(10), 
           ylabelfont = font(10), 
           titlefontsize = 12)
png(pf, "outputs/box_model-rkc77.png")

#plotting percent differences
#CO₂
CO₂p = plot(t, CO₂_d,
     xlabel = "t (s)", ylabel = "CO₂ %",
     title = "CO₂")
# CO₃
CO₃p = plot(t, CO₃_d,
     xlabel = "t (s)", ylabel = "CO₃ %",
     title = "CO₃")
# HCO₃
HCO₃p = plot(t, HCO₃_d,
     xlabel = "t (s)", ylabel = "HCO₃ %",
     title = "HCO₃")
# OH
OHp = plot(t, OH_d,
     xlabel = "t (s)", ylabel = "OH %",
     title = "OH")
# BOH₃
BOH₃p = plot(t, BOH₃_d,
     xlabel = "t (s)", ylabel = "BOH₃ %",
     title = "BOH₃")
# BOH₄
BOH₄p = plot(t, BOH₄_d,
     xlabel = "t (s)", ylabel = "BOH₄ %",
     title = "BOH₄")
# layout 
gr()
pf = plot(CO₂p, CO₃p, HCO₃p, OHp, BOH₃p, BOH₄p, layout = (3, 2), legend = false,         # << shared legend at the top
           size = (1600, 1200),
           left_margin = 30mm,
           bottom_margin = 10mm,
           plot_margin = 5mm, 
           titlefont = font(12), 
           tickfont = font(9), 
           labelfont = font(10), 
           xlabelfont = font(10), 
           ylabelfont = font(10), 
           titlefontsize = 12)
png(pf, "outputs/percent-difference.png")