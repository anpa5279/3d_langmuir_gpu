using Pkg
using Plots
using Printf
using JLD2
using Oceananigans
using Measures
model = 1 # 0 for box model, 1 for 0d case
# opening oceananigans output file
if model == 0
     fld_file="outputs/box_model.jld2"
     image = "outputs/box_model.png"
     CO₂_oc = FieldTimeSeries(fld_file, "CO₂")
     CO₃_oc = FieldTimeSeries(fld_file, "CO₃")
     HCO₃_oc = FieldTimeSeries(fld_file, "HCO₃")
     OH_oc = FieldTimeSeries(fld_file, "OH")
     BOH₃_oc = FieldTimeSeries(fld_file, "BOH₃")
     BOH₄_oc = FieldTimeSeries(fld_file, "BOH₄")
     t = CO₂_oc.times[1:11]
     dt = Float64[t.step][1]

     CO₂_oc = vec(CO₂_oc.data[1:11])
     CO₃_oc = vec(CO₃_oc.data[1:11])
     HCO₃_oc = vec(HCO₃_oc.data[1:11])
     OH_oc = vec(OH_oc.data[1:11])
     BOH₃_oc = vec(BOH₃_oc.data[1:11])
     BOH₄_oc = vec(BOH₄_oc.data[1:11])
else
     @load "outputs/0d-case.jld2" t u 
     image = "outputs/0d-case.png"
     u = reduce(hcat, sol.u)'
     CO₂_oc = u[:, 1]
     CO₃_oc = u[:, 2]
     HCO₃_oc = u[:, 3]
     OH_oc = u[:, 4]
     BOH₃_oc = u[:, 5]
     BOH₄_oc = u[:, 6]
     dt = 0.05
end 
N = length(t)
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
     if N > line_count
          values = parse.(Float64, split(lines))
          global line_count += 1 
          # Store in respective arrays (assuming correct ordering in file)
          push!(CO₂_f,   values[2]/(1e6))
          push!(HCO₃_f,  values[3]/(1e6))
          push!(CO₃_f,   values[4]/(1e6))
          push!(BOH₃_f,  values[5]/(1e6))
          push!(BOH₄_f,  values[6]/(1e6))
          push!(OH_f,    values[7]/(1e6))
          push!(CO₂_d,  (CO₂_oc[line_count] - CO₂_f[line_count]) / CO₂_f[line_count] * 100)
          push!(CO₃_d,  (CO₃_oc[line_count] - CO₃_f[line_count]) / CO₃_f[line_count] * 100)
          push!(HCO₃_d, (HCO₃_oc[line_count] - HCO₃_f[line_count]) / HCO₃_f[line_count] * 100)
          push!(OH_d,   (OH_oc[line_count] - OH_f[line_count]) / OH_f[line_count] * 100) 
          push!(BOH₃_d, (BOH₃_oc[line_count] - BOH₃_f[line_count]) / BOH₃_f[line_count] * 100)
          push!(BOH₄_d, (BOH₄_oc[line_count] - BOH₄_f[line_count]) / BOH₄_f[line_count] * 100)
     else
          break 
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
png(pf, image)

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