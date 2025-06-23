using Pkg
using Plots
using Printf
using JLD2
using Oceananigans
using Measures
model = 2 # 0 for box model, 1 for carbonate diffeq test, 2 for 0d case
# opening oceananigans output file
if model == 0
     fld_file="outputs/box_model.jld2"
     image = "outputs/box_model.png"
     pd_image = "outputs/percent_difference_box.png"
     CO₂_oc = FieldTimeSeries(fld_file, "CO₂")
     CO₃_oc = FieldTimeSeries(fld_file, "CO₃")
     HCO₃_oc = FieldTimeSeries(fld_file, "HCO₃")
     OH_oc = FieldTimeSeries(fld_file, "OH")
     BOH₃_oc = FieldTimeSeries(fld_file, "B(OH)₃")
     BOH₄_oc = FieldTimeSeries(fld_file, "B(OH)₄")
     t = CO₂_oc.times[1:11]
     dt = Float64[t.step][1]

     CO₂_oc = vec(CO₂_oc.data[1:11])
     CO₃_oc = vec(CO₃_oc.data[1:11])
     HCO₃_oc = vec(HCO₃_oc.data[1:11])
     OH_oc = vec(OH_oc.data[1:11])
     BOH₃_oc = vec(BOH₃_oc.data[1:11])
     BOH₄_oc = vec(BOH₄_oc.data[1:11])
elseif model == 1
     @load "outputs/carbonate-diffeq-test.jld2" t u 
     image = "outputs/carbonate-diffeq-test.png"
     pd_image = "outputs/percent_difference_carbonate-diffeq-test.png"
     u = reduce(hcat, sol.u)'
     CO₂_oc = u[:, 1]
     HCO₃_oc = u[:, 2]
     CO₃_oc = u[:, 3]
     H_oc = u[:, 4]
     OH_oc = u[:, 5]
     BOH₃_oc = u[:, 6]
     BOH₄_oc = u[:, 7]
     dt = 0.05
else
     @load "outputs/0d-case.jld2" t u 
     image = "outputs/0d-case.png"
     pd_image = "outputs/percent_difference_0d-case.png"
     u = reduce(hcat, u)'
     CO₂_oc = u[:, 1]/(1e6)
     HCO₃_oc = u[:, 2]/(1e6)
     CO₃_oc = u[:, 3]/(1e6)
     OH_oc = u[:, 4]/(1e6)
     BOH₃_oc = u[:, 5]/(1e6)
     BOH₄_oc = u[:, 6]/(1e6)
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
stepping = 26 #N
# CO₂
CO₂p = plot(t[1:stepping], [CO₂_oc[1:stepping] CO₂_f[1:stepping]],
     xlabel = "t (s)", ylabel = "CO₂ (mol/kg)",
     title = "CO₂")
# CO₃
CO₃p = plot(t[1:stepping], [CO₃_oc[1:stepping] CO₃_f[1:stepping]],
     xlabel = "t (s)", ylabel = "CO₃ (mol/kg)",
     title = "CO₃")
# HCO₃
HCO₃p = plot(t[1:stepping], [HCO₃_oc[1:stepping] HCO₃_f[1:stepping]],
     xlabel = "t (s)", ylabel = "HCO₃ (mol/kg)",
     title = "HCO₃")
# OH
OHp = plot(t[1:stepping], [OH_oc[1:stepping] OH_f[1:stepping]],
     xlabel = "t (s)", ylabel = "OH (mol/kg)",
     title = "OH")
# BOH₃
BOH₃p = plot(t[1:stepping], [BOH₃_oc[1:stepping] BOH₃_f[1:stepping]],
     xlabel = "t (s)", ylabel = "B(OH)₃ (mol/kg)",
     title = "B(OH)₃")
# BOH₄
BOH₄p = plot(t[1:stepping], [BOH₄_oc[1:stepping] BOH₄_f[1:stepping]],
     xlabel = "t (s)", ylabel = "B(OH)₄ (mol/kg)",
     title = "B(OH)₄")
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
CO₂p = plot(t[1:stepping], CO₂_d[1:stepping],
     xlabel = "t (s)", ylabel = "CO₂ %",
     title = "CO₂")
# CO₃
CO₃p = plot(t[1:stepping], CO₃_d[1:stepping],
     xlabel = "t (s)", ylabel = "CO₃ %",
     title = "CO₃")
# HCO₃
HCO₃p = plot(t[1:stepping], HCO₃_d[1:stepping],
     xlabel = "t (s)", ylabel = "HCO₃ %",
     title = "HCO₃")
# OH
OHp = plot(t[1:stepping], OH_d[1:stepping],
     xlabel = "t (s)", ylabel = "OH %",
     title = "OH")
# BOH₃
BOH₃p = plot(t[1:stepping], BOH₃_d[1:stepping],
     xlabel = "t (s)", ylabel = "B(OH)₃ %",
     title = "B(OH)₃")
# BOH₄
BOH₄p = plot(t[1:stepping], BOH₄_d[1:stepping],
     xlabel = "t (s)", ylabel = "B(OH)₄ %",
     title = "B(OH)₄")
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
png(pf, pd_image)