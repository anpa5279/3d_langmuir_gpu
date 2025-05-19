function num_check(sim)
    fields_to_check = merge(sim.model.velocities, sim.model.tracers, sim.model.pressures, (; νₑ=sim.model.auxiliary_fields.νₑ, ∂z_u =sim.model.stokes_drift.∂z_uˢ))
    n_fields = keys(fields_to_check)
    #@show n_fields
    for (n, name) in enumerate(n_fields)
        #@show name
        field = fields_to_check[name]
        #@show field
        x = field.data
        if any(isnan, parent(field))
            @show name, field
            index = collect.(Tuple.(findall(isnan, parent(field))))
            # for i in 1:length(index)
            #     msg = @sprintf("iteration: %d, time: %s, NaN in field %s at: %d, %d, %d \n
            #                     surrounding values: 
            #                     +Δx: %6.3e,\n
            #                     -Δx: %6.3e, \n
            #                     +Δy: %6.3e,\n
            #                     -Δy: %6.3e,\n
            #                     +Δz: %6.3e, \n
            #                     -Δz: %6.3e, \n",
            #                     iteration(sim), 
            #                     prettytime(time(sim)), 
            #                     "$(n_fields[n])",
            #                     index[i][1], 
            #                     index[i][2], 
            #                     index[i][3], 
            #                     x[index[i][1]+1, index[i][2], index[i][3]],
            #                     x[index[i][1]-1, index[i][2], index[i][3]],
            #                     x[index[i][1], index[i][2]+1, index[i][3]],
            #                     x[index[i][1], index[i][2]-1, index[i][3]],
            #                     x[index[i][1], index[i][2], index[i][3]+1],
            #                     x[index[i][1], index[i][2], index[i][3]-1])
            #
            #     @info msg
            # end
            error()
        elseif any(isinf, parent(field))
            index = collect.(Tuple.(findall(isinf, parent(field))))
            for i in 1:length(index)
                msg = @sprintf("iteration: %d, time: %s, Inf in field %s at: %d, %d, %d \n",
                                iteration(sim), 
                                prettytime(time(sim)), 
                                "$(n_fields[n])",
                                index[i][1], 
                                index[i][2], 
                                index[i][3])
                @info msg
            end
            error()
        end
    end
    
    return nothing
end
