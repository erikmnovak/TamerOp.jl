# Display ranges are navigation windows, never replacements for exact grades.
# Event buttons use the stored values, including algebraic radii.
function _filtration_controls(x, parameter)
    G=DataIngestion.graded_complex(x)
    n=DataTypes.parameter_dim(G)
    grades=collect(Iterators.flatten(DataTypes.cell_grade_set(G,i) for i in eachindex(x.geometry)))
    family=get(x.provenance,:requested,:supplied)
    scale=get(x.provenance,:grade_scale,:supplied)
    geometric=haskey(x.provenance,:geometric) ? DataIngestion.geometric_data(x) : nothing
    return ntuple(n) do d
        events=sort!(unique([g[d] for g in grades]))
        lo,hi=isempty(events) ? (parameter[d],parameter[d]) : extrema(events)
        lo=min(lo,parameter[d]);hi=max(hi,parameter[d])
        label="Grade $d"
        discrete=false
        bounded=false
        if d==1 && scale in (:radius,:diameter,:squared_radius)
            label=scale===:radius ? "Physical radius" : scale===:diameter ? "Rips diameter" : "Squared radius"
            lo=zero(lo)
            params=get(x.provenance,:filtration_parameters,(;))
            cap=get(params,:radius,nothing)
            # Only a declared radius/diameter cap bounds this coordinate.
            if cap!==nothing && scale!==:squared_radius
                hi=cap;bounded=true
            end
        elseif d==2 && geometric!==nothing
            label=geometric.family===:multicover ? "Coverage depth k" : "Neighbour depth k"
            lo,hi=geometric.depth_window
            discrete=true
        elseif d==2 && family===:degree_rips
            label="Stored graph degree threshold k"
            lo=0;discrete=true
        elseif d==2 && family in (:rips_density,:rips_codensity)
            label=family===:rips_codensity ? "Stored codensity" : "Stored density"
        end
        # A constant coordinate still has an honest, small navigation range.
        if lo==hi && !discrete && !bounded
            hi=lo+one(lo)
        end
        drawable=all(v->isfinite(Float64(v)),(lo,hi)) && (lo==hi || Float64(lo)<Float64(hi))
        (;label,lower=lo,upper=hi,events,discrete,orientation=x.orientation[d],drawable)
    end
end

function _filtration_control_parameter(controls, values)
    length(controls)==length(values) || throw(ArgumentError("Use one value per grade coordinate."))
    return ntuple(length(controls)) do d
        c=controls[d];v=values[d]
        v isa Real && !(v isa Bool) && isfinite(v) && c.lower<=v<=c.upper ||
            throw(ArgumentError("$(c.label) must lie in [$(c.lower), $(c.upper)] for this control."))
        c.discrete && !isinteger(v) && throw(ArgumentError("$(c.label) requires an integer."))
        c.discrete ? Int(v) : v
    end
end

function _filtration_event_parameter(controls, parameter, axis, step)
    c=controls[axis]
    # Forward follows the filtration order, including superlevel coordinates.
    forward=step*c.orientation>0
    candidates=filter(v->c.lower<=v<=c.upper && (forward ? v>parameter[axis] : v<parameter[axis]),c.events)
    isempty(candidates) && return nothing
    event=forward ? minimum(candidates) : maximum(candidates)
    ntuple(d->d==axis ? event : parameter[d],length(parameter))
end
