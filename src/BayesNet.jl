module BayesNet

using DataFrames, Lasso
using Graphs, SimpleWeightedGraphs
using StatsPlots, LassoPlot, Plots
using MLDataUtils
using Turing

function modeldata(df::DataFrame, response::String)
    allvars = names(df)
    @assert response ∈ allvars
    predvars = allvars[allvars .!== response]
    X = df[:, predvars]
    y = df[:, response]
    return (
        X = (
            labels = predvars,
            data = Matrix(X)
        ),
        y = (
            labels = response,
            data = y
        )
    )
end

function fitlassopath(X, y; criterion=MinBIC())
    modelpath = fit(LassoPath, X.data, y.data)
    modelcoef = coef(modelpath; select=criterion)
    intercept = popfirst!(modelcoef)
    modelmask = abs.(modelcoef) .> 0.0
    predictors = X.labels[modelmask]
    betas = modelcoef[modelmask]
    return (
        lassopath = modelpath,
        node = y.labels,
        edges = predictors .=> betas,
        intercept = intercept
    )
end

function fitlasso(X, y, λ; criterion=MinBIC())
    model = fit(
        LassoModel, 
        X.data, y.data, 
        λ=[λ...], select=criterion
    )
    modelcoefs = collect(coef(model))
    intercept = popfirst!(modelcoefs)
    modelmask = abs.(modelcoefs) .> 0.0
    predictors = X.labels[modelmask]
    betas = modelcoefs[modelmask]
    return (
        lasso = model,
        node = y.labels,
        edges = predictors .=> betas,
        intercept = intercept
    )
end

function integratenode!(g, lnode, regions)
    nodeid = findfirst(region -> region == lnode.node, regions)
    foreach(lnode.edges) do (neighbor, weight)
        neighborid = findfirst(region -> region == neighbor, regions)
        add_edge!(g, nodeid, neighborid, weight)
    end
end

function timeseries2graph(dftime::DataFrame; criterion=MinBIC())
    regions = names(dftime)
    g = SimpleWeightedDiGraph(length(regions))
    for region in regions
        X, y = modeldata(dftime, region)
        lnode = fitlassopath(X, y; criterion=criterion)
        integratenode!(g, lnode, regions)
    end
    return g
end

function lagdf(df::DataFrame, lag::Int)
    head_end = 1
    tail_end = nrow(df) - lag
    dfvec = Vector{DataFrame}(undef,0)
    while tail_end <= nrow(df)
        dfslice = copy(df[head_end:tail_end,:])
        rename_scheme = names(dfslice) .=> ("t" .* string(head_end) .* "-" .* names(dfslice))
        rename!(dfslice, rename_scheme...)
        push!(dfvec, dfslice)
        tail_end += 1
        head_end += 1
    end
    return hcat(dfvec...)
end


function bayeslasso(df, response=nothing, criterion=MinBIC(), bywhat=:β)
    if isnothing(response) response = rand(names(df)) end
    X, y = modeldata(df, response)
    model = fit(LassoPath, X.data, y.data)
    βs = collect(coef(model; select=criterion))
    β0 = popfirst!(βs)
    β_mask = abs.(βs) .> 0.0
    β_retained = βs[β_mask]
    predictors_retained = X.labels[β_mask]
    model_df = DataFrame(
        predictor = predictors_retained,
        β = β_retained
    )
    sort!(model_df, bywhat)
    model_predictors = df[:,model_df.predictor]
    model_priors = model_df
    model_response = df[:,response]
    return (
        name = response,
        model = model,
        criterion = criterion,
        predictors = model_predictors,
        response = model_response,
        priors = model_priors,
        intercept = β0
    )
end

function plotlasso(lasso, x=:logλ)
    LassoPlot.plot(lasso.model; 
        x=x, select=lasso.criterion, 
        showselectors=[lasso.criterion]
    )
    plot!(legend = false)
    yaxis!("β")
end

@model function bayesmodel(X, y, β_prior=nothing, β0_prior=nothing)
    # Set variance prior.
    σ₂ ~ truncated(Normal(0, 100), 0, Inf)
    # Set intercept prior.
    # Set the priors on our coefficients.
    nfeatures = size(X, 2)
    if isnothing(β0_prior)
        intercept ~ Normal(0, sqrt(3))
    else
        intercept ~ Normal(β0_prior, sqrt(3))
    end
    if !isnothing(β_prior) && length(β_prior) == nfeatures
        coefficients = Vector{eltype(β_prior)}(undef,nfeatures)
        for i in 1:nfeatures
            coefficients[i] ~ Normal(β_prior[i], sqrt(10))
        end
    else
        coefficients ~ MvNormal(nfeatures, sqrt(10))
    end
    # Calculate all the mu terms.
    mu = intercept .+ X * coefficients
    y ~ MvNormal(mu, sqrt(σ₂))
end

function fitbayeslasso(lasso, sampler=NUTS(0.6), iters=1_3000, chains=6, mc=MCMCSerial)
    model = bayesmodel(
        Matrix(lasso.predictors),
        lasso.response,
        lasso.priors.β,
        lasso.intercept
    )
    chain = sample(model, sampler, mc(), iters, chains, progress=true)
    renamescheme = chain.name_map.parameters[3:end] .=> names(lasso.predictors)
    return replacenames(chain, renamescheme...)
end

function plotposterior(chain, predictor::String)
    StatsPlots.plot(chain, Symbol(predictor))
end

end # module