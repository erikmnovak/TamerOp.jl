function _a124_cloud()
    TamerOp.DataTypes.PointCloud([0.0 0.0;2.0 0.0;2.0 1.0;0.0 1.0])
end
_a124_session(;kwargs...)=TamerOp.source_laboratory(_a124_cloud();max_diameter=3.0,dtm_mass=0.3,codensity=3.0,diameter=2.0,kwargs...)
