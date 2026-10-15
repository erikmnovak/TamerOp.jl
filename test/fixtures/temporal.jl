function temporal_cloud_fixture(;representatives=true)
    T=TamerOp.TemporalPersistence
    points=[(0.,0.),(2.,0.),(1.,2.)]
    T.temporal_persistence([points,[points;[(1.,0.5)]],points];scales=[0,1,2,3,4],
        point_ids=[[1,2,3],[1,2,3,4],[1,2,3]],representatives)
end
