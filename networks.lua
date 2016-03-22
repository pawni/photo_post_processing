function get_network_1() -- first convolution -> output -> convolution
    local input = nn.Identity()()
    -- in: 224x224x3 out: 224x224x3
    local ce1 = nn.SpatialConvolutionMM(3, 3, 3, 3, 1, 1, 1, 1)(input)
    local ae1 = nn.ReLU()(ce1)
    -- in 224x224x3 out 224x224x64
    local ce2 = nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1, 1, 1)(ae1)
    local ae2 = nn.ReLU()(ce2)
    -- 224x224x64
    local ce3 = nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1, 1, 1)(ae2)
    local ae3 = nn.ReLU()(ce3)
    local mp3 = nn.SpatialMaxPooling(2, 2)(ae3)
    -- 112 x 112
    local ce4 = nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1, 1, 1)(mp3)
    local ae4 = nn.ReLU()(ce4)
    local mp4 = nn.SpatialMaxPooling(2, 2)(ae4)
    -- 56x56
    local ce5 = nn.SpatialConvolutionMM(256, 512, 3, 3, 1, 1, 1, 1)(mp4)
    local ae5 = nn.ReLU()(ce5)
    local mp5 = nn.SpatialMaxPooling(2, 2)(ae5)
    -- 28x28
    local ct = nn.SpatialConvolutionMM(512, 256, 1, 1)(mp5)
    local at = nn.ReLU()(ct)
    local bnt = nn.SpatialBatchNormalization(256)(at)
    -- 28x28
    local up1 = nn.SpatialUpSamplingNearest(2)(bnt)
    local bn4 = nn.SpatialBatchNormalization(256)(mp4)
    local add1 = nn.CAddTable()({up1, bn4})
    local cd1 = nn.SpatialConvolutionMM(256, 128, 3, 3, 1, 1, 1, 1)(add1)
    local ad1 = nn.ReLU()(cd1)
    -- 56x56
    local up2 = nn.SpatialUpSamplingNearest(2)(ad1)
    local bn3 = nn.SpatialBatchNormalization(128)(mp3)
    local add2 = nn.CAddTable()({up2, bn3})
    local cd2 = nn.SpatialConvolutionMM(128, 64, 3, 3, 1, 1, 1, 1)(add2)
    local ad2 = nn.ReLU()(cd2)
    -- 112x112
    local up3 = nn.SpatialUpSamplingNearest(2)(ad2)
    local bn2 = nn.SpatialBatchNormalization(64)(ae2)
    local add3 = nn.CAddTable()({up3, bn2})
    local cd3 = nn.SpatialConvolutionMM(64, 3, 3, 3, 1, 1, 1, 1)(add3)
    local ad3 = nn.ReLU()(cd3)
    -- 224x224
    local bn1 = nn.SpatialBatchNormalization(3)(ae1)
    local add4 = nn.CAddTable()({ad3, bn1})
    local cd4 = nn.SpatialConvolutionMM(3, 3, 3, 3, 1, 1, 1, 1)(add4)
    local ad4 = nn.ReLU()(cd4)

    local cd5 = nn.SpatialConvolutionMM(3, 3, 3, 3, 1, 1, 1, 1)(ad4)
    local ad5 = nn.ReLU()(cd5)

    local output = nn.Identity()(ad5)
    local network= nn.gModule({input},{output})
    --nngraph.annotateNodes()
    return network
end

function get_network_2() -- input -> output
    local input = nn.Identity()()
    -- in: 224x224x3 out: 224x224x3
    local ce1 = nn.SpatialConvolutionMM(3, 3, 3, 3, 1, 1, 1, 1)(input)
    local ae1 = nn.ReLU()(ce1)
    -- in 224x224x3 out 224x224x64
    local ce2 = nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1, 1, 1)(ae1)
    local ae2 = nn.ReLU()(ce2)
    -- 224x224x64
    local ce3 = nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1, 1, 1)(ae2)
    local ae3 = nn.ReLU()(ce3)
    local mp3 = nn.SpatialMaxPooling(2, 2)(ae3)
    -- 112 x 112
    local ce4 = nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1, 1, 1)(mp3)
    local ae4 = nn.ReLU()(ce4)
    local mp4 = nn.SpatialMaxPooling(2, 2)(ae4)
    -- 56x56
    local ce5 = nn.SpatialConvolutionMM(256, 512, 3, 3, 1, 1, 1, 1)(mp4)
    local ae5 = nn.ReLU()(ce5)
    local mp5 = nn.SpatialMaxPooling(2, 2)(ae5)
    -- 28x28
    local ct = nn.SpatialConvolutionMM(512, 256, 1, 1)(mp5)
    local at = nn.ReLU()(ct)
    local bnt = nn.SpatialBatchNormalization(256)(at)
    -- 28x28
    local up1 = nn.SpatialUpSamplingNearest(2)(bnt)
    local bn4 = nn.SpatialBatchNormalization(256)(mp4)
    local add1 = nn.CAddTable()({up1, bn4})
    local cd1 = nn.SpatialConvolutionMM(256, 128, 3, 3, 1, 1, 1, 1)(add1)
    local ad1 = nn.ReLU()(cd1)
    -- 56x56
    local up2 = nn.SpatialUpSamplingNearest(2)(ad1)
    local bn3 = nn.SpatialBatchNormalization(128)(mp3)
    local add2 = nn.CAddTable()({up2, bn3})
    local cd2 = nn.SpatialConvolutionMM(128, 64, 3, 3, 1, 1, 1, 1)(add2)
    local ad2 = nn.ReLU()(cd2)
    -- 112x112
    local up3 = nn.SpatialUpSamplingNearest(2)(ad2)
    local bn2 = nn.SpatialBatchNormalization(64)(ae2)
    local add3 = nn.CAddTable()({up3, bn2})
    local cd3 = nn.SpatialConvolutionMM(64, 3, 3, 3, 1, 1, 1, 1)(add3)
    local ad3 = nn.ReLU()(cd3)
    -- 224x224
    local bn1 = nn.SpatialBatchNormalization(3)(ae1)
    local add4 = nn.CAddTable()({ad3, bn1})
    local cd4 = nn.SpatialConvolutionMM(3, 3, 3, 3, 1, 1, 1, 1)(add4)
    local ad4 = nn.ReLU()(cd4)

    local bni = nn.SpatialBatchNormalization(3)(input)
    local add5 = nn.CAddTable()({ad4, bni})
    local ad5 = nn.ReLU()(add5)

    local output = nn.Identity()(ad5)
    local network= nn.gModule({input},{output})
    --nngraph.annotateNodes()
    return network
end
