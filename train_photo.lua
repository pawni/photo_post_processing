require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'cltorch'
require 'clnn'

----------------------------------------------------------------------
-- parse command-line options
--
--
local opt = lapp[[
   -r,--learningRate  (default 0.1)        learning rate, for SGD only
   -b,--batchSize     (default 2)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   --cl                                     enable open cl
]]

-- fix seed
torch.manualSeed(1)

-- threads
--num_threads = 4;
torch.setnumthreads(4)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')
--torch.setdefaulttensortype('torch.ClTensor')

function get_network_1()
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

function get_network_2()
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

function get_network_1_2()
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
    local add1 = nn.CAddTable()({up1, mp4})
    local cd1 = nn.SpatialConvolutionMM(256, 128, 3, 3, 1, 1, 1, 1)(add1)
    local ad1 = nn.ReLU()(cd1)
    -- 56x56
    local up2 = nn.SpatialUpSamplingNearest(2)(ad1)
    local add2 = nn.CAddTable()({up2, mp3})
    local cd2 = nn.SpatialConvolutionMM(128, 64, 3, 3, 1, 1, 1, 1)(add2)
    local ad2 = nn.ReLU()(cd2)
    -- 112x112
    local up3 = nn.SpatialUpSamplingNearest(2)(ad2)
    local add3 = nn.CAddTable()({up3, ae2})
    local cd3 = nn.SpatialConvolutionMM(64, 3, 3, 3, 1, 1, 1, 1)(add3)
    local ad3 = nn.ReLU()(cd3)
    -- 224x224
    local add4 = nn.CAddTable()({ad3, ae1})
    local cd4 = nn.SpatialConvolutionMM(3, 3, 3, 3, 1, 1, 1, 1)(add4)
    local ad4 = nn.ReLU()(cd4)

    local cd5 = nn.SpatialConvolutionMM(3, 3, 3, 3, 1, 1, 1, 1)(ad4)
    local ad5 = nn.ReLU()(cd5)

    local output = nn.Identity()(ad5)
    local network= nn.gModule({input},{output})
    --nngraph.annotateNodes()
    return network
end

function get_network_2_2()
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
    -- 28x28
    local up1 = nn.SpatialUpSamplingNearest(2)(at)
    local add1 = nn.CAddTable()({up1, mp4})
    local cd1 = nn.SpatialConvolutionMM(256, 128, 3, 3, 1, 1, 1, 1)(add1)
    local ad1 = nn.ReLU()(cd1)
    -- 56x56
    local up2 = nn.SpatialUpSamplingNearest(2)(ad1)
    local add2 = nn.CAddTable()({up2, mp3})
    local cd2 = nn.SpatialConvolutionMM(128, 64, 3, 3, 1, 1, 1, 1)(add2)
    local ad2 = nn.ReLU()(cd2)
    -- 112x112
    local up3 = nn.SpatialUpSamplingNearest(2)(ad2)
    local add3 = nn.CAddTable()({up3, ae2})
    local cd3 = nn.SpatialConvolutionMM(64, 3, 3, 3, 1, 1, 1, 1)(add3)
    local ad3 = nn.ReLU()(cd3)
    -- 224x224
    local add4 = nn.CAddTable()({ad3, ae1})
    local cd4 = nn.SpatialConvolutionMM(3, 3, 3, 3, 1, 1, 1, 1)(add4)
    local ad4 = nn.ReLU()(cd4)

    local add5 = nn.CAddTable()({ad4, input})
    local ad5 = nn.ReLU()(add5)

    local output = nn.Identity()(ad5)
    local network= nn.gModule({input},{output})
    --nngraph.annotateNodes()
    return network
end
print(opt.cl)
if opt.cl then
    print('doing cl stuff')
    criterion = nn.MSECriterion():cl()
    model = get_network_2_2():cl()

    print("get parameters")
    parameters,gradParameters = model:cl():getParameters()
else
    criterion = nn.MSECriterion()
    model = get_network_2_2()

    print("get parameters")
    parameters,gradParameters = model:getParameters()
end
print('<mnist> using model:')
print(model)

function generateDataset(inp_path, out_path)
    local fnum = #paths.dir(out_path)
    print("generating dataset with " .. fnum .. " images")
    local ds = torch.Tensor(fnum, 2, 3, 240, 160)
    local i = 1
    for file in paths.iterfiles(out_path) do
        collectgarbage()
        if file ~= ".DS_Store" then
           local tmp_inp = image.load(paths.concat(inp_path, file))
           local tmp_out = image.load(paths.concat(out_path, file))
           if tmp_inp:size()[2] ~= 240 then
               tmp_inp = tmp_inp:resize(1,3,160,240);
               tmp_out = tmp_out:resize(1,3,160,240);
               tmp_inp = tmp_inp:transpose(3,4);
               tmp_out = tmp_out:transpose(3,4);
           else
               tmp_inp = tmp_inp:resize(1,3,240,160);
               tmp_out = tmp_out:resize(1,3,240,160);
           end
           ds[i][1] = tmp_inp
           ds[i][2] = tmp_out
           i = i+1
           xlua.progress(i, fnum)
       end
    end
    print("done")
    return ds
end
local trainingSet = generateDataset("raw/train/", "processed/train/")
local validationSet = generateDataset("raw/val/", "processed/val/")
local testSet = generateDataset("raw/test/", "processed/test/")




--torch.save(paths.concat("photos.net"), model)
print(model)
print("dataset generated")

--trainLogger = optim.Logger('train.log')
batchsize = opt.batchSize;
function train(trainingSet, validationSet)
    -- epoch tracker
    epoch = epoch or 1
    train_error = 0
    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchsize .. ']')
    for t = 1,trainingSet:size()[1],batchsize do
        -- create mini batch
        local inputs = torch.Tensor(batchsize,3,240,160)
        local targets = torch.Tensor(batchsize,3,240,160)
        local k = 1
        for i = t,math.min(t+batchsize-1,trainingSet:size()[1]) do
            -- load new sample
            local sample = trainingSet[i]
            local input = sample[1]:clone()
            local target = sample[2]:clone()
            --print(input:size())
            --if input:size()

            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        if opt.cl then
            inputs = inputs:cl()
            targets = targets:cl()
        end
        model:zeroGradParameters()
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()
            -- evaluate function for complete mini batch
            --print(inputs:type())
            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            --train_error += f
            --print("error is " .. f)
            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            -- return f and df/dX
            return f,gradParameters
        end

        -- Perform SGD step:
        sgdState = sgdState or {
                learningRate = opt.learningRate,
                momentum = opt.momentum,
                learningRateDecay = 0
            }
        optim.sgd(feval, parameters, sgdState)

        -- disp progress
        xlua.progress(t, trainingSet:size()[1])

    end

    -- time taken
    time = sys.clock() - time
    time = time / trainingSet:size()[1]
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print("<trainer> training error is " .. train_error)
    -- print("current error is " .. f)
    --print("<trainer> epoch: " ..epoch .. "current validation error is " .. criterion:forward(model:forward(validationSet[{{},1}]), validationSet[{{},2}]))
    -- next epoch
    epoch = epoch + 1
end

maxEpochs = 15

for c=1,maxEpochs do
    train(trainingSet, validationSet)
end
--print("<trainer>  test error is " .. criterion:forward(model:forward(testSet[{{},1}]), testSet[{{},2}]))
print(model)
model2 = model:float()
model2:clearState()
--clone:share(model2,"weight","gradWeight","bias","gradBias") -- this deletes and replaces them
torch.save(paths.concat("photos.net"), model2)
