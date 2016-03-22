require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'image'
require 'pl'
require 'paths'

----------------------------------------------------------------------
-- parse command-line options
--
--
local opt = lapp[[
   -r,--learningRate  (default 0.1)        learning rate, for SGD only
   -b,--batchSize     (default 2)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   --cl                                     enable open cl
   --cuda                                   enable cuda
   --maxEpochs        (default 15)          max epochs
]]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(4)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

require('networks')

-- initialise model etc
criterion = nn.MSECriterion()
model = get_network_1()

print("get parameters")
parameters,gradParameters = model:getParameters()


if opt.cuda then
    require 'cutorch'
    require 'cunn'
    --torch.setdefaulttensortype('torch.CudaTensor')

    criterion = criterion:cuda()
    model = model:cuda()

    print("get parameters")
    parameters,gradParameters = model:getParameters()
end

print('<photo> using model:')
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

-- TODO: change to enable loading from hdd

local trainingSet = generateDataset("raw/train/", "processed/train/")
local validationSet = generateDataset("raw/val/", "processed/val/")
local testSet = generateDataset("raw/test/", "processed/test/")

--if opt.cuda then -- currently working because small sets
--    trainingSet = trainingSet:cuda()
--    validationSet = validationSet:cuda()
--    testSet = testSet:cuda()
--end

print("dataset generated")
batchsize = opt.batchSize;
local inputs = torch.Tensor(batchsize,3,240,160)
local targets = torch.Tensor(batchsize,3,240,160)
if opt.cuda then
    inputs = inputs:cuda()
    targets = targets:cuda()
end

local train_error = 0
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

        local k = 1
        for i = t,math.min(t+batchsize-1,trainingSet:size()[1]) do
            -- load new sample
            local sample = trainingSet[i]
            if opt.cuda then
                input = sample[1]:clone():cuda()
                target = sample[2]:clone():cuda()
            else
                input = sample[1]:clone()
                target = sample[2]:clone()
            end
            inputs[k] = input
            targets[k] = target
            k = k + 1
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
            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            train_error = train_error + f
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
    print("<trainer> time for one epoch " .. time .. "s")
    time = time / trainingSet:size()[1]
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')
    print("<trainer> training error is " .. train_error)

    -- next epoch
    epoch = epoch + 1
end

maxEpochs = 15

for c=1,maxEpochs do
    train(trainingSet, validationSet)
end

-- saving model
model2 = model:float()
model2:clearState()

torch.save(paths.concat("photos.net"), model2)
