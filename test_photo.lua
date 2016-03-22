--require 'torch'
require 'nn'
require 'nngraph'
require 'image'
require 'paths'
require 'torch'

model = torch.load(paths.concat("photos_1.net"))
print("model loaded")
--print(model:listModules())
--w1 = image.window()
--w2 = image.window()
--w3 = image.window()
for file in paths.iterfiles("../photo_postprocessing/processed") do
    collectgarbage()
   if file ~= ".DS_Store" then
       print(file)
       local tmp_inp = image.load(paths.concat("../photo_postprocessing/raw", file))
       local tmp_out = image.load(paths.concat("../photo_postprocessing/processed", file))
       if tmp_inp:size()[2] ~= 240 then
           tmp_inp = tmp_inp:resize(1,3,160,240);
           tmp_out = tmp_out:resize(1,3,160,240);
           tmp_inp = tmp_inp:transpose(3,4);
           tmp_out = tmp_out:transpose(3,4);

       else
           tmp_inp = tmp_inp:resize(1,3,240,160);
           tmp_out = tmp_out:resize(1,3,240,160);
       end
       tmp = model:forward(tmp_inp:float())
       image.save(paths.concat("generated", file), tmp[1])
       --[[
       image.display{image=tmp_inp, window=w1, legend="input"}
       image.display{image=tmp, window=w2, legend="gen"}
       image.display{image=tmp_out, window=w3, legend="out"}
       ]]

       --os.execute("sleep 5")
   end
end
print("generated all pics")
