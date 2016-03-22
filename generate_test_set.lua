require 'paths'

--- https://snipt.net/raw/90723367d21c68824f58658d8efd0b38/?nice
-- Function to retrieve console output
--
function os.capture(cmd, raw)
    local handle = assert(io.popen(cmd, 'r'))
    local output = assert(handle:read('*a'))

    handle:close()

    if raw then
        return output
    end

    output = string.gsub(
        string.gsub(
            string.gsub(output, '^%s+', ''),
            '%s+$',
            ''
        ),
        '[\n\r]+',
        ' '
    )

   return output
end


local opt = lapp[[
   -i,--path_inp       (default "raw/")          subdirectory to find input images
   -o,--path_out       (default "processed/")    subdirectory to find output images
   -t,--testratio      (default "0.1")           ratio of images used for test
   -v,--validratio     (default "0.1")           ratio of images used for validation
]]


_, num_pics = string.gsub(os.capture("ls " .. opt.path_out), "%S+", "")

num_test = math.floor(num_pics * opt.testratio)
num_valid = math.floor(num_pics * opt.validratio)

order = torch.randperm(num_pics)


i = 1
for file in paths.iterfiles(opt.path_out) do
    collectgarbage()
   if file ~= ".DS_Store" then
       if order[i] < num_test then
           os.execute("cp " .. opt.path_out .. file .. " " .. opt.path_out .. "test")
           os.execute("cp " .. opt.path_inp .. file .. " " .. opt.path_inp .. "test")
       elseif order[i] < num_test+num_valid then
           os.execute("cp " .. opt.path_out .. file .. " " .. opt.path_out .. "val")
           os.execute("cp " .. opt.path_inp .. file .. " " .. opt.path_inp .. "val")
       else
           os.execute("cp " .. opt.path_out .. file .. " " .. opt.path_out .. "train")
           os.execute("cp " .. opt.path_inp .. file .. " " .. opt.path_inp .. "train")
       end
       i = i+1
       xlua.progress(i, num_pics)
   end
end
