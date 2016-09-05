function runner(layer_size)
% Example: runner(2850000);

% feature accel off;
% feature accel on;

sum_of_hidden_weights = 0;
expected_sum_of_hidden_weights = 10.855641469359398;
expected_layer_size = 2850000;
input_n = layer_size;
hidden_n= 16;
output_n= 1;
in = input_n + 1;
hid= hidden_n + 1;
out= output_n + 1;

% set rand seed
%s = RandStream('mcg16807','Seed',49734321);
%RandStream.setGlobalStream(s);
%setRandomSeed();

% bpnn_create
n_in     = input_n + 1;
n_hidden = hidden_n + 1;
n_out    = output_n + 1;
input_weights       = createMatrixRandJS( in, hid); %rand
hidden_weights      = createMatrixRandJS(hid, out); %rand
input_prev_weights  = zeros(in, hid);
hidden_prev_weights = zeros(hid, out);
target              = ones(1,out) * 0.1; %vector
hidden_units = zeros(1,hid);
output_units = zeros(1,out);
hidden_delta = zeros(1,hid);
output_delta = zeros(1,out);

% load(net)
nr = in;
input_units = zeros(1, nr);
input_units(2:nr) = createMatrixRandJS(1, nr-1);

tic
hidden_weights = bp_core(input_units,hidden_units,input_weights,...
    in,hid,out,output_delta,target,output_units,hidden_delta,...
    hidden_prev_weights,input_prev_weights,hidden_weights);
elapsedTime  = toc;

for i=2:hidden_n+1
    for j=2:output_n+1
        sum_of_hidden_weights = sum_of_hidden_weights + hidden_weights(i,j);
    end
end


disp('{');
disp('"options":');
disp(layer_size);
disp(', "time": ');
disp(elapsedTime);

% Round the least significant bits of the answer by 
% scaling up the sum proportionally to the layer_size 
% and flooring the output, adjust by a given factor
% to take the differences between implementations into account
ADJUST = 0.1;
disp(', "output":');
disp(int32(floor((sum_of_hidden_weights*layer_size*ADJUST))));
disp('}');
end
