function [hidden_weights] = bp_core(input_units,hidden_units,input_weights,in,hid,out,output_delta,target,output_units,hidden_delta,hidden_prev_weights,input_prev_weights,hidden_weights)
    % bpnn_train_kernel
    %[input_units,hidden_units]  = bpnn_layerforward(input_units, hidden_units, input_weights, in, hid);
    %[hidden_units,output_units] = bpnn_layerforward(hidden_units, output_units, hidden_weights, hid, out);
    %[out_err,output_delta]      = bpnn_output_error(output_delta, target, output_units, out);
    %[hid_err,hidden_delta]      = bpnn_hidden_error(hidden_delta,hid,output_delta,out,hidden_weights,hidden_units);
    %[hidden_units,hidden_weights,hidden_prev_weights] = bpnn_adjust_weights(output_delta,out,hidden_units,hid,hidden_weights,hidden_prev_weights);
    %[input_units,input_weights,input_prev_weights]    = bpnn_adjust_weights(hidden_delta,hid,input_units,in,input_weights,input_prev_weights);
    % Propagate input forward 
    hidden_units = squash(input_units * input_weights);
    output_units = squash(hidden_units * hidden_weights);

    % Compute the error contributions backward
    output_delta = output_units .* (1-output_units) .* (target - output_units);
    hidden_delta = hidden_units .* (1-hidden_units) .* (output_delta * hidden_weights');

    % Adjust weights from errors
    ETA = 0.3;
    MOMENTUM = 0.3;
    %% The previous weights are actually the previous delta,
    %% the choice of variable name is confusing
    input_prev_weights = (ETA * input_units' * hidden_delta) + (MOMENTUM * input_prev_weights);
    hidden_prev_weights = (ETA * hidden_units' * output_delta) + (MOMENTUM * hidden_prev_weights);

    input_weights = input_weights + input_prev_weights;
    hidden_weights = hidden_weights + hidden_prev_weights;
end

function [output] = squash(input)
    output = 1.0 ./ (1.0 + exp(-input));
end
